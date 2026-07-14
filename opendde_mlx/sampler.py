# SPDX-License-Identifier: Apache-2.0
# MLX port of opendde/model/generator.py :: InferenceNoiseScheduler + sample_diffusion
# (AF3 Alg.18 EDM predictor-corrector, Euler step; delta uses x_noisy per the torch code).
#
# The sampler math is isolated from RNG so it can be gated bit-for-bit against torch:
# init noise, per-step augmentation rotation+translation, and per-step corrector noise
# can all be *injected* as numpy/mlx arrays. When not injected, mx.random is used.
from typing import Any, Callable, List, Optional

import math
import numpy as np
import mlx.core as mx


# ----------------------------------------------------------------------------- schedule
def noise_schedule(
    N_step: int = 200,
    s_max: float = 160.0,
    s_min: float = 4e-4,
    rho: float = 7.0,
    sigma_data: float = 16.0,
) -> mx.array:
    """Karras sigma schedule, port of InferenceNoiseScheduler.__call__.

    Returns [N_step+1] with the last entry forced to 0.
    """
    step_size = 1.0 / N_step
    step_indices = mx.arange(N_step + 1, dtype=mx.float32)
    a = s_max ** (1.0 / rho)
    b = s_min ** (1.0 / rho)
    t = sigma_data * (a + step_indices * step_size * (b - a)) ** rho
    # replace last time step by 0 (t_N = 0)
    t = mx.concatenate([t[:-1], mx.zeros((1,), dtype=t.dtype)])
    return t


class InferenceNoiseScheduler:
    def __init__(self, s_max=160.0, s_min=4e-4, rho=7.0, sigma_data=16.0):
        self.s_max = s_max
        self.s_min = s_min
        self.rho = rho
        self.sigma_data = sigma_data

    def __call__(self, N_step: int = 200) -> mx.array:
        return noise_schedule(
            N_step, self.s_max, self.s_min, self.rho, self.sigma_data
        )


# ------------------------------------------------------------------- augmentation math
def _rot_vec_mul(r: mx.array, t: mx.array) -> mx.array:
    """r [..., 3, 3], t [..., 3] -> [..., 3]. Written out (matches torch rot_vec_mul)."""
    x = t[..., 0]
    y = t[..., 1]
    z = t[..., 2]
    ox = r[..., 0, 0] * x + r[..., 0, 1] * y + r[..., 0, 2] * z
    oy = r[..., 1, 0] * x + r[..., 1, 1] * y + r[..., 1, 2] * z
    oz = r[..., 2, 0] * x + r[..., 2, 1] * y + r[..., 2, 2] * z
    return mx.stack([ox, oy, oz], axis=-1)


def centre_random_augmentation(
    x_input_coords: mx.array,
    rot: mx.array,
    trans: mx.array,
    s_trans: float = 1.0,
) -> mx.array:
    """Port of Alg.19 (no mask, N_sample=1, centre + rotate + translate).

    Args:
        x_input_coords: [..., N_atom, 3]
        rot:   [..., 1, 3, 3]  injected rotation matrices (one per sample)
        trans: [..., 1, 3]     injected translation vectors (pre-scale)
    Returns:
        [..., N_atom, 3] (the N_sample=1 axis is squeezed back out)
    """
    N_atom = x_input_coords.shape[-2]
    # move to origin
    x = x_input_coords - mx.mean(x_input_coords, axis=-2, keepdims=True)
    # expand to [..., 1, N_atom, 3]
    x = mx.expand_dims(x, axis=-3)
    # expand rot to [..., 1, N_atom, 3, 3]
    rot_e = mx.expand_dims(rot, axis=-3)
    rot_e = mx.broadcast_to(rot_e, x.shape[:-1] + (3, 3))
    x_aug = _rot_vec_mul(rot_e, x) + (s_trans * trans)[..., None, :]
    # squeeze dim=-3
    return mx.squeeze(x_aug, axis=-3)


# ------------------------------------------------------------------------- sampler loop
def sample_diffusion(
    denoise_net: Callable,
    schedule: mx.array,
    x_shape: tuple,
    *,
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
    injected_init_noise: Optional[mx.array] = None,
    injected_rots: Optional[List[mx.array]] = None,
    injected_trans: Optional[List[mx.array]] = None,
    injected_step_noise: Optional[List[mx.array]] = None,
    denoise_context: Optional[dict] = None,
) -> mx.array:
    """AF3 Alg.18 predictor-corrector (Euler) sampling loop, MLX port.

    denoise_net(x_noisy, t_hat, **denoise_context) -> x_denoised, same shape as x_noisy.
    x_shape is (..., n_sample, N_atom, 3). schedule is [N_iter] (t_0 .. t_N=0).
    Injected arrays (if given) fix all randomness for parity gating.
    """
    if denoise_context is None:
        denoise_context = {}
    batch_sample_shape = x_shape[:-2]  # (..., n_sample)

    # init noise: schedule[0] * randn
    if injected_init_noise is not None:
        base = mx.array(injected_init_noise)
    else:
        base = mx.random.normal(shape=x_shape)
    x_l = schedule[0] * base

    num_steps = schedule.shape[0] - 1
    for step_i in range(num_steps):
        c_tau_last = schedule[step_i]
        c_tau = schedule[step_i + 1]

        # --- centre_random_augmentation (rot + trans injected/sampled) ---
        if injected_rots is not None:
            rot = mx.array(injected_rots[step_i])
        else:
            rot = _random_rotation(batch_sample_shape)
        if injected_trans is not None:
            trans = mx.array(injected_trans[step_i])
        else:
            trans = mx.random.normal(shape=batch_sample_shape + (3,))
        # rot -> [..., 1, 3, 3]; trans -> [..., 1, 3]
        rot = mx.expand_dims(rot, axis=-3)
        trans = mx.expand_dims(trans, axis=-2)
        x_l = centre_random_augmentation(x_l, rot=rot, trans=trans)

        # --- 1. add noise: x_{c_tau_last} -> x_{t_hat} ---
        gamma = float(gamma0) if float(c_tau) > gamma_min else 0.0
        t_hat = c_tau_last * (gamma + 1.0)
        delta_noise_level = mx.sqrt(t_hat ** 2 - c_tau_last ** 2)

        if injected_step_noise is not None:
            step_noise = mx.array(injected_step_noise[step_i])
        else:
            step_noise = mx.random.normal(shape=x_l.shape)
        x_noisy = x_l + noise_scale_lambda * delta_noise_level * step_noise

        # --- 2. denoise x_{t_hat} -> x_{c_tau}, Euler step ---
        t_hat_b = mx.broadcast_to(mx.reshape(t_hat, (1,) * len(batch_sample_shape)),
                                  batch_sample_shape)
        x_denoised = denoise_net(x_noisy, t_hat_b, **denoise_context)

        delta = (x_noisy - x_denoised) / t_hat_b[..., None, None]
        dt = c_tau - t_hat
        x_l = x_noisy + step_scale_eta * dt * delta

    return x_l


def _random_rotation(batch_sample_shape: tuple) -> mx.array:
    """Uniform random rotation matrices via scipy (matches torch path)."""
    from scipy.spatial.transform import Rotation

    n = int(np.prod(batch_sample_shape)) if len(batch_sample_shape) else 1
    m = Rotation.random(num=n).as_matrix().astype(np.float32)
    return mx.array(m.reshape(batch_sample_shape + (3, 3)))
