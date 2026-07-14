"""Compare ubiquitin structures: OpenFold3-MLX vs OpenDDE vs experimental 1UBQ.
Cα RMSD (after Kabsch superposition) + TM-score."""
import sys
import numpy as np
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx


def load_ca(path):
    f = pdbx.CIFFile.read(path)
    arr = pdbx.get_structure(f, model=1)
    arr = arr[struc.filter_amino_acids(arr)]
    # first polypeptide chain
    ch = arr.chain_id[0]
    arr = arr[arr.chain_id == ch]
    ca = arr[arr.atom_name == "CA"]
    order = np.argsort(ca.res_id)
    return ca.coord[order], ca.res_id[order]


def kabsch_rmsd(P, Q):
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    Pr = Pc @ R.T
    rmsd = np.sqrt(((Pr - Qc) ** 2).sum(1).mean())
    di = np.sqrt(((Pr - Qc) ** 2).sum(1))
    return rmsd, di


def tm_score(di, L):
    d0 = max(0.5, 1.24 * (L - 15) ** (1 / 3) - 1.8)
    return (1.0 / (1.0 + (di / d0) ** 2)).sum() / L


def compare(name, pa, pb):
    A, ra = load_ca(pa); B, rb = load_ca(pb)
    L = min(len(A), len(B))
    A, B = A[:L], B[:L]
    rmsd, di = kabsch_rmsd(A, B)
    tm = tm_score(di, L)
    print(f"  {name:38s} L={L:3d}  CA-RMSD={rmsd:6.2f} A   TM-score={tm:.3f}")


if __name__ == "__main__":
    OF3 = sys.argv[1]
    ODE = sys.argv[2]
    REF = "1ubq.cif"
    print("Ubiquitin structure comparison (Cα, Kabsch-superposed):")
    compare("OpenFold3-MLX  vs  OpenDDE", OF3, ODE)
    compare("OpenFold3-MLX  vs  1UBQ (exp)", OF3, REF)
    compare("OpenDDE        vs  1UBQ (exp)", ODE, REF)
