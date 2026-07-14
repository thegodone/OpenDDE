"""Ubiquitin accuracy: OpenFold3 vs OpenDDE, every saved sample vs experimental 1UBQ.
Kabsch Ca-RMSD + TM-score (L=76) + mean pLDDT from confidence JSONs."""
import glob, json, os
import numpy as np

EXP = "/Users/tgg/Github/OpenDDE/1ubq.cif"


def ca_from_cif(path):
    cas = {}
    hdr, data = [], False
    for ln in open(path):
        s = ln.strip()
        if s.startswith("_atom_site."):
            hdr.append(s); continue
        if hdr and (s.startswith("ATOM") or s.startswith("HETATM")):
            data = True
            p = s.split()
            rec = {h.split(".")[1]: p[i] for i, h in enumerate(hdr) if i < len(p)}
            if rec.get("group_PDB") == "ATOM" and rec.get("label_atom_id", "").strip('"') == "CA":
                try:
                    rn = int(rec.get("label_seq_id") or rec.get("auth_seq_id"))
                    if rn not in cas:
                        cas[rn] = (float(rec["Cartn_x"]), float(rec["Cartn_y"]), float(rec["Cartn_z"]))
                except Exception:
                    pass
        elif data and s and not s.startswith("_") and not (s.startswith("ATOM") or s.startswith("HETATM")):
            break
    return cas


def match(ref, pred):
    keys = sorted(set(ref) & set(pred))
    return (np.array([ref[k] for k in keys]), np.array([pred[k] for k in keys]))


def kabsch_rmsd_tm(P, Q, L_ref):
    """Superpose P onto Q (Kabsch, RMSD-optimal); return RMSD and TM-score(L_ref)."""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    Pr = Pc @ R.T
    di2 = ((Pr - Qc) ** 2).sum(1)
    rmsd = float(np.sqrt(di2.mean()))
    d0 = 1.24 * (L_ref - 15) ** (1 / 3) - 1.8
    tm = float((1.0 / (1.0 + di2 / d0 ** 2)).sum() / L_ref)
    return rmsd, tm


def conf_metrics(cif):
    """Find the sibling confidence JSON and return (pLDDT, pTM)."""
    d = os.path.dirname(cif); base = os.path.basename(cif)
    idx = base.split("sample_")[-1].split("_")[0].split(".")[0]
    # ODDE layout
    j = os.path.join(d, f"ubiquitin_summary_confidence_sample_{idx}.json")
    if os.path.exists(j):
        o = json.load(open(j)); return o.get("plddt", float("nan")), o.get("ptm", float("nan"))
    # OF3 layout
    stem = base.replace("_model.cif", "")
    j = os.path.join(d, f"{stem}_confidences_aggregated.json")
    if os.path.exists(j):
        o = json.load(open(j)); return o.get("avg_plddt", float("nan")), o.get("ptm", float("nan"))
    return float("nan"), float("nan")


RUNS = [
    ("OF3  + ColabFold MSA", "/Users/tgg/Github/openfold-3-mlx/prediction_colabfold/ubiquitin/seed_42/ubiquitin_seed_42_sample_*_model.cif"),
    ("OF3  single-seq (prof)", "/Users/tgg/Github/openfold-3-mlx/prediction_prof/ubiquitin/seed_42/ubiquitin_seed_42_sample_*_model.cif"),
    ("ODDE single-seq N=5",  "/Users/tgg/Github/OpenDDE/opendde_nsample5/ubiquitin/seed_65387/predictions/ubiquitin_sample_*.cif"),
    ("ODDE + dummy MSA",     "/Users/tgg/Github/OpenDDE/opendde_out_msa/ubiquitin/seed_32675/predictions/ubiquitin_sample_*.cif"),
    ("ODDE N_cycle=1 (abl)", "/Users/tgg/Github/OpenDDE/opendde_cyc1/ubiquitin/seed_62193/predictions/ubiquitin_sample_*.cif"),
]


def main():
    ref = ca_from_cif(EXP)
    L = len(ref)
    print(f"reference: experimental 1UBQ, {L} Ca (res {min(ref)}..{max(ref)})\n")
    print(f"{'run':24s} {'sample':>7s} {'RMSD_A':>7s} {'TM':>6s} {'pLDDT':>6s} {'pTM':>5s}")
    print("-" * 60)
    summary = {}
    for name, pat in RUNS:
        files = sorted(glob.glob(pat))
        if not files:
            continue
        rows = []
        for f in files:
            keys = sorted(set(ref) & set(ca_from_cif(f)))
            cf = ca_from_cif(f)
            P = np.array([cf[k] for k in keys]); Q = np.array([ref[k] for k in keys])
            rmsd, tm = kabsch_rmsd_tm(P, Q, L)
            pl, ptm = conf_metrics(f)
            rows.append((os.path.basename(f).split("sample_")[-1].split("_")[0].split(".")[0], rmsd, tm, pl, ptm))
        rows.sort(key=lambda r: r[1])
        for i, (sm, rmsd, tm, pl, ptm) in enumerate(rows):
            mark = "  <- best" if i == 0 else ""
            print(f"{name if i==0 else '':24s} {sm:>7s} {rmsd:>7.3f} {tm:>6.3f} {pl:>6.1f} {ptm:>5.2f}{mark}")
        rr = [r[1] for r in rows]; tt = [r[2] for r in rows]; pls = [r[3] for r in rows]
        summary[name] = (min(rr), np.mean(rr), max(tt), np.mean(pls), len(rows))
        print()
    print("=" * 60)
    print(f"{'SUMMARY':24s} {'best_RMSD':>9s} {'mean_RMSD':>9s} {'best_TM':>8s} {'plddt':>6s} {'n':>3s}")
    for name, (br, mr, bt, mpl, n) in summary.items():
        print(f"{name:24s} {br:>9.3f} {mr:>9.3f} {bt:>8.3f} {mpl:>6.1f} {n:>3d}")


if __name__ == "__main__":
    main()
