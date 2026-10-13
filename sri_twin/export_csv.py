"""Export per-episode results (results/*.npz) to CSV and write paired McNemar
contingency tables for every comparison reported in the manuscript."""
import os, glob, json
import numpy as np
from .analysis import load, compare

SKIP = {"sur_train", "sur_val"}


def export_all(out_dir="results/csv"):
    os.makedirs(out_dir, exist_ok=True)
    for f in sorted(glob.glob("results/*.npz")):
        tag = os.path.basename(f)[:-4]
        if tag in SKIP or tag.startswith("latency"):
            continue
        d = np.load(f)
        cols = [k for k in d.files if np.ndim(d[k]) == 1 and k not in ("plan_time",)]
        n = len(d["viol"])
        cols = [k for k in cols if len(d[k]) == n]
        with open(f"{out_dir}/{tag}.csv", "w") as fh:
            fh.write("episode," + ",".join(cols) + "\n")
            for i in range(n):
                fh.write(str(i) + "," + ",".join(str(float(d[k][i])) for k in cols) + "\n")
        if "plan_time" in d.files and len(d["plan_time"]):
            np.savetxt(f"{out_dir}/{tag}_plan_time_s_per_episode.csv", d["plan_time"], delimiter=",",
                       header="batched wall time per planning cycle divided by batch size (s)")


def contingency(out="results/mcnemar_contingency.json"):
    fam = {
        "main_vs_joint": ("joint", ["nominal", "traj", "inf"]),
        "ablation_vs_joint": ("joint", ["lam0.00", "abl_nomap", "abl_decoupled", "abl_adaptive"]),
        "policy_vs_others": ("policy", ["joint", "inf", "nominal", "traj"]),
        "high_regime_vs_joint": ("joint_high", ["nominal_high", "traj_high", "inf_high", "lam0.00_high"]),
    }
    for v in ["puncture", "compliance", "saccade", "heavydelay", "driftdelay", "dropout", "shift"]:
        fam[f"ood_{v}"] = (f"joint_ood_{v}", [f"nominal_ood_{v}", f"inf_ood_{v}"])
    res = {}
    for name, (ref, cmps) in fam.items():
        r = load(ref); out_f = {}
        for c in cmps:
            m = load(c); n = min(len(r["viol"]), len(m["viol"]))
            for key in ["viol", "success"]:
                a = r[key][:n].astype(bool); b = m[key][:n].astype(bool)
                out_f[f"{c}:{key}"] = dict(n=n, both=int((a & b).sum()), ref_only=int((a & ~b).sum()),
                                           cmp_only=int((~a & b).sum()), neither=int((~a & ~b).sum()))
        out_f["tests_holm"] = compare(r, {c: load(c) for c in cmps})
        res[f"{name} (reference: {ref})"] = out_f
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    export_all(); contingency()
