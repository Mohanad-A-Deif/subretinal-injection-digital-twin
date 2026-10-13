"""Tuning runs on the tuning seed partition (10_000+). Same interface as run_exp.
Experiment driver: runs named configurations on held-out test seeds and stores
per-episode metrics (paired across methods through common random numbers)."""
import sys, json, time, os, argparse
import numpy as np
from .episode import run_batch

TEST_SEED0 = 10_000  # tuning partition (used only for weight selection; results in results_tuning/)

BASE = dict(method="mpc", mode="joint", lam=0.75, est_mode="fixed", posterior=True, inner="twin")

CONFIGS = {
    # main comparison
    "nominal": dict(method="nominal"),
    "traj": dict(BASE, mode="traj"),
    "inf": dict(BASE, mode="inf"),
    "joint": dict(BASE),
    # risk-weight sweep
    "lam0.00": dict(BASE, lam=0.0), "lam0.25": dict(BASE, lam=0.25), "lam0.50": dict(BASE, lam=0.5),
    "lam1.00": dict(BASE, lam=1.0), "lam1.50": dict(BASE, lam=1.5),
    # ablations
    "abl_nomap": dict(BASE, est_mode="frozen", posterior=False),
    "abl_decoupled": dict(BASE, mode="decoupled"),
    "abl_adaptive": dict(BASE, est_mode="adaptive"),
    "joint_safety": dict(BASE, safety=True),
    "joint_gated": dict(BASE, est_mode="gated"),
}

VARIANTS = {
    "puncture": dict(contact="puncture"),
    "compliance": dict(bleb="compliance"),
    "saccade": dict(saccades=True),
    "heavydelay": dict(delay_model="heavy"),
    "driftdelay": dict(delay_model="drift"),
    "dropout": dict(dropout=0.10),
    "shift": dict(shift=True),
}

SENSING = {
    "noise0.5": dict(noise_mult=0.5), "noise2": dict(noise_mult=2.0), "noise4": dict(noise_mult=4.0),
    "biasF": dict(bias_F=1.0), "biasD": dict(bias_D=10.0),
}


def run(name, cfg, n_batches, B, regime="medium", variant=None, sensing=None, tag=None, surrogate=None,
        policy=None, log_est=False):
    tag = tag or name
    out_path = f"results_tuning/{tag}.npz"
    if os.path.exists(out_path):
        print("skip", tag); return
    allm = {}; pt = []; ninf = []; est = []
    t0 = time.time()
    for i in range(n_batches):
        o = run_batch(cfg, TEST_SEED0 + i, B, regime=regime, variant=variant, sensing=sensing,
                      surrogate=surrogate, policy=policy, log_est=log_est)
        for k, v in o["metrics"].items():
            allm.setdefault(k, []).append(np.asarray(v))
        pt.append(o["plan_time"])
        if "n_infl" in o:
            ninf.append(o["n_infl"])
        if log_est:
            est.append(o["est"])
    res = {k: np.concatenate(v) for k, v in allm.items()}
    res["plan_time"] = np.concatenate(pt) if len(pt) and sum(len(p) for p in pt) else np.zeros(0)
    if ninf:
        res["n_infl"] = np.concatenate(ninf)
    res["wall"] = np.array(time.time() - t0)
    np.savez_compressed(out_path, **res)
    if log_est:
        import pickle
        with open(f"results_tuning/{tag}_est.pkl", "wb") as f:
            pickle.dump(est, f)
    m = res
    print(f"{tag:22s} n={len(m['viol'])} VR={m['viol'].mean():.3f} succ={m['success'].mean():.3f} "
          f"RR={np.mean(m['RR']):.3f} IE={np.mean(m['IE']):.3f} Fmax={np.mean(m['Fmax']):.2f} wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--nb", type=int, default=8)
    ap.add_argument("--B", type=int, default=50)
    ap.add_argument("--regime", default="medium")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--sensing", default=None)
    a = ap.parse_args()
    for n in a.names:
        tag = a.prefix + n + ("" if a.regime == "medium" else "_" + a.regime)
        if a.variant:
            tag += "_ood_" + a.variant
        if a.sensing:
            tag += "_sens_" + a.sensing
        run(n, CONFIGS[n], a.nb, a.B, regime=a.regime, tag=tag,
            variant=VARIANTS.get(a.variant), sensing=SENSING.get(a.sensing))
