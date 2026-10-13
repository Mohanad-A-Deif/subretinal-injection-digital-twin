"""Train the block surrogate (learning curve over data size) and measure planning
fidelity on held-out planning problems (validation seeds)."""
import time, pickle, sys
import numpy as np
from . import params as PR
from . import episode as E
from .surrogate import BlockSurrogate, OUT_NAMES
from .controller import rollout_twin, cvar, W


def record_problems(seed=31_000, B=20, every=10, path="results/logs/plan_problems.pkl"):
    probs = []
    orig = E.make_planner

    def mk(cfg, B_, rng, surrogate=None):
        p = orig(cfg, B_, rng, surrogate)
        def col(x, th, mot, U, ctx, t0, n_plan):
            if n_plan % every == 0:
                probs.append(dict(x=x.copy(), th={k: np.array(v).copy() for k, v in th.items()},
                                  mot={k: np.array(v).copy() for k, v in mot.items()}, U=U.copy(),
                                  ctx={k: np.array(v).copy() for k, v in ctx.items() if not isinstance(v, bool)}, t0=t0))
        p.collector = col
        return p
    E.make_planner = mk
    E.run_batch(dict(method="mpc", mode="joint", lam=0.75), seed, B)
    E.make_planner = orig
    with open(path, "wb") as f:
        pickle.dump(probs, f)
    return probs


def total_cost(J, R, lam=0.75, alpha=0.9):
    return J.mean(-1) + W["viol"] * R.mean(-1) + lam * W["viol"] * cvar(R, alpha)


def fidelity(sur, probs):
    from scipy.stats import spearmanr
    rho, top1, gap, t_tw, t_su = [], [], [], [], []
    for pr in probs:
        t0 = time.perf_counter()
        Jt, Rt, _ = rollout_twin(pr["x"], pr["th"], pr["mot"], pr["U"], pr["ctx"], pr["t0"])
        t_tw.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        Js, Rs, _ = sur.rollout(pr["x"], pr["th"], pr["mot"], pr["U"], pr["ctx"], pr["t0"])
        t_su.append(time.perf_counter() - t0)
        ct = total_cost(Jt, Rt); cs = total_cost(Js, Rs)
        for b in range(ct.shape[0]):
            r = spearmanr(ct[b], cs[b]).correlation
            rho.append(r if np.isfinite(r) else 0.0)
            it = np.argmin(ct[b]); i_s = np.argmin(cs[b])
            top1.append(float(it == i_s))
            gap.append((ct[b, i_s] - ct[b, it]) / max(abs(ct[b, it]), 1e-9))
    return dict(spearman=np.array(rho), top1=np.array(top1), gap=np.array(gap),
                t_twin=np.array(t_tw), t_sur=np.array(t_su))


def main():
    tr = np.load("results/sur_train.npz"); va = np.load("results/sur_val.npz")
    X, Y = tr["X"].astype(np.float64), tr["Y"].astype(np.float64)
    Xv, Yv = va["X"].astype(np.float64), va["Y"].astype(np.float64)
    gen_time = float(tr["gen_time"])
    try:
        probs = pickle.load(open("results/logs/plan_problems.pkl", "rb"))
    except FileNotFoundError:
        probs = record_problems()
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    out = []
    for frac in [0.1, 0.25, 0.5, 1.0]:
        n = int(frac * len(X)); idx = perm[:n]
        sur = BlockSurrogate()
        t0 = time.time()
        vl = sur.fit(X[idx], Y[idx], Xv, Yv, epochs=200, patience=20, max_minutes=float(sys.argv[1]) if len(sys.argv) > 1 else 8.0, verbose=False)
        pred = sur.predict(Xv)
        nrmse = np.sqrt(((pred - Yv) ** 2).mean(0)) / (Yv.std(0) + 1e-9)
        fid = fidelity(sur, probs)
        rec = dict(frac=frac, n=n, val_mse=vl, train_time=sur.train_time, epochs=sur.epochs_run,
                   nrmse=dict(zip(OUT_NAMES, nrmse)), spearman=float(np.mean(fid["spearman"])),
                   top1=float(np.mean(fid["top1"])), gap_med=float(np.median(fid["gap"])), gap_mean=float(np.mean(fid["gap"])),
                   gen_time_full=gen_time, t_twin=float(np.mean(fid["t_twin"])), t_sur=float(np.mean(fid["t_sur"])))
        print(rec, flush=True)
        out.append(rec)
        sur.save(f"results/models/surrogate_{frac}.pt")
    pickle.dump(out, open("results/logs/surrogate_learning_curve.pkl", "wb"))


if __name__ == "__main__":
    main()
