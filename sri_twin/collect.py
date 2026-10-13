"""Generate surrogate training data from full-twin block transitions along
closed-loop receding-horizon trajectories (training/validation seeds only)."""
import sys, time, numpy as np
from . import params as PR
from . import episode as E
from .surrogate import block_inputs, twin_block, N_IN, N_OUT
from .core import NX, IP, IES

class Collector:
    def __init__(self, rng, n_cand=4, n_scen=4, every=4):
        self.rng = rng; self.nc = n_cand; self.ns = n_scen; self.every = every
        self.X = []; self.Y = []
    def __call__(self, x, th, mot, U, ctx, t0, n_plan):
        if n_plan % self.every:
            return
        B, P_, N, _ = U.shape; M = x.shape[1]
        rng = self.rng
        ci = rng.choice(P_, self.nc, replace=False); si = rng.choice(M, self.ns, replace=False)
        Uc = U[:, ci].copy()
        # randomized controls for coverage: perturb half of the selected candidates
        Uc[:, self.nc // 2:] += rng.normal(0, 1, Uc[:, self.nc // 2:].shape) * np.array([15.0, 25.0, 0.08, 1.5])
        Uc = np.clip(Uc, [-100, -60, -0.3, 0], [100, 320, 0.3, 8])
        shp = (B, self.nc, self.ns)
        S = [np.ascontiguousarray(np.broadcast_to(x[:, None, si, i], shp)).copy() for i in range(NX)]
        thb = {k: np.broadcast_to(np.asarray(v)[:, None, si], shp) for k, v in th.items()}
        motb = {k: np.asarray(v)[:, None, si] for k, v in mot.items()}
        refp = np.broadcast_to(ctx["ref_prev"][:, None, None, :], shp + (4,))
        for i in range(N):
            ref = np.broadcast_to(Uc[:, :, i][:, :, None, :], shp + (4,))
            Xi = block_inputs(S, ref, refp, motb, i * PR.DT_PLAN, thb)
            S, Yi = twin_block(S, ref, refp, motb, i * PR.DT_PLAN, thb)
            self.X.append(Xi.reshape(-1, N_IN)); self.Y.append(Yi.reshape(-1, N_OUT))
            refp = ref

def collect(seed0, n_batches, B, n_cand, n_scen, every, out):
    t0 = time.time()
    col = Collector(np.random.default_rng(seed0 + 5), n_cand, n_scen, every)
    orig = E.make_planner
    def mk(cfg, B_, rng, surrogate=None):
        p = orig(cfg, B_, rng, surrogate); p.collector = col; return p
    E.make_planner = mk
    for b in range(n_batches):
        E.run_batch(dict(method="mpc", mode="joint", lam=0.75), seed0 + b, B)
        print(" batch", b, "samples", sum(len(a) for a in col.X), f"{time.time()-t0:.0f}s", flush=True)
    E.make_planner = orig
    X = np.concatenate(col.X).astype(np.float32); Y = np.concatenate(col.Y).astype(np.float32)
    np.savez_compressed(out, X=X, Y=Y, gen_time=time.time() - t0, n_episodes=n_batches * B)
    print("saved", out, X.shape, f"{time.time()-t0:.0f}s")

if __name__ == "__main__":
    which = sys.argv[1]
    if which == "train":
        collect(20_000, 2, 50, 4, 4, 4, "results/sur_train.npz")
    else:
        collect(30_000, 1, 50, 2, 2, 4, "results/sur_val.npz")
