"""Practical identifiability of the online-identified parameter block via the
empirical Fisher information matrix along closed-loop trajectories.

For each test episode, the applied command sequence of the proposed controller
is replayed on noise-free copies of the full twin with perturbed parameters
(identical stochastic motion/drift realizations), and output sensitivities are
obtained by central differences."""
import numpy as np
from . import params as PR
from .plant import Plant
from .episode import run_batch

NAMES = ["log_kk", "log_alpha_r", "log_beta_q", "log_P_tract", "h"]
STEP = [0.02, 0.02, 0.02, 0.02, 1.0]


def perturb(theta, j, s):
    th = {k: np.array(v, dtype=float).copy() for k, v in theta.items()}
    if j == 0: th["kk"] *= np.exp(s)
    elif j == 1: th["alpha_r"] *= np.exp(s)
    elif j == 2: th["beta_q"] *= np.exp(s)
    elif j == 3: th["P_tract"] *= np.exp(s)
    else: th["h"] += s
    return th


def replay(theta, U, seed):
    pl = Plant(theta, np.random.default_rng(seed + 7919), sensing=dict(noise_mult=0.0))
    Ys = []; alive = []
    y = pl.observe()
    for k in range(U.shape[0]):
        Ys.append(np.stack([y["F"], y["D"], y["P"], y["S"], y["V"]], -1)); alive.append(~pl.done.copy())
        pl.step(U[k])
        y = pl.observe()
    return np.stack(Ys, 0), np.stack(alive, 0)


def fisher(seed, B, cfg):
    o = run_batch(cfg, seed, B, record=True)
    U = np.stack([r["u"] for r in o["rec"]], 0)  # (K,B,4)
    theta = PR.sample_theta(np.random.default_rng(seed), B, "medium")
    Y0, al0 = replay(theta, U, seed)
    K = Y0.shape[0]
    Jac = np.zeros((K, B, 5, 5))
    alive = al0.copy()
    for j in range(5):
        Yp, ap = replay(perturb(theta, j, STEP[j]), U, seed)
        Ym, am = replay(perturb(theta, j, -STEP[j]), U, seed)
        Jac[..., j] = (Yp - Ym) / (2 * STEP[j])
        alive &= ap & am
    sig = np.array([PR.SENS["sig_F"], PR.SENS["sig_D"], PR.SENS["sig_P"], PR.SENS["sig_S"], 0.0])
    Rinv = np.zeros((K, B, 5))
    Rinv[..., :4] = 1.0 / sig[:4] ** 2
    Rinv[..., 4] = 1.0 / (PR.SENS["sig_V0"] + PR.SENS["sig_Vrel"] * np.maximum(Y0[..., 4], 0)) ** 2
    Rinv *= alive[..., None]
    t = np.arange(K) * PR.DT_CTRL
    return Jac, Rinv, t, o["metrics"]


def summarize(Jac, Rinv, t, t_upto=None):
    msk = np.ones(len(t), bool) if t_upto is None else (t <= t_upto)
    F = np.einsum("kbmi,kbm,kbmj->bij", Jac[msk], Rinv[msk], Jac[msk])
    ev = np.linalg.eigvalsh(F)
    cond = ev[:, -1] / np.maximum(ev[:, 0], 1e-300)
    Finv = np.linalg.pinv(F)
    crlb = np.sqrt(np.maximum(np.einsum("bii->bi", Finv), 0))
    d = np.sqrt(np.maximum(np.einsum("bii->bi", Finv), 1e-300))
    corr = Finv / (d[:, :, None] * d[:, None, :])
    # Fisher information per channel (diagonal), to show which sensor informs which parameter
    per_ch = np.einsum("kbmi,kbm,kbmi->bmi", Jac[msk], Rinv[msk], Jac[msk])
    return dict(F=F, ev=ev, cond=cond, crlb=crlb, corr=corr, per_ch=per_ch)
