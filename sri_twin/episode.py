"""Closed-loop episode runner (vectorized over a batch of held-out scenarios)."""
import time
import numpy as np
from . import params as PR
from .plant import Plant
from .estimator import UKF, ZMN, ZMS, ZMDN, ZV, ZA, ZP, ZBF, ZLK, ZLA, ZLB, ZLT, ZH
from .controller import CEMPlanner, NominalController, q_nominal, tracking_law
from .estimator import ZMDS
from .core import IP, IES, IV, IA, IPR

K_ANC = 4.0  # ancillary tracking gain of the 20 ms layer (1/s)


def make_planner(cfg, B, rng, surrogate=None):
    return CEMPlanner(B, rng, mode=cfg.get("mode", "joint"), lam=cfg.get("lam", 0.75),
                      alpha=cfg.get("alpha", 0.9), M=cfg.get("M", 16), P=cfg.get("P", 32),
                      iters=cfg.get("iters", 3), n_elite=cfg.get("n_elite", 5), posterior=cfg.get("posterior", True),
                      inner=cfg.get("inner", "twin"), surrogate=surrogate,
                      refine_every=cfg.get("refine_every", 0), dt_sub=cfg.get("dt_sub", PR.DT_SUB))


def safety_filter(u, ukf, enc, y):
    """20 ms safety layer (optional): blocks advancing toward the RPE margin,
    advancing under high force, and infusion near the pressure limit."""
    p_hat = enc["n_enc"] - ukf.z[:, ZMN]
    margin = ukf.z[:, ZH] - p_hat
    sd = np.sqrt(np.maximum(ukf.P[:, ZMN, ZMN] + ukf.P[:, ZH, ZH] - 2 * ukf.P[:, ZMN, ZH], 0))
    lim_d = margin - PR.DELTA_RPE - 2.0 * sd - 10.0
    u = u.copy()
    un_max = np.maximum(lim_d / 0.08, -300.0)  # reach margin no sooner than in 80 ms
    u[:, 1] = np.minimum(u[:, 1], un_max + ukf.z[:, ZMDN])
    F_hat = y["F"] - ukf.z[:, ZBF]
    hiF = np.nan_to_num(F_hat, nan=0.0) > 0.85 * PR.F_SAFE
    u[hiF, 1] = np.minimum(u[hiF, 1], -50.0)
    hiP = ukf.z[:, ZP] > 0.92 * PR.P_MAX
    u[hiP, 3] = 0.0
    return u


def run_batch(cfg, seed, B, regime="medium", variant=None, sensing=None, surrogate=None,
              policy=None, log_est=False, record=False):
    """cfg: dict(method='nominal'|'mpc'|'policy', mode, lam, est_mode, posterior, inner,
    refine_every, safety, ancillary)."""
    rng_th = np.random.default_rng(seed)
    theta = PR.sample_theta(rng_th, B, regime)
    rng = np.random.default_rng(seed + 7919)
    plant = Plant(theta, rng, variant=variant, sensing=sensing)
    y = plant.observe()
    for _ in range(20):  # initial readings: repeat acquisition for dropped samples
        bad = {k: ~np.isfinite(v) for k, v in y.items()}
        if not any(b.any() for b in bad.values()):
            break
        y2 = plant.observe()
        y = {k: np.where(bad[k], y2[k], v) for k, v in y.items()}
    nom = NominalController(y)
    method = cfg["method"]
    ukf = None
    if method != "nominal" or log_est:
        ukf = UKF(B, y, mode=cfg.get("est_mode", "fixed"))
    planner = make_planner(cfg, B, np.random.default_rng(seed + 104729), surrogate) if method == "mpc" else None
    if method == "policy":
        policy.reset(B)
    nsteps = int(round(PR.T_MAX / PR.DT_CTRL))
    u_blk = np.zeros((B, 4)); q_prev = np.zeros(B)
    plan_times = []; est_log = []; rec = []
    enc_prev = y
    for k in range(nsteps):
        t = k * PR.DT_CTRL
        if k > 0 and ukf is not None:
            ukf.predict(enc_prev, q_prev)
            ukf.update(y)
        elif ukf is not None:
            ukf.update(y)
        if method == "nominal":
            u = nom.command(t, y)
        elif method == "policy":
            t0 = time.perf_counter()
            u = policy.act(t, ukf, y)
            plan_times.append((time.perf_counter() - t0) / B)
        else:
            if k % PR.BLOCK == 0:
                t0 = time.perf_counter()
                ref = planner.plan(t, ukf, y, nom)
                plan_times.append((time.perf_counter() - t0) / B)
            p_hat = y["n_enc"] - ukf.z[:, ZMN]; es_hat = y["s_enc"] - ukf.z[:, ZMS]
            u = tracking_law(ref, p_hat, es_hat, y["phi_enc"], ukf.z[:, ZMDN], ukf.z[:, ZMDS])
            if planner.mode == "inf":
                un, us, uphi = nom.traj_cmd(y)
                u[:, 0], u[:, 1], u[:, 2] = us, un, uphi
            if planner.mode == "traj":
                u[:, 3] = q_nominal(t)
        if cfg.get("safety", False):
            u = safety_filter(u, ukf, y, y)
        u = np.clip(u, PR.U_LO, PR.U_HI)
        if log_est and ukf is not None:
            tr = plant.truth()
            est_log.append(dict(t=t, p_err=(y["n_enc"] - ukf.z[:, ZMN]) - tr["x"][:, IP],
                                V_err=ukf.z[:, ZV] - tr["x"][:, IV], P_err=ukf.z[:, ZP] - tr["x"][:, IPR],
                                kk_err=np.exp(ukf.z[:, ZLK]) / tr["kk"] - 1,
                                ar_err=PR.NOM["alpha_r"] * np.exp(ukf.z[:, ZLA]) / tr["alpha_r"] - 1,
                                bq_err=PR.NOM["beta_q"] * np.exp(ukf.z[:, ZLB]) / tr["beta_q"] - 1,
                                h_err=ukf.z[:, ZH] - tr["h"],
                                pt_err=PR.NOM["P_tract"] * np.exp(ukf.z[:, ZLT]) / tr["P_tract"] - 1, bF_err=ukf.z[:, ZBF] - tr["bF"],
                                kk_sd=np.sqrt(ukf.P[:, ZLK, ZLK]), h_sd=np.sqrt(ukf.P[:, ZH, ZH]),
                                p_sd=np.sqrt(np.maximum(ukf.P[:, ZMN, ZMN], 0)),
                                done=plant.done.copy(), contact=tr["x"][:, IP] > 0))
        if record:
            tr = plant.truth()
            rec.append(dict(t=t, x=tr["x"], u=u.copy(), mn=tr["mn"], h=tr["h"], ref=(ref.copy() if method == "mpc" else None)))
        plant.step(u)
        if planner is not None:
            planner.Qin = planner.Qin + PR.DT_CTRL * u[:, 3] * (~plant.done)
        enc_prev = y
        q_prev = u[:, 3]
        y = plant.observe()
        if plant.done.all():
            break
    m = plant.metrics()
    out = dict(metrics=m, plan_time=np.array(plan_times))
    if ukf is not None:
        out["n_infl"] = ukf.n_infl
    if log_est:
        out["est"] = est_log
    if record:
        out["rec"] = rec
    return out
