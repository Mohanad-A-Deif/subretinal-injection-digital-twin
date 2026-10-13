"""Risk-aware scenario MPC (cross-entropy solver), baselines and the 20 ms
execution layer.

Needle motion is parameterized by block-wise depth and lateral set-points
(p_ref, es_ref) that the 20 ms layer converts into velocity commands through a
motion-compensating tracking law u_n = mdn_hat + K_T (p_ref - p_hat).  The
planner decides, per 100 ms block, the vector [es_ref, p_ref, u_phi, q_inj].
"""
import numpy as np
from . import params as PR
from .core import deriv, deriv_fast, NX, IP, IES, IVN, IVS, IPHI, IOM, IV, IA, IPR, ID, IQIN, IQRF
from .estimator import NZ, ZMN, ZMDN, ZMS, ZMDS, ZV, ZA, ZP, ZBF, ZLK, ZLA, ZLB, ZLT, ZH, TAU_NOM, ID_IDX

OTHER = ["mu", "alpha_l", "P_l", "eta_A", "lam_A", "tau_P", "beta_A", "beta_V"]

W = dict(n=1e-3, s=1e-3, F=0.02, D=5.0, q=0.01, R=1.0, phi=0.05, du=2e-4, dq=0.05,
         V=0.3, Vover=0.5, Pend=20.0, viol=10.0, ttg=3.0, A=0.5)
Q_SUST = 4.0                   # sustainable infusion rate used in the time-to-go term (uL/s)
K_T = 3.0                      # tracking gain of the 20 ms execution law (1/s)
K_PHI = 4.0                    # orientation set-point gain (1/s)
K_FF = 0.0                     # motion feed-forward gain
V_FB = 200.0                   # bound on the feedback (approach/retraction) speed (um/s)
K_SERVO = 4.0                  # nominal encoder servo gain (1/s)
SIG0 = np.array([8.0, 12.0, 0.05, 1.2])
D_SOFT = 5.0                   # soft depth-risk band above the hard RPE margin (um)
REF_MARGIN = 10.0              # set-point margin beyond the hard RPE margin (um)
REF_LO = np.array([-100.0, -60.0, -0.3, 0.0])
REF_HI = np.array([100.0, 320.0, 0.3, 8.0])


def v_ref(t):
    """Reference cumulative delivered-volume schedule (uL)."""
    return 1.03 * PR.V_STAR * np.clip((t - 0.8) / (PR.T_MAX - 2.4), 0.0, 1.0)


def q_nominal(t):
    return np.where(t >= 1.5, 1.08 * PR.V_STAR / (PR.T_MAX - 1.5), 0.0)


def tracking_law(ref, p, es, phi, mdn, mds):
    """20 ms execution law: set-points -> velocity commands (vectorized)."""
    un = K_FF * mdn + np.clip(K_T * (ref[..., 1] - p), -V_FB, V_FB)
    us = K_FF * mds + np.clip(K_T * (ref[..., 0] - es), -V_FB, V_FB)
    uphi = np.clip(K_PHI * (ref[..., 2] - phi), PR.U_LO[2], PR.U_HI[2])
    return np.stack([np.clip(us, PR.U_LO[0], PR.U_HI[0]), np.clip(un, PR.U_LO[1], PR.U_HI[1]),
                     uphi, ref[..., 3]], -1)


class ScenarioSampler:
    def __init__(self, rng, M):
        self.rng = rng; self.M = M

    def sample(self, ukf, enc, use_posterior=True):
        """Scenario set per episode: initial states x (B,M,12), parameters (B,M)
        and motion forecasts, drawn from the current posterior of the estimator
        (or from the fixed prior when use_posterior=False)."""
        B = ukf.B; M = self.M; rng = self.rng
        z, P = ukf.z, ukf.P
        if not use_posterior:
            mu_id, S_id = PR.prior_id()
            z = z.copy(); P = P.copy()
            idx = ID_IDX
            z[:, idx] = mu_id
            for i in idx:
                P[:, i, :] = 0; P[:, :, i] = 0
            for j, i in enumerate(idx):
                P[:, i, i] = S_id[j, j]
        Ps = 0.5 * (P + np.swapaxes(P, -1, -2)) + 1e-6 * np.eye(NZ)
        w, V = np.linalg.eigh(Ps)
        L = V * np.sqrt(np.maximum(w, 0))[..., None, :]
        Zs = z[:, None, :] + np.einsum("bnk,bmk->bmn", L, rng.normal(0, 1, (B, M, NZ)))
        Zs[:, 0] = z  # nominal (estimate) scenario
        th = {}
        from .params import sample_theta
        prior = sample_theta(rng, B * M, "medium")
        for k in OTHER:
            th[k] = prior[k].reshape(B, M).copy()
            th[k][:, 0] = PR.NOM[k]
        th["kk"] = np.exp(np.clip(Zs[..., ZLK], -0.7, 0.7))
        th["alpha_r"] = PR.NOM["alpha_r"] * np.exp(np.clip(Zs[..., ZLA], -0.7, 0.7))
        th["beta_q"] = PR.NOM["beta_q"] * np.exp(np.clip(Zs[..., ZLB], -0.7, 0.7))
        th["h"] = np.clip(Zs[..., ZH], 120, 320)
        th["P_tract"] = PR.NOM["P_tract"] * np.exp(np.clip(Zs[..., ZLT], -0.7, 0.7))
        x = np.zeros((B, M, NX))
        x[..., IP] = enc["n_enc"][:, None] - Zs[..., ZMN]
        x[..., IES] = enc["s_enc"][:, None] - Zs[..., ZMS]
        x[..., IVN] = enc["vn_enc"][:, None]; x[..., IVS] = enc["vs_enc"][:, None]
        x[..., IPHI] = enc["phi_enc"][:, None]; x[..., IOM] = enc["om_enc"][:, None]
        x[..., IV] = np.maximum(Zs[..., ZV], 0); x[..., IA] = np.maximum(Zs[..., ZA], 0); x[..., IPR] = Zs[..., ZP]
        amp_n = rng.uniform(0, 12, (B, M)); amp_s = rng.uniform(0, 15, (B, M))
        f = rng.uniform(0.8, 1.8, (B, M)); ph = rng.uniform(0, 2 * np.pi, (B, M))
        amp_n[:, 0] = 0; amp_s[:, 0] = 0
        mot = dict(mdn0=Zs[..., ZMDN], mds0=Zs[..., ZMDS], amp_n=amp_n, amp_s=amp_s, f=f, ph=ph,
                   mn0=Zs[..., ZMN], ms0=Zs[..., ZMS])
        return x, th, mot


def motion_vel(mot, tau):
    w = 2 * np.pi * mot["f"]
    dec = np.exp(-tau / 0.25)
    mdn = mot["mdn0"] * dec + mot["amp_n"] * w * (np.cos(w * tau + mot["ph"]) - np.cos(mot["ph"]))
    mds = mot["mds0"] * dec + mot["amp_s"] * w * (np.cos(w * tau + mot["ph"]) - np.cos(mot["ph"]))
    return mdn, mds


def rollout_twin(x0, th, mot, U, ctx, t0, dt=PR.DT_SUB, bleb_terms=True, dtype=np.float64):
    """Full-twin scenario rollouts of the closed loop (planner set-points +
    20 ms execution law).  x0 (B,M,12); U (B,P,N,4) -> J, R of shape (B,P,M).
    Arrays are flattened over (B,P,M) and integrated with the allocation-light
    derivative deriv_fast (identical equations to core.deriv)."""
    B, P_, N, _ = U.shape; M = x0.shape[1]
    shp = (B, P_, M)
    S = [np.ascontiguousarray(np.broadcast_to(x0[:, None, :, i], shp), dtype=dtype).copy() for i in range(NX)]
    thb = {k: np.ascontiguousarray(np.broadcast_to(np.asarray(v)[:, None], shp), dtype=dtype) for k, v in th.items()}
    motb = {k: np.asarray(v, dtype=dtype)[:, None] for k, v in mot.items()}
    c = PR.NOM
    nsub = int(round(PR.DT_PLAN / dt))
    sub_per_ctrl = max(1, int(round(PR.DT_CTRL / dt)))
    J = np.zeros(shp, dtype); Fpk = np.zeros(shp, dtype); Ppk = S[IPR].copy()
    dmin = np.full(shp, 1e9, dtype); qin = np.zeros(shp, dtype); qlost = np.zeros(shp, dtype)
    dmn = np.zeros((B, 1, M), dtype); dms = np.zeros((B, 1, M), dtype)
    servo = ctx.get("servo", False)
    if servo:
        cn = (ctx["n_tgt"][:, None, None] - motb["mn0"]); cs = (ctx["s_tgt"][:, None, None] - motb["ms0"])
    rp = ctx["ref_prev"]
    htgt = thb["h"] - PR.D_TARGET
    hh = thb["h"]
    for i in range(N):
        for j in range(nsub):
            tau = (i * nsub + j) * dt
            mdn, mds = motion_vel(motb, tau)
            if j % sub_per_ctrl == 0:
                if tau < TAU_NOM["a"]:
                    r = [rp[:, k][:, None, None] for k in range(4)]
                else:
                    r = [U[:, :, i, k][:, :, None] for k in range(4)]
                if servo:
                    un = np.clip(K_SERVO * (cn - dmn - S[IP]), -150, 150)
                    us = np.clip(K_SERVO * (cs - dms - S[IES]), -150, 150)
                    uphi = -2.0 * S[IPHI]
                else:
                    un = np.clip(K_FF * mdn + np.clip(K_T * (r[1] - S[IP]), -V_FB, V_FB), PR.U_LO[1], PR.U_HI[1])
                    us = np.clip(K_FF * mds + np.clip(K_T * (r[0] - S[IES]), -V_FB, V_FB), PR.U_LO[0], PR.U_HI[0])
                    uphi = np.clip(K_PHI * (r[2] - S[IPHI]), PR.U_LO[2], PR.U_HI[2])
                q = np.broadcast_to(r[3], shp)
            D, (Fn, ql) = deriv_fast(S, us, un, uphi, q, thb, mdn, mds, c)
            for k in range(NX):
                S[k] = S[k] + dt * D[k]
            np.maximum(S[IV], 0, out=S[IV]); np.maximum(S[IA], 0, out=S[IA])
            dmn = dmn + dt * mdn; dms = dms + dt * mds
            en = S[IP] - htgt
            stage = W["n"] * en * en + W["s"] * S[IES] * S[IES] + W["F"] * Fn * Fn + W["D"] * D[ID] + W["phi"] * uphi * uphi
            if bleb_terms:
                stage = stage + W["q"] * q * q + W["R"] * ql
            J += dt * stage
            np.maximum(Fpk, Fn, out=Fpk); np.maximum(Ppk, S[IPR], out=Ppk)
            np.minimum(dmin, hh - S[IP], out=dmin)
            qin += dt * q; qlost += dt * ql
    Uc = np.concatenate([np.broadcast_to(ctx["ref_prev"][:, None, None], (B, P_, 1, 4)), U], 2)
    dU = np.diff(Uc, axis=2)
    J += (W["du"] * (dU[..., 0] ** 2 + dU[..., 1] ** 2) + W["dq"] * dU[..., 3] ** 2).sum(-1)[..., None]
    if bleb_terms:
        tend = t0 + N * PR.DT_PLAN
        V, A, P = S[IV], S[IA], S[IPR]
        J += W["V"] * np.maximum(v_ref(tend) - V, 0) ** 2 + W["Vover"] * np.maximum(V - PR.V_STAR, 0) ** 2
        J += W["A"] * np.maximum(PR.A_STAR * v_ref(tend) / PR.V_STAR - A, 0) ** 2
        J += W["Pend"] * np.maximum(P - 1.0, 0) ** 2
        r_req = (PR.V_MIN + 1.0 - V) / max(PR.T_MAX - tend, 0.2)
        J += W["ttg"] * np.maximum(r_req - Q_SUST, 0) ** 2
    # episode-level reflux ratio projected to the end of the horizon (chance constraint on RR)
    Qin0 = ctx.get("Qin0", np.zeros(B))[:, None, None]; Ql0 = ctx.get("Ql0", np.zeros(B))[:, None, None]
    RRh = (Ql0 + qlost) / (Qin0 + qin + 1.0)
    R = (np.maximum(Fpk / PR.F_SAFE - 0.80, 0) + np.maximum((PR.DELTA_RPE + D_SOFT - dmin) / PR.DELTA_RPE, 0)
         + np.maximum(Ppk / PR.P_MAX - 0.88, 0) + np.maximum(RRh - 0.8 * PR.RR_MAX, 0))
    return J.astype(np.float64), R.astype(np.float64), None


def cvar(R, alpha):
    """Empirical CVaR_alpha over the last axis (Rockafellar--Uryasev)."""
    M = R.shape[-1]
    Rs = np.sort(R, -1)
    k = int(np.floor(alpha * M))
    var = Rs[..., k - 1] if k > 0 else np.zeros(R.shape[:-1])
    return var + np.maximum(R - var[..., None], 0).sum(-1) / ((1 - alpha) * M)


KNOTS = np.array([0, 2, 5, 9])


def knots_to_blocks(Kn):
    """Linear interpolation of knot values (...,nk,4) to the N planning blocks (...,N,4)."""
    N = PR.N_PLAN
    out = np.empty(Kn.shape[:-2] + (N, Kn.shape[-1]))
    for i in range(N):
        j = np.searchsorted(KNOTS, i, side="right") - 1
        j = min(j, len(KNOTS) - 2)
        a = (i - KNOTS[j]) / (KNOTS[j + 1] - KNOTS[j])
        out[..., i, :] = (1 - a) * Kn[..., j, :] + a * Kn[..., j + 1, :]
    return out


def blocks_to_knots_shifted(U):
    """Warm start: knots of the previous plan shifted by one block."""
    N = PR.N_PLAN
    idx = np.minimum(KNOTS + 1, N - 1)
    return U[..., idx, :]


class CEMPlanner:
    def __init__(self, B, rng, mode="joint", lam=0.75, alpha=0.9, M=16, P=32, iters=3, n_elite=5,
                 posterior=True, inner="twin", surrogate=None, refine_every=0, dt_sub=PR.DT_SUB):
        self.B = B; self.rng = rng; self.mode = mode; self.lam = lam; self.alpha = alpha
        self.M = M; self.P = P; self.iters = iters; self.ne = n_elite
        self.posterior = posterior; self.inner = inner; self.sur = surrogate
        self.refine_every = refine_every; self.dt_sub = dt_sub
        self.sampler = ScenarioSampler(rng, M)
        self.mu = None
        self.ref_prev = None
        self.n_plans = 0
        self.last_cost = None
        self.Qin = np.zeros(B)   # cumulative issued infusion volume (uL), updated by the runner

    def _evaluate(self, U, x, th, mot, ctx, t0, use_sur, bleb_terms=True):
        if use_sur:
            return self.sur.rollout(x, th, mot, U, ctx, t0, bleb_terms=bleb_terms)
        return rollout_twin(x, th, mot, U, ctx, t0, dt=self.dt_sub, bleb_terms=bleb_terms)

    def init_mean(self, ukf, t0):
        B, N = self.B, PR.N_PLAN
        mu = np.zeros((B, N, 4))
        mu[..., 1] = (ukf.z[:, ZH] - PR.D_TARGET)[:, None]
        mu[..., 3] = q_nominal(t0 + PR.DT_PLAN * np.arange(N))[None]
        mu[..., 2] = 0.0
        return mu

    def plan(self, t0, ukf, enc, nominal_ctl):
        B, N = self.B, PR.N_PLAN
        x, th, mot = self.sampler.sample(ukf, enc, self.posterior)
        if self.mu is None:
            self.mu = self.init_mean(ukf, t0)
            self.ref_prev = np.zeros((B, 4)); self.ref_prev[:, 1] = x[:, 0, IP]; self.ref_prev[:, 0] = x[:, 0, IES]
        use_sur = self.inner == "surrogate" or (self.inner == "hybrid" and (self.n_plans % self.refine_every) != 0)
        nk = len(KNOTS)
        mu = blocks_to_knots_shifted(self.mu)
        sig = np.broadcast_to(SIG0, (B, nk, 4)).copy()
        ctx = dict(ref_prev=self.ref_prev, Qin0=self.Qin, Ql0=np.maximum(self.Qin - ukf.z[:, ZV], 0.0))
        if self.mode == "traj":
            opt = np.array([1, 1, 1, 0], bool)
            mu[..., 3] = q_nominal(t0 + PR.DT_PLAN * KNOTS)[None]
        elif self.mode == "inf":
            opt = np.array([0, 0, 0, 1], bool)
            ctx.update(servo=True, n_tgt=nominal_ctl.n_tgt, s_tgt=nominal_ctl.s_tgt)
        else:
            opt = np.array([1, 1, 1, 1], bool)
        best_K = mu.copy(); best_c = np.full(B, np.inf)
        dec = self.mode == "decoupled"
        iters = 4 if dec else self.iters
        Pc = (self.P * self.iters) // iters
        best_cm = np.full(B, np.inf); best_cq = np.full(B, np.inf)
        for it in range(iters):
            vary = opt.copy()
            if dec:
                vary = np.array([1, 1, 1, 0], bool) if it % 2 == 0 else np.array([0, 0, 0, 1], bool)
            eps = self.rng.normal(0, 1, (B, Pc, nk, 4))
            Kc = mu[:, None] + sig[:, None] * eps * vary
            Kc[:, 0] = mu
            Kc = np.clip(Kc, REF_LO, REF_HI)
            # admissible depth set-points: terminal tip-to-RPE distance >= DELTA_RPE + REF_MARGIN
            Kc[..., 1] = np.minimum(Kc[..., 1], (ukf.z[:, ZH] - PR.DELTA_RPE - REF_MARGIN)[:, None, None])
            U = knots_to_blocks(Kc)
            if dec and it % 2 == 0:
                # motion planner: bleb/infusion outcomes are not part of its objective
                J, R, _ = self._evaluate(U, x, th, mot, ctx, t0, use_sur, bleb_terms=False)
            elif dec:
                # infusion planner: needle assumed to hold its current depth over the horizon
                Ue = U.copy(); Ue[..., 1] = x[:, 0, IP][:, None, None]; Ue[..., 0] = x[:, 0, IES][:, None, None]
                J, R, _ = self._evaluate(Ue, x, th, mot, ctx, t0, use_sur)
            else:
                J, R, _ = self._evaluate(U, x, th, mot, ctx, t0, use_sur)
            cost = J.mean(-1) + W["viol"] * R.mean(-1) + self.lam * W["viol"] * cvar(R, self.alpha)
            order = np.argsort(cost, 1)
            el = np.take_along_axis(Kc, order[:, :self.ne, None, None], 1)
            mu = np.where(vary, el.mean(1), mu)
            sig = np.where(vary, np.maximum(el.std(1), 0.1 * SIG0), sig)
            cbest = np.take_along_axis(cost, order[:, :1], 1)[:, 0]
            Kbest = el[:, 0]
            if dec:
                if it % 2 == 0:
                    imp = cbest < best_cm; best_cm = np.where(imp, cbest, best_cm)
                    best_K[imp, :, :3] = Kbest[imp, :, :3]
                else:
                    imp = cbest < best_cq; best_cq = np.where(imp, cbest, best_cq)
                    best_K[imp, :, 3] = Kbest[imp, :, 3]
            else:
                imp = cbest < best_c
                best_c = np.where(imp, cbest, best_c); best_K[imp] = Kbest[imp]
        best_U = knots_to_blocks(best_K)
        if getattr(self, "collector", None) is not None:
            self.collector(x, th, mot, U, ctx, t0, self.n_plans)
        self.mu = best_U
        self.last_cost = best_c
        self.n_plans += 1
        self.ref_prev = best_U[:, 0].copy()
        return best_U[:, 0].copy()


class NominalController:
    """Pre-planned path from the initial OCT depth reading, servoed on robot
    encoders (no motion compensation), and fixed-rate infusion."""
    def __init__(self, y0):
        self.n_tgt = y0["n_enc"] + y0["D"] - PR.D_TARGET
        self.s_tgt = y0["s_enc"] - y0["S"]

    def traj_cmd(self, enc):
        un = np.clip(K_SERVO * (self.n_tgt - enc["n_enc"]), -150, 150)
        us = np.clip(K_SERVO * (self.s_tgt - enc["s_enc"]), -150, 150)
        uphi = -2.0 * enc["phi_enc"]
        return un, us, uphi

    def command(self, t, enc):
        un, us, uphi = self.traj_cmd(enc)
        q = q_nominal(t) * np.ones_like(un)
        return np.stack([us, un, uphi, q], -1)
