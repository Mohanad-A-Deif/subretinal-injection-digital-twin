"""Augmented-state unscented Kalman filter for joint state estimation and
online (recursive) MAP identification of the identifiable parameter block.

z = [mn, mdn, ms, mds, V, A, P, bF, log_kk, log_ar, log_bq, log_pt, h]
Robot encoders (n, s, vn, vs, phi, om) are treated as exact exogenous inputs.
Delayed channels are fused with stored encoder states at the nominal channel delay.
"""
import numpy as np
from . import params as PR
from .core import deriv, NX, IP, IES, IVN, IVS, IPHI, IOM, IV, IA, IPR, contact_force

NZ = 13
ZMN, ZMDN, ZMS, ZMDS, ZV, ZA, ZP, ZBF, ZLK, ZLA, ZLB, ZLT, ZH = range(NZ)
TAU_NOM = dict(F=0.0125, D=0.0225, P=0.020, S=0.040, V=0.050, a=0.040)
ID_IDX = [ZLK, ZLA, ZLB, ZLT, ZH]


class UKF:
    def __init__(self, B, y0, mode="fixed", alpha=1.0, beta=2.0, kappa=0.0):
        """mode: 'fixed' (constant prior/process covariance), 'adaptive'
        (innovation-triggered covariance inflation), 'frozen' (no parameter update)."""
        self.B = B; self.mode = mode
        n = NZ
        lam = alpha ** 2 * (n + kappa) - n
        self.c = n + lam
        self.Wm = np.full(2 * n + 1, 1.0 / (2 * self.c)); self.Wm[0] = lam / self.c
        self.Wc = self.Wm.copy(); self.Wc[0] += 1 - alpha ** 2 + beta
        mu_id, S_id = PR.prior_id()
        self.mu_id, self.S_id = mu_id, S_id
        z = np.zeros((B, n)); P = np.zeros((B, n, n))
        z[:, ZMN] = y0["D"] + y0["n_enc"] - mu_id[4]
        z[:, ZMS] = y0["s_enc"] - y0["S"]
        z[:, ZP] = PR.NOM["P_base"]
        z[:, ZLK:ZH + 1] = mu_id
        d = np.zeros(n)
        d[ZMN] = 36.0 + S_id[4, 4]; d[ZMDN] = 150.0 ** 2; d[ZMS] = 64.0; d[ZMDS] = 300.0 ** 2
        d[ZV] = 0.01; d[ZA] = 0.01; d[ZP] = 1e-4; d[ZBF] = 0.3 ** 2
        d[ZLK:ZH + 1] = np.diag(S_id)
        P[:] = np.diag(d)
        P[:, ZMN, ZH] = P[:, ZH, ZMN] = -S_id[4, 4]
        self.z, self.P = z, P
        q = np.zeros(n)
        q[ZMN] = 9.0; q[ZMDN] = 45.0 ** 2; q[ZMS] = 4.0; q[ZMDS] = 70.0 ** 2
        q[ZV] = 0.08 ** 2; q[ZA] = 0.05 ** 2; q[ZP] = 0.01 ** 2; q[ZBF] = 0.03 ** 2
        q[ZLK] = q[ZLA] = q[ZLB] = q[ZLT] = 0.003 ** 2; q[ZH] = 0.5 ** 2
        if mode == "frozen":
            q[ZLK:ZH + 1] = 0.0
            P[:, ZLK:ZH + 1, :] = 0.0; P[:, :, ZLK:ZH + 1] = 0.0
            for i in range(ZLK, ZH + 1):
                P[:, i, i] = 1e-8
            self.P = P
        self.Q = np.diag(q)
        self.R = np.diag([PR.SENS["sig_F"] ** 2 + 0.15 ** 2, PR.SENS["sig_D"] ** 2 + 4.0 ** 2, PR.SENS["sig_P"] ** 2, PR.SENS["sig_S"] ** 2, 1.0])
        self.enc_hist = []   # list of encoder dicts per step
        self.q_hist = []     # issued infusion commands
        self.nis = np.zeros(B)
        self.n_infl = np.zeros(B, int)

    # ------------------------------------------------------------ sigma points
    def _sigma(self):
        n = NZ
        Pj = 0.5 * (self.P + np.swapaxes(self.P, -1, -2)) + 1e-9 * np.eye(n)
        try:
            L = np.linalg.cholesky(self.c * Pj)
        except np.linalg.LinAlgError:
            w, V = np.linalg.eigh(Pj)
            w = np.maximum(w, 1e-9)
            self.P = V @ (w[..., None] * np.swapaxes(V, -1, -2))
            L = V * np.sqrt(self.c * w)[..., None, :]
        Lt = np.swapaxes(L, -1, -2)  # rows are columns of L
        X = np.concatenate([self.z[:, None], self.z[:, None] + Lt, self.z[:, None] - Lt], 1)
        return X  # (B, 2n+1, n)

    def _th(self, X):
        th = {k: PR.NOM[k] for k in ["mu", "alpha_l", "P_l", "eta_A", "lam_A", "tau_P", "beta_A", "beta_V"]}
        th = {k: np.full(X.shape[:-1], v) for k, v in th.items()}
        th["kk"] = np.exp(X[..., ZLK]); th["alpha_r"] = PR.NOM["alpha_r"] * np.exp(X[..., ZLA])
        th["beta_q"] = PR.NOM["beta_q"] * np.exp(X[..., ZLB]); th["h"] = X[..., ZH]
        th["P_tract"] = PR.NOM["P_tract"] * np.exp(X[..., ZLT])
        return th

    def _enc_at(self, lag_s):
        k = len(self.enc_hist) - 1 - int(round(lag_s / PR.DT_CTRL))
        return self.enc_hist[max(k, 0)]

    def _q_applied(self, lag_s):
        k = len(self.q_hist) - 1 - int(round(lag_s / PR.DT_CTRL))
        return self.q_hist[k] if k >= 0 else np.zeros(self.B)

    # ------------------------------------------------------------ predict
    def predict(self, enc_prev, q_cmd):
        """Propagate over one control interval using encoder values at t_k."""
        self.q_hist.append(q_cmd.copy())
        X = self._sigma()
        th = self._th(X)
        q = self._q_applied(TAU_NOM["a"])[:, None] * np.ones(X.shape[:2])
        nsub = 4; dt = PR.DT_CTRL / nsub
        Xn = X.copy()
        for _ in range(nsub):
            x = np.zeros(X.shape[:2] + (NX,))
            x[..., IP] = enc_prev["n_enc"][:, None] - Xn[..., ZMN]
            x[..., IES] = enc_prev["s_enc"][:, None] - Xn[..., ZMS]
            x[..., IVN] = enc_prev["vn_enc"][:, None]; x[..., IVS] = enc_prev["vs_enc"][:, None]
            x[..., IPHI] = enc_prev["phi_enc"][:, None]; x[..., IOM] = enc_prev["om_enc"][:, None]
            x[..., IV] = Xn[..., ZV]; x[..., IA] = Xn[..., ZA]; x[..., IPR] = Xn[..., ZP]
            u = np.zeros(X.shape[:2] + (4,)); u[..., 3] = q
            xd, _ = deriv(x, u, th, Xn[..., ZMDN], Xn[..., ZMDS])
            Xn[..., ZV] = np.maximum(Xn[..., ZV] + dt * xd[..., IV], 0)
            Xn[..., ZA] = np.maximum(Xn[..., ZA] + dt * xd[..., IA], 0)
            Xn[..., ZP] = Xn[..., ZP] + dt * xd[..., IPR]
            Xn[..., ZMN] += dt * Xn[..., ZMDN]; Xn[..., ZMS] += dt * Xn[..., ZMDS]
        Xn[..., ZBF] *= 0.985
        z = np.einsum("j,bjn->bn", self.Wm, Xn)
        dX = Xn - z[:, None]
        P = np.einsum("j,bjn,bjm->bnm", self.Wc, dX, dX) + self.Q
        self.z, self.P = z, P

    # ------------------------------------------------------------ update
    def _hmeas(self, X):
        th = self._th(X)
        out = np.zeros(X.shape[:2] + (5,))
        eF = self._enc_at(TAU_NOM["F"]); eD = self._enc_at(TAU_NOM["D"]); eS = self._enc_at(TAU_NOM["S"])
        mnF = X[..., ZMN] - X[..., ZMDN] * TAU_NOM["F"]
        pF = eF["n_enc"][:, None] - mnF
        pdF = eF["vn_enc"][:, None] - X[..., ZMDN]
        out[..., 0] = contact_force(pF, pdF, th["kk"]) + X[..., ZBF]
        mnD = X[..., ZMN] - X[..., ZMDN] * TAU_NOM["D"]
        out[..., 1] = X[..., ZH] - (eD["n_enc"][:, None] - mnD)
        out[..., 2] = X[..., ZP] + PR.NOM["r_c"] * self._q_applied(TAU_NOM["P"] + TAU_NOM["a"])[:, None]
        msS = X[..., ZMS] - X[..., ZMDS] * TAU_NOM["S"]
        out[..., 3] = eS["s_enc"][:, None] - msS
        out[..., 4] = X[..., ZV]
        return out

    def update(self, y):
        self.enc_hist.append({k: y[k].copy() for k in ["n_enc", "s_enc", "vn_enc", "vs_enc", "phi_enc", "om_enc"]})
        if len(self.enc_hist) > 50:
            self.enc_hist.pop(0)
        X = self._sigma()
        Y = self._hmeas(X)
        yhat = np.einsum("j,bjm->bm", self.Wm, Y)
        dY = Y - yhat[:, None]; dX = X - self.z[:, None]
        R = np.broadcast_to(self.R, (self.B, 5, 5)).copy()
        R[:, 4, 4] = (PR.SENS["sig_V0"] + PR.SENS["sig_Vrel"] * np.maximum(self.z[:, ZV], 0)) ** 2
        S = np.einsum("j,bjm,bjk->bmk", self.Wc, dY, dY) + R
        C = np.einsum("j,bjn,bjm->bnm", self.Wc, dX, dY)
        yv = np.stack([y["F"], y["D"], y["P"], y["S"], y["V"]], -1)
        miss = ~np.isfinite(yv)
        if "gated" in self.mode:
            # chi-square innovation gating (3 sigma) of the force and pressure channels
            zz = (yv - yhat) ** 2 / np.einsum("bii->bi", S)
            gate = np.zeros_like(miss); gate[:, [0, 2]] = np.nan_to_num(zz[:, [0, 2]], nan=0.0) > 9.0
            self.n_gated = getattr(self, "n_gated", 0) + gate.sum(0)
            miss = miss | gate
        innov = np.where(miss, 0.0, yv - yhat)
        S = S + np.einsum("bm,mk->bmk", miss * 1e12, np.eye(5))
        Sinv = np.linalg.inv(S)
        K = C @ Sinv
        if self.mode == "frozen":
            K[:, ZLK:ZH + 1, :] = 0.0
        self.z = self.z + np.einsum("bnm,bm->bn", K, innov)
        self.P = self.P - K @ S @ np.swapaxes(K, -1, -2)
        self.P = 0.5 * (self.P + np.swapaxes(self.P, -1, -2))
        if self.mode == "frozen":
            # parameters fixed at the prior mean: no parameter uncertainty inside the filter
            ii = np.arange(ZLK, ZH + 1)
            self.P[:, ii, :] = 0.0; self.P[:, :, ii] = 0.0
            self.P[:, ii, ii] = 1e-8
            self.z[:, ZLK:ZH + 1] = self.mu_id
        # normalized innovation squared on force and pressure channels
        e = innov[:, [0, 2]]; Sd = np.stack([S[:, 0, 0], S[:, 2, 2]], -1)
        nis = (e * e / Sd).sum(-1) / 2.0
        self.nis = 0.9 * self.nis + 0.1 * nis
        if "adaptive" in self.mode:
            trig = self.nis > 3.0
            if trig.any():
                infl = np.zeros((NZ, NZ))
                infl[ZLK:ZH, ZLK:ZH] = 0.5 * self.S_id[:4, :4]
                infl[ZBF, ZBF] = 0.5 ** 2
                self.P[trig] += infl
                self.nis[trig] = 1.0
                self.n_infl += trig
        # keep identified parameters inside the admissible set
        self.z[:, ZLK:ZLT + 1] = np.clip(self.z[:, ZLK:ZLT + 1], -0.7, 0.7)
        self.z[:, ZH] = np.clip(self.z[:, ZH], 120, 320)
        self.z[:, ZV] = np.maximum(self.z[:, ZV], 0); self.z[:, ZA] = np.maximum(self.z[:, ZA], 0)

    # ------------------------------------------------------------ outputs
    def state_estimate(self, enc):
        """Twin state estimate (B,12) in the relative frame + motion velocities."""
        x = np.zeros((self.B, NX))
        x[:, IP] = enc["n_enc"] - self.z[:, ZMN]
        x[:, IES] = enc["s_enc"] - self.z[:, ZMS]
        x[:, IVN] = enc["vn_enc"]; x[:, IVS] = enc["vs_enc"]; x[:, IPHI] = enc["phi_enc"]; x[:, IOM] = enc["om_enc"]
        x[:, IV] = self.z[:, ZV]; x[:, IA] = self.z[:, ZA]; x[:, IPR] = self.z[:, ZP]
        return x, self.z[:, ZMDN].copy(), self.z[:, ZMDS].copy()

    def param_posterior(self):
        idx = ID_IDX
        return self.z[:, idx].copy(), self.P[:, idx][:, :, idx].copy()
