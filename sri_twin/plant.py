"""Truth plant: the full digital twin propagated at 1 ms with actuation delay,
retinal micro-motion (harmonic + Ornstein--Uhlenbeck), stochastic leakage and
delayed/noisy/drifting sensing channels.  Vectorized over B episodes."""
import numpy as np
from . import params as PR
from .core import deriv, clamp_state, NX, IP, IES, IVN, IVS, IPHI, IOM, IV, IA, IPR, ID, IQIN, IQRF, sigmoid

HIST = 200  # 1 ms samples of history kept for delayed sensing


class Plant:
    def __init__(self, theta, rng, variant=None, sensing=None):
        """variant: structural/OOD options
            contact: 'kv'|'puncture'; bleb: 'dyn'|'compliance'
            saccades: bool; delay_model: 'uniform'|'heavy'|'drift'
            dropout: float; shift: bool
        sensing: dict(noise_mult, bias_F, bias_D)"""
        self.th = {k: np.array(v, dtype=float) for k, v in theta.items()}
        self.B = B = len(theta["kk"])
        self.rng = rng
        self.var = variant or {}
        self.sens = dict(noise_mult=1.0, bias_F=0.0, bias_D=0.0, noise_mult_F=1.0, noise_mult_D=1.0)
        if sensing:
            self.sens.update(sensing)
        th = self.th
        x = np.zeros((B, NX))
        x[:, IP] = th["p0"]; x[:, IES] = th["es0"]; x[:, IPHI] = th["phi0"]; x[:, IPR] = PR.NOM["P_base"]
        self.x = x
        self.t = 0.0
        self.eta = np.zeros((B, 2))  # OU residual [n, s]
        self.jump = np.zeros((B, 2))  # saccade-like offsets (target value)
        self.jump_cur = np.zeros((B, 2))
        self.bF = np.zeros(B)
        self.U = np.zeros((int(round(PR.T_MAX / PR.DT_CTRL)) + 3, B, 4))
        self.nU = 1
        self.hist = {k: np.zeros((B, HIST)) for k in ["Fn", "p", "es", "P", "q", "dRPE", "V"]}
        self.hptr = 1
        self.hist["p"][:] = x[:, IP:IP + 1]; self.hist["es"][:] = x[:, IES:IES + 1]
        self.hist["P"][:] = PR.NOM["P_base"]; self.hist["dRPE"][:] = (th["h"] - x[:, IP])[:, None]
        self.done = np.zeros(B, bool)
        self.aborted = np.zeros(B, bool)
        self.success_time = np.full(B, np.nan)
        self.viol = {k: np.zeros(B, bool) for k in ["force", "depth", "pressure"]}
        self.Fmax = np.zeros(B); self.Pmax = np.full(B, PR.NOM["P_base"]); self.dmin = np.full(B, 1e9)
        self.shift_t = rng.uniform(3.0, 5.0, B) if self.var.get("shift") else np.full(B, np.inf)
        self.shifted = np.zeros(B, bool)
        # time-varying delay states
        self.tau_a_cur = th["tau_a"].copy()
        self.k = 0

    # ---------------------------------------------------------------- motion
    def motion(self, t):
        th = self.th
        w_s = 2 * np.pi * th["f_s"]; w_n = 2 * np.pi * th["f_n"]
        ms = (th["a_s"] * np.sin(w_s * t + th["ph_s"])).sum(1)
        mn = (th["a_n"] * np.sin(w_n * t + th["ph_n"])).sum(1)
        mds = (th["a_s"] * w_s * np.cos(w_s * t + th["ph_s"])).sum(1)
        mdn = (th["a_n"] * w_n * np.cos(w_n * t + th["ph_n"])).sum(1)
        ms = ms + self.eta[:, 1] + self.jump_cur[:, 1]
        mn = mn + self.eta[:, 0] + self.jump_cur[:, 0]
        return mn, ms, mdn, mds

    # ---------------------------------------------------------------- one control step
    def step(self, u_cmd):
        """Apply command u_cmd (B,4) issued at time t_k, integrate to t_{k+1}."""
        u_cmd = np.clip(u_cmd, PR.U_LO, PR.U_HI)
        u_cmd = np.where(self.done[:, None], np.array([0, 0, 0, 0.0]), u_cmd)
        self.U[self.nU] = u_cmd
        self.nU += 1
        th = self.th; B = self.B; rng = self.rng
        nsub = int(round(PR.DT_CTRL / PR.DT_INT))
        dt = PR.DT_INT
        # delay model
        dm = self.var.get("delay_model", "uniform")
        if dm == "heavy":
            jit = np.minimum(rng.lognormal(np.log(0.010), 0.8, B), 0.120)
            tau_a = th["tau_a"] + jit
        elif dm == "drift":
            self.tau_a_cur = np.clip(self.tau_a_cur + rng.normal(0, 0.004, B), 0.015, 0.120)
            tau_a = self.tau_a_cur
        else:
            tau_a = th["tau_a"]
        self._tau_sense_extra = 0.0
        for j in range(nsub):
            t = self.t
            # abrupt mid-procedure shift
            if self.var.get("shift"):
                trig = (~self.shifted) & (t >= self.shift_t)
                if trig.any():
                    th["kk"][trig] *= 1.5; th["alpha_r"][trig] *= 2.0
                    self.bF[trig] += 1.0
                    self.shifted |= trig
            # delayed command
            idx = np.floor((t - tau_a) / PR.DT_CTRL + 1e-9).astype(int) + 1
            idx = np.clip(idx, 0, self.nU - 1)
            u = self.U[idx, np.arange(B)]
            mn, ms, mdn, mds = self.motion(t)
            # OU drift contributes to motion velocity used in the force damping term
            mdn_eff = mdn - th["ou_lam"] * self.eta[:, 0]
            mds_eff = mds - th["ou_lam"] * self.eta[:, 1]
            xd, aux = deriv(self.x, u, th, mdn_eff, mds_eff, variant=self.var)
            active = ~self.done
            noise_leak = PR.NOM["sig_l"] * rng.normal(0, 1, B) * (self.x[:, IV] > 1.0)
            xd[:, IV] -= noise_leak
            xn = self.x + dt * xd
            # OU and saccade increments shift the relative coordinates
            deta = -th["ou_lam"][:, None] * self.eta * dt + th["ou_sig"][:, None] * np.sqrt(dt) * rng.normal(0, 1, (B, 2))
            # harmonic part already in mdn_eff; add the stochastic part of the OU increment
            dstoch = deta + th["ou_lam"][:, None] * self.eta * dt
            djump = np.zeros((B, 2))
            if self.var.get("saccades"):
                newj = rng.random(B) < 0.35 * dt  # ~0.35 events per second
                if newj.any():
                    amp = rng.uniform(20, 60, (newj.sum(), 2)) * rng.choice([-1, 1], (newj.sum(), 2))
                    self.jump[newj] += amp
                djump = (self.jump - self.jump_cur) * min(1.0, dt / 0.02)
                self.jump_cur += djump
            xn[:, IP] -= dstoch[:, 0] + djump[:, 0]
            xn[:, IES] -= dstoch[:, 1] + djump[:, 1]
            self.eta += deta
            xn = clamp_state(xn)
            self.x = np.where(active[:, None], xn, self.x)
            self.t = t + dt
            # bookkeeping
            Fn = aux["Fn"]
            dRPE = th["h"] - self.x[:, IP]
            self.Fmax = np.where(active, np.maximum(self.Fmax, Fn), self.Fmax)
            self.Pmax = np.where(active, np.maximum(self.Pmax, self.x[:, IPR]), self.Pmax)
            self.dmin = np.where(active, np.minimum(self.dmin, dRPE), self.dmin)
            vF = active & (Fn > PR.F_SAFE); vD = active & (dRPE < PR.DELTA_RPE); vP = active & (self.x[:, IPR] > PR.P_MAX)
            self.viol["force"] |= vF; self.viol["depth"] |= vD; self.viol["pressure"] |= vP
            ab = vF | vD | vP
            self.aborted |= ab
            self.done |= ab
            h = self.hptr % HIST
            self.hist["Fn"][:, h] = Fn; self.hist["p"][:, h] = self.x[:, IP]; self.hist["es"][:, h] = self.x[:, IES]
            self.hist["P"][:, h] = self.x[:, IPR]; self.hist["q"][:, h] = u[:, 3]; self.hist["dRPE"][:, h] = dRPE
            self.hist["V"][:, h] = self.x[:, IV]
            self.hptr += 1
        # success: target volume reached
        reached = (~self.done) & (self.x[:, IV] >= PR.V_STAR)
        self.success_time[reached] = self.t
        self.done |= reached
        self.k += 1
        # force drift
        self.bF = th["rho_F"] * self.bF + th["sig_bF"] * rng.normal(0, 1, B)
        if self.t >= PR.T_MAX - 1e-9:
            self.done[:] = True

    # ---------------------------------------------------------------- sensing
    def _delayed(self, key, tau):
        lag = np.clip(np.round(tau / PR.DT_INT).astype(int), 0, HIST - 1)
        idx = (self.hptr - 1 - lag) % HIST
        return self.hist[key][np.arange(self.B), idx]

    def observe(self):
        """Delayed, noisy channels at t_k plus exact robot encoders.
        Returns dict with y_F, y_D, y_P, y_S (NaN when dropped) and encoder values."""
        th = self.th; rng = self.rng; B = self.B; s = self.sens
        extra = 0.0
        if self.var.get("delay_model") == "heavy":
            extra = np.minimum(rng.lognormal(np.log(0.010), 0.8, B), 0.120)
        elif self.var.get("delay_model") == "drift":
            extra = self.tau_a_cur - th["tau_a"]
        nm = s["noise_mult"]
        y = {}
        y["F"] = self._delayed("Fn", th["tau_F"] + extra) + self.bF + s["bias_F"] + PR.SENS["sig_F"] * nm * s["noise_mult_F"] * rng.normal(0, 1, B)
        y["D"] = self._delayed("dRPE", th["tau_D"] + extra) + s["bias_D"] + PR.SENS["sig_D"] * nm * s["noise_mult_D"] * rng.normal(0, 1, B)
        y["P"] = (self._delayed("P", th["tau_P"] + extra) + PR.NOM["r_c"] * self._delayed("q", th["tau_P"] + extra)
                  + PR.SENS["sig_P"] * nm * rng.normal(0, 1, B))
        y["S"] = self._delayed("es", th["tau_S"] + extra) + PR.SENS["sig_S"] * nm * rng.normal(0, 1, B)
        Vd = self._delayed("V", th["tau_V"] + extra)
        y["V"] = Vd + (PR.SENS["sig_V0"] + PR.SENS["sig_Vrel"] * Vd) * nm * rng.normal(0, 1, B)
        dp = self.var.get("dropout", 0.0)
        if dp > 0:
            for k in y:
                y[k] = np.where(rng.random(B) < dp, np.nan, y[k])
        mn, ms, _, _ = self.motion(self.t)
        y["n_enc"] = self.x[:, IP] + mn      # robot-frame needle depth (encoder)
        y["s_enc"] = self.x[:, IES] + ms     # robot-frame lateral position
        y["vn_enc"] = self.x[:, IVN]; y["vs_enc"] = self.x[:, IVS]
        y["phi_enc"] = self.x[:, IPHI]; y["om_enc"] = self.x[:, IOM]
        return y

    # ---------------------------------------------------------------- truth for evaluation
    def truth(self):
        mn, ms, mdn, mds = self.motion(self.t)
        return dict(x=self.x.copy(), mn=mn, ms=ms, mdn=mdn, mds=mds, bF=self.bF.copy(),
                    kk=self.th["kk"].copy(), alpha_r=self.th["alpha_r"].copy(),
                    beta_q=self.th["beta_q"].copy(), h=self.th["h"].copy(), P_tract=self.th["P_tract"].copy())

    def metrics(self):
        x = self.x; th = self.th
        Qin = np.maximum(x[:, IQIN], 1e-9)
        RR = x[:, IQRF] / Qin
        IE = x[:, IV] / Qin
        V, A = x[:, IV], x[:, IA]
        S = sigmoid(0.5 * (V - PR.V_MIN)) * sigmoid(1.0 * (A - PR.A_MIN)) * sigmoid(20 * (PR.P_MAX - self.Pmax))
        S = np.where(self.aborted, 0.0, S)
        vR = (RR > PR.RR_MAX) & (x[:, IQIN] > 1.0)
        viol = self.viol["force"] | self.viol["depth"] | self.viol["pressure"] | vR
        success = (~self.aborted) & (V >= PR.V_MIN) & (A >= PR.A_MIN) & (~vR)
        return dict(viol=viol, v_force=self.viol["force"], v_depth=self.viol["depth"],
                    v_pressure=self.viol["pressure"], v_reflux=vR, RR=RR, IE=IE, S=S,
                    success=success, Fmax=self.Fmax, Pmax=self.Pmax, dmin=self.dmin, D=x[:, ID],
                    V=V, A=A, Qin=x[:, IQIN], En=np.abs(x[:, IP] - (th["h"] - PR.D_TARGET)),
                    Es=np.abs(x[:, IES]), aborted=self.aborted, t_end=np.where(np.isnan(self.success_time), self.t, self.success_time))
