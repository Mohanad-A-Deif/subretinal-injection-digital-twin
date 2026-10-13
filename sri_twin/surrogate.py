"""Physics-constrained neural surrogate of the planner's block transition.

One forward pass maps (block-start state, set-points, previous set-points,
motion-forecast features, scenario parameters) to the block-end state and the
block-level cost/risk features that the scenario objective needs.  It replaces
the 5 sub-steps of the full-twin rollout per 100 ms block.
"""
import time
import numpy as np
import torch
import torch.nn as nn
from . import params as PR
from .core import deriv_fast, NX, IP, IES, IVN, IVS, IPHI, IOM, IV, IA, IPR, ID
from . import controller as C

torch.set_num_threads(2)
PKEYS = ["kk", "alpha_r", "beta_q", "P_tract", "h", "mu", "alpha_l", "P_l", "eta_A", "lam_A", "tau_P", "beta_A", "beta_V"]
SX = [IP, IES, IVN, IVS, IPHI, IOM, IV, IA, IPR]
N_IN = len(SX) + 4 + 4 + 9 + len(PKEYS) + 1
OUT_NAMES = ["dp", "des", "dvn", "dvs", "dphi", "dom", "dV", "dA", "dP",
             "Jmot", "Jbleb", "Fpk", "Ppk", "pmax", "qin", "qlost", "dD"]
N_OUT = len(OUT_NAMES)


def block_inputs(S, ref, refp, mot, tau0, th):
    """Assemble surrogate inputs (..., N_IN) from block-start quantities."""
    w = 2 * np.pi * mot["f"]
    dec = np.exp(-tau0 / 0.25)
    ph = w * tau0 + mot["ph"]
    shp = S[IP].shape
    feats = [S[i] for i in SX]
    feats += [np.broadcast_to(ref[..., k], shp) for k in range(4)]
    feats += [np.broadcast_to(refp[..., k], shp) for k in range(4)]
    mf = [mot["mdn0"] * dec, mot["mds0"] * dec, mot["amp_n"] * w, mot["amp_s"] * w, w,
          np.cos(ph), np.sin(ph), np.cos(mot["ph"]), np.full_like(w, tau0)]
    feats += [np.broadcast_to(m, shp) for m in mf]
    feats += [np.broadcast_to(th[k], shp) for k in PKEYS]
    feats += [np.full(shp, 1.0 if tau0 < C.TAU_NOM["a"] else 0.0)]
    return np.stack(feats, -1)


def twin_block(S, ref, refp, mot, tau0, thb, dt=PR.DT_SUB):
    """Integrate one 100 ms block with the full twin and return (S_next, outputs)."""
    c = PR.NOM
    nsub = int(round(PR.DT_PLAN / dt)); spc = max(1, int(round(PR.DT_CTRL / dt)))
    shp = S[IP].shape
    S0 = [s.copy() for s in S]
    S = [s.copy() for s in S]
    Jm = np.zeros(shp); Jb = np.zeros(shp); Fpk = np.zeros(shp); Ppk = S[IPR].copy(); pmax = S[IP].copy()
    qin = np.zeros(shp); ql_ = np.zeros(shp); D0 = S[ID].copy()
    htgt = thb["h"] - PR.D_TARGET
    for j in range(nsub):
        tau = tau0 + j * dt
        mdn, mds = C.motion_vel(mot, tau)
        if j % spc == 0:
            r = refp if tau < C.TAU_NOM["a"] else ref
            un = np.clip(C.K_FF * mdn + np.clip(C.K_T * (r[..., 1] - S[IP]), -C.V_FB, C.V_FB), PR.U_LO[1], PR.U_HI[1])
            us = np.clip(C.K_FF * mds + np.clip(C.K_T * (r[..., 0] - S[IES]), -C.V_FB, C.V_FB), PR.U_LO[0], PR.U_HI[0])
            uphi = np.clip(C.K_PHI * (r[..., 2] - S[IPHI]), PR.U_LO[2], PR.U_HI[2]); q = np.broadcast_to(r[..., 3], shp)
        D, (Fn, ql) = deriv_fast(S, us, un, uphi, q, thb, mdn, mds, c)
        for k in range(NX):
            S[k] = S[k] + dt * D[k]
        np.maximum(S[IV], 0, out=S[IV]); np.maximum(S[IA], 0, out=S[IA])
        en = S[IP] - htgt
        W = C.W
        Jm += dt * (W["n"] * en * en + W["s"] * S[IES] ** 2 + W["F"] * Fn * Fn + W["D"] * D[ID] + W["phi"] * uphi * uphi)
        Jb += dt * (W["q"] * q * q + W["R"] * ql)
        np.maximum(Fpk, Fn, out=Fpk); np.maximum(Ppk, S[IPR], out=Ppk); np.maximum(pmax, S[IP], out=pmax)
        qin += dt * q; ql_ += dt * ql
    dx = np.stack([S[i] - S0[i] for i in SX], -1)
    feats = np.stack([Jm, Jb, Fpk, Ppk, pmax, qin, ql_, S[ID] - D0], -1)
    return S, np.concatenate([dx, feats], -1)


class MLP(nn.Module):
    def __init__(self, n_in=N_IN, n_out=N_OUT, width=128, depth=3):
        super().__init__()
        self.inp = nn.Linear(n_in, width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU())
                                     for _ in range(depth)])
        self.out = nn.Linear(width, n_out)

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(h)  # residual skip connections
        return self.out(h)


class BlockSurrogate:
    def __init__(self, width=128, depth=3):
        self.net = MLP(width=width, depth=depth)
        self.mu_x = self.sd_x = self.mu_y = self.sd_y = None

    # -------------------------------------------------------- training
    def fit(self, X, Y, Xv, Yv, epochs=200, bs=1024, lr=1e-3, wd=1e-4, patience=20,
            lam=(1.0, 0.1, 10.0), verbose=True, max_minutes=None):
        """Loss = lam_traj*MSE + lam_phys*L_phys + lam_bound*L_bound (Eq. surrogate loss)."""
        self.mu_x = X.mean(0); self.sd_x = X.std(0) + 1e-6
        self.mu_y = Y.mean(0); self.sd_y = Y.std(0) + 1e-6
        tX = torch.tensor((X - self.mu_x) / self.sd_x, dtype=torch.float32)
        tY = torch.tensor((Y - self.mu_y) / self.sd_y, dtype=torch.float32)
        vX = torch.tensor((Xv - self.mu_x) / self.sd_x, dtype=torch.float32)
        vY = torch.tensor((Yv - self.mu_y) / self.sd_y, dtype=torch.float32)
        Xraw = torch.tensor(X, dtype=torch.float32)
        my = torch.tensor(self.mu_y, dtype=torch.float32); sy = torch.tensor(self.sd_y, dtype=torch.float32)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        best = (np.inf, None); bad = 0; hist = []
        iV = OUT_NAMES.index("dV"); iA = OUT_NAMES.index("dA"); iqin = OUT_NAMES.index("qin"); iql = OUT_NAMES.index("qlost")
        iFp = OUT_NAMES.index("Fpk")
        xV = SX.index(IV); xA = SX.index(IA)
        t_start = time.time()
        n = len(tX)
        for ep in range(epochs):
            self.net.train()
            perm = torch.randperm(n)
            tot = 0.0
            for s in range(0, n, bs):
                idx = perm[s:s + bs]
                pred = self.net(tX[idx])
                l_traj = ((pred - tY[idx]) ** 2).mean()
                yr = pred * sy + my
                V1 = Xraw[idx, xV] + yr[:, iV]; A1 = Xraw[idx, xA] + yr[:, iA]
                # bound penalties: non-negative volume, area, flows and force peak
                l_bound = (torch.relu(-V1) ** 2 + torch.relu(-A1) ** 2 + torch.relu(-yr[:, iqin]) ** 2
                           + torch.relu(-yr[:, iql]) ** 2 + torch.relu(-yr[:, iFp]) ** 2).mean()
                # physics: volume gained in a block cannot exceed the volume injected
                l_phys = (torch.relu(yr[:, iV] - yr[:, iqin]) ** 2).mean()
                loss = lam[0] * l_traj + lam[1] * l_phys + lam[2] * l_bound
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(idx)
            sched.step()
            self.net.eval()
            with torch.no_grad():
                vl = ((self.net(vX) - vY) ** 2).mean().item()
            hist.append((tot / n, vl))
            if vl < best[0] - 1e-6:
                best = (vl, {k: v.clone() for k, v in self.net.state_dict().items()}); bad = 0
            else:
                bad += 1
            if verbose and ep % 10 == 0:
                print(f"  epoch {ep} train {tot/n:.4f} val {vl:.4f}", flush=True)
            if bad >= patience:
                break
            if max_minutes and (time.time() - t_start) > 60 * max_minutes:
                break
        self.net.load_state_dict(best[1])
        self.train_time = time.time() - t_start
        self.epochs_run = ep + 1
        self.history = hist
        return best[0]

    def predict(self, Xin):
        with torch.no_grad():
            x = torch.tensor(((Xin - self.mu_x) / self.sd_x).reshape(-1, N_IN), dtype=torch.float32)
            y = self.net(x).numpy().astype(np.float64)
        return (y * self.sd_y + self.mu_y).reshape(Xin.shape[:-1] + (N_OUT,))

    # -------------------------------------------------------- planner interface
    def rollout(self, x0, th, mot, U, ctx, t0, bleb_terms=True):
        B, P_, N, _ = U.shape; M = x0.shape[1]
        shp = (B, P_, M)
        S = [np.ascontiguousarray(np.broadcast_to(x0[:, None, :, i], shp)).copy() for i in range(NX)]
        thb = {k: np.broadcast_to(np.asarray(th[k])[:, None], shp) for k in PKEYS}
        motb = {k: np.asarray(v)[:, None] for k, v in mot.items()}
        J = np.zeros(shp); Fpk = np.zeros(shp); Ppk = S[IPR].copy(); pmax = S[IP].copy()
        qin = np.zeros(shp); ql = np.zeros(shp)
        refp = np.broadcast_to(ctx["ref_prev"][:, None, None, :], shp + (4,))
        for i in range(N):
            ref = np.broadcast_to(U[:, :, i][:, :, None, :], shp + (4,))
            X = block_inputs(S, ref, refp, motb, i * PR.DT_PLAN, thb)
            y = self.predict(X)
            for j, k in enumerate(SX):
                S[k] = S[k] + y[..., j]
            S[IV] = np.maximum(S[IV], 0); S[IA] = np.maximum(S[IA], 0)
            J += y[..., 9] + (y[..., 10] if bleb_terms else 0.0)
            Fpk = np.maximum(Fpk, y[..., 11]); Ppk = np.maximum(Ppk, y[..., 12]); pmax = np.maximum(pmax, y[..., 13])
            qin += np.maximum(y[..., 14], 0); ql += np.maximum(y[..., 15], 0)
            refp = ref
        W = C.W
        Uc = np.concatenate([np.broadcast_to(ctx["ref_prev"][:, None, None], (B, P_, 1, 4)), U], 2)
        dU = np.diff(Uc, axis=2)
        J += (W["du"] * (dU[..., 0] ** 2 + dU[..., 1] ** 2) + W["dq"] * dU[..., 3] ** 2).sum(-1)[..., None]
        if bleb_terms:
            tend = t0 + N * PR.DT_PLAN
            V, A, P = S[IV], S[IA], S[IPR]
            J += W["V"] * np.maximum(C.v_ref(tend) - V, 0) ** 2 + W["Vover"] * np.maximum(V - PR.V_STAR, 0) ** 2
            J += W["A"] * np.maximum(PR.A_STAR * C.v_ref(tend) / PR.V_STAR - A, 0) ** 2
            J += W["Pend"] * np.maximum(P - 1.0, 0) ** 2
            r_req = (PR.V_MIN + 1.0 - V) / max(PR.T_MAX - tend, 0.2)
            J += W["ttg"] * np.maximum(r_req - C.Q_SUST, 0) ** 2
        dmin = thb["h"] - pmax
        Qin0 = ctx.get("Qin0", np.zeros(B))[:, None, None]; Ql0 = ctx.get("Ql0", np.zeros(B))[:, None, None]
        RRh = (Ql0 + ql) / (Qin0 + qin + 1.0)
        R = (np.maximum(Fpk / PR.F_SAFE - 0.80, 0) + np.maximum((PR.DELTA_RPE + C.D_SOFT - dmin) / PR.DELTA_RPE, 0)
             + np.maximum(Ppk / PR.P_MAX - 0.88, 0) + np.maximum(RRh - 0.8 * PR.RR_MAX, 0))
        return J, R, None

    def save(self, path):
        torch.save(dict(state=self.net.state_dict(), mu_x=self.mu_x, sd_x=self.sd_x, mu_y=self.mu_y, sd_y=self.sd_y), path)

    @classmethod
    def load(cls, path, width=128, depth=3):
        d = torch.load(path, weights_only=False)
        s = cls(width, depth); s.net.load_state_dict(d["state"]); s.net.eval()
        s.mu_x, s.sd_x, s.mu_y, s.sd_y = d["mu_x"], d["sd_x"], d["mu_y"], d["sd_y"]
        return s
