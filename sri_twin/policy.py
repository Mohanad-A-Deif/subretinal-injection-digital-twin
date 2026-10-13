"""Explicit (distilled) receding-horizon policy for the 20 ms control period.

The risk-aware scenario MPC acts as an offline expert.  A compact MLP is fitted
to the expert's first-block set-points by dataset aggregation (DAgger) on
training seeds, and is executed at every 20 ms step followed by the same
tracking law (and, optionally, the safety filter)."""
import time, sys
import numpy as np
import torch
import torch.nn as nn
from . import params as PR
from . import episode as E
from .estimator import ZMN, ZMDN, ZMS, ZMDS, ZV, ZA, ZP, ZBF, ZLK, ZLA, ZLB, ZLT, ZH
from .controller import tracking_law, REF_LO, REF_HI, REF_MARGIN, NominalController

torch.set_num_threads(2)
N_OBS = 25


def features(t, ukf, y, Qin, ref_prev):
    z = ukf.z; P = ukf.P
    p_hat = y["n_enc"] - z[:, ZMN]; es_hat = y["s_enc"] - z[:, ZMS]
    f = [p_hat, es_hat, y["vn_enc"], y["vs_enc"], y["phi_enc"], y["om_enc"], z[:, ZV], z[:, ZA], z[:, ZP],
         z[:, ZMDN], z[:, ZMDS], z[:, ZLK], z[:, ZLA], z[:, ZLB], z[:, ZLT], z[:, ZH] - p_hat,
         np.sqrt(np.maximum(P[:, ZMN, ZMN], 0)), np.sqrt(np.maximum(P[:, ZH, ZH], 0)),
         np.full_like(p_hat, t), Qin, np.maximum(Qin - z[:, ZV], 0)]
    f += [ref_prev[:, k] for k in range(4)]
    return np.stack(f, -1)


class Net(nn.Module):
    def __init__(self, width=128, depth=3):
        super().__init__()
        layers = [nn.Linear(N_OBS, width), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.GELU()]
        layers += [nn.Linear(width, 4)]
        self.f = nn.Sequential(*layers)

    def forward(self, x):
        return self.f(x)


class DistilledPolicy:
    def __init__(self):
        self.net = Net(); self.mu = self.sd = self.my = self.sy = None
        self.reset(1)

    def reset(self, B):
        self.Qin = np.zeros(B); self.ref_prev = None

    def fit(self, O, Y, epochs=150, bs=512, lr=1e-3):
        self.mu = O.mean(0); self.sd = O.std(0) + 1e-6; self.my = Y.mean(0); self.sy = Y.std(0) + 1e-6
        X = torch.tensor((O - self.mu) / self.sd, dtype=torch.float32)
        T = torch.tensor((Y - self.my) / self.sy, dtype=torch.float32)
        n = len(X); iv = torch.randperm(n)[: n // 10]; it = torch.ones(n, dtype=bool); it[iv] = False
        Xt, Tt, Xv, Tv = X[it], T[it], X[iv], T[iv]
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        best = (np.inf, None)
        for ep in range(epochs):
            perm = torch.randperm(len(Xt))
            for s in range(0, len(Xt), bs):
                i = perm[s:s + bs]
                loss = ((self.net(Xt[i]) - Tt[i]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            sch.step()
            with torch.no_grad():
                vl = ((self.net(Xv) - Tv) ** 2).mean().item()
            if vl < best[0]:
                best = (vl, {k: v.clone() for k, v in self.net.state_dict().items()})
        self.net.load_state_dict(best[1])
        return best[0]

    def ref(self, O):
        with torch.no_grad():
            r = self.net(torch.tensor((O - self.mu) / self.sd, dtype=torch.float32)).numpy()
        return r * self.sy + self.my

    def act(self, t, ukf, y, done=None):
        B = len(y["n_enc"])
        if self.ref_prev is None or len(self.ref_prev) != B:
            self.Qin = np.zeros(B)
            self.ref_prev = np.zeros((B, 4)); self.ref_prev[:, 1] = y["n_enc"] - ukf.z[:, ZMN]
        O = features(t, ukf, y, self.Qin, self.ref_prev)
        r = np.clip(self.ref(O), REF_LO, REF_HI)
        r[:, 1] = np.minimum(r[:, 1], ukf.z[:, ZH] - PR.DELTA_RPE - REF_MARGIN)
        self.ref_prev = r
        p_hat = y["n_enc"] - ukf.z[:, ZMN]; es_hat = y["s_enc"] - ukf.z[:, ZMS]
        u = tracking_law(r, p_hat, es_hat, y["phi_enc"], ukf.z[:, ZMDN], ukf.z[:, ZMDS])
        self.Qin = self.Qin + PR.DT_CTRL * u[:, 3]
        return u

    def save(self, path):
        torch.save(dict(s=self.net.state_dict(), mu=self.mu, sd=self.sd, my=self.my, sy=self.sy), path)

    @classmethod
    def load(cls, path):
        d = torch.load(path, weights_only=False); p = cls()
        p.net.load_state_dict(d["s"]); p.mu, p.sd, p.my, p.sy = d["mu"], d["sd"], d["my"], d["sy"]
        return p


def dagger_run(seed, B, policy, beta, rng):
    """Closed loop where the executed set-points come from the expert with prob. beta
    (per episode) and from the policy otherwise; expert labels are recorded at every block."""
    from .plant import Plant
    from .estimator import UKF
    theta = PR.sample_theta(np.random.default_rng(seed), B, "medium")
    plant = Plant(theta, np.random.default_rng(seed + 7919))
    y = plant.observe(); nom = NominalController(y)
    ukf = UKF(B, y)
    planner = E.make_planner(dict(mode="joint", lam=0.75), B, np.random.default_rng(seed + 104729))
    use_exp = rng.random(B) < beta
    O_all, Y_all = [], []
    q_prev = np.zeros(B); enc_prev = y
    pol_ref_prev = None
    Qin = np.zeros(B)
    nsteps = int(round(PR.T_MAX / PR.DT_CTRL))
    ref_e = None
    for k in range(nsteps):
        t = k * PR.DT_CTRL
        if k > 0:
            ukf.predict(enc_prev, q_prev)
        ukf.update(y)
        if pol_ref_prev is None:
            pol_ref_prev = np.zeros((B, 4)); pol_ref_prev[:, 1] = y["n_enc"] - ukf.z[:, ZMN]
        if k % PR.BLOCK == 0:
            ref_e = planner.plan(t, ukf, y, nom)
            O = features(t, ukf, y, Qin, pol_ref_prev)
            act = ~plant.done
            O_all.append(O[act]); Y_all.append(ref_e[act])
        if policy is not None:
            O = features(t, ukf, y, Qin, pol_ref_prev)
            r_p = np.clip(policy.ref(O), REF_LO, REF_HI)
            r_p[:, 1] = np.minimum(r_p[:, 1], ukf.z[:, ZH] - PR.DELTA_RPE - REF_MARGIN)
        else:
            r_p = ref_e
        r = np.where(use_exp[:, None], ref_e, r_p)
        pol_ref_prev = r
        p_hat = y["n_enc"] - ukf.z[:, ZMN]; es_hat = y["s_enc"] - ukf.z[:, ZMS]
        u = tracking_law(r, p_hat, es_hat, y["phi_enc"], ukf.z[:, ZMDN], ukf.z[:, ZMDS])
        plant.step(u)
        Qin = Qin + PR.DT_CTRL * u[:, 3] * (~plant.done)
        planner.Qin = Qin.copy()
        enc_prev = y; q_prev = u[:, 3]; y = plant.observe()
        if plant.done.all():
            break
    return np.concatenate(O_all), np.concatenate(Y_all)


def train_dagger(n_iter=3, B=50, batches_per_iter=2, out="results/models/policy.pt"):
    t0 = time.time(); rng = np.random.default_rng(1)
    O, Y = [], []
    pol = None; log = []
    for it in range(n_iter):
        beta = 1.0 if it == 0 else 0.5 ** it
        for b in range(batches_per_iter):
            o, y_ = dagger_run(40_000 + it * 100 + b, B, pol, beta, rng)
            O.append(o); Y.append(y_)
        pol = DistilledPolicy()
        vl = pol.fit(np.concatenate(O), np.concatenate(Y))
        log.append(dict(iter=it, n=int(sum(len(o) for o in O)), val_mse=vl, t=time.time() - t0))
        print("DAgger iter", it, log[-1], flush=True)
    pol.save(out)
    np.save(out.replace(".pt", "_log.npy"), log, allow_pickle=True)
    return pol, log


if __name__ == "__main__":
    train_dagger()
