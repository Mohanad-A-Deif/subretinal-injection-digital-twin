"""Single-procedure (B=1) computation latency of each controller layer."""
import time, json, sys
import numpy as np
from . import params as PR
from .episode import run_batch
from .surrogate import BlockSurrogate
from .policy import DistilledPolicy

def run(n_eps=5):
    sur = BlockSurrogate.load("results/models/surrogate_1.0.pt")
    pol = DistilledPolicy.load("results/models/policy.pt")
    cfgs = {"full_twin": dict(method="mpc", mode="joint", lam=0.75, inner="twin"),
            "surrogate": dict(method="mpc", mode="joint", lam=0.75, inner="surrogate"),
            "hybrid_1_4": dict(method="mpc", mode="joint", lam=0.75, inner="hybrid", refine_every=4),
            "policy": dict(method="policy")}
    out = {}; raw = {}
    for name, cfg in cfgs.items():
        ts = []
        for s in range(n_eps):
            o = run_batch(cfg, 50_000 + 1000 + s, 1, surrogate=sur, policy=pol)
            ts.append(o["plan_time"])
        t = np.concatenate(ts) * 1e3
        budget = 20.0 if name == "policy" else 100.0
        out[name] = dict(n=int(len(t)), mean=float(t.mean()), p50=float(np.percentile(t, 50)),
                         p95=float(np.percentile(t, 95)), p99=float(np.percentile(t, 99)), max=float(t.max()),
                         budget_ms=budget, frac_over_budget=float((t > budget).mean()))
        raw[name] = t
        print(name, out[name], flush=True)
    # estimator update cost per 20 ms step
    from .plant import Plant
    from .estimator import UKF
    th = PR.sample_theta(np.random.default_rng(1), 1); pl = Plant(th, np.random.default_rng(2)); y = pl.observe(); ukf = UKF(1, y)
    tt = []
    for k in range(200):
        t0 = time.perf_counter(); ukf.predict(y, np.zeros(1)); ukf.update(y); tt.append(time.perf_counter() - t0)
        pl.step(np.zeros((1, 4))); y = pl.observe()
    t = np.array(tt) * 1e3
    out["ukf_step"] = dict(mean=float(t.mean()), p99=float(np.percentile(t, 99)))
    raw["ukf_step"] = t
    import platform, os
    out["hardware"] = dict(cpu=open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": "),
                           n_cpu=os.cpu_count(), python=platform.python_version())
    np.savez_compressed("results/logs/latency_raw_ms.npz", **raw)
    print("ukf", out["ukf_step"])
    json.dump(out, open("results/logs/latency.json", "w"), indent=1)

if __name__ == "__main__":
    run()
