"""Statistics for the held-out evaluation: rates with bootstrap CIs, paired exact
McNemar tests for binary outcomes, Wilcoxon signed-rank tests for continuous
outcomes, and Holm adjustment within each comparison family."""
import json
import numpy as np
from scipy.stats import wilcoxon, binomtest

RNG = np.random.default_rng(12345)


def load(tag):
    d = np.load(f"results/{tag}.npz")
    return {k: d[k] for k in d.files}


def boot_ci(x, n=5000, stat=np.mean):
    x = np.asarray(x, float)
    idx = RNG.integers(0, len(x), (n, len(x)))
    s = stat(x[idx], axis=1)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def summary(m):
    out = {}
    for k in ["viol", "success", "v_force", "v_depth", "v_pressure", "v_reflux"]:
        x = m[k].astype(float)
        out[k] = dict(mean=100 * x.mean(), ci=[100 * c for c in boot_ci(x)])
    for k, sc in [("RR", 100), ("IE", 100), ("S", 100), ("Fmax", 1), ("Pmax", 1), ("dmin", 1), ("D", 1), ("En", 1), ("Es", 1), ("t_end", 1), ("V", 1), ("A", 1)]:
        x = m[k].astype(float) * sc
        out[k] = dict(mean=float(x.mean()), sd=float(x.std(ddof=1)), median=float(np.median(x)),
                      iqr=[float(np.percentile(x, 25)), float(np.percentile(x, 75))])
    out["n"] = int(len(m["viol"]))
    if "plan_time" in m and len(m["plan_time"]):
        out["plan_time_ms"] = float(1e3 * m["plan_time"].mean())
    return out


def mcnemar(a, b):
    a = a.astype(bool); b = b.astype(bool)
    n01 = int((~a & b).sum()); n10 = int((a & ~b).sum())
    if n01 + n10 == 0:
        return 1.0, n01, n10
    return float(binomtest(n10, n01 + n10, 0.5).pvalue), n01, n10


def wil(a, b):
    d = a - b
    if np.allclose(d, 0):
        return 1.0
    return float(wilcoxon(a, b, zero_method="pratt").pvalue)


def holm(pvals):
    p = np.asarray(pvals); m = len(p); order = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj.tolist()


def compare(ref, others):
    """Paired tests of the reference configuration against each comparator."""
    rows = {}
    for name, m in others.items():
        n = min(len(ref["viol"]), len(m["viol"]))
        r = {k: v[:n] for k, v in ref.items() if np.ndim(v) and len(v) >= n}
        o = {k: v[:n] for k, v in m.items() if np.ndim(v) and len(v) >= n}
        p_vr, n01, n10 = mcnemar(r["viol"], o["viol"])
        p_su, _, _ = mcnemar(r["success"], o["success"])
        rows[name] = dict(p_viol=p_vr, n_ref_only=n10, n_cmp_only=n01, p_success=p_su,
                          p_RR=wil(r["RR"], o["RR"]), p_IE=wil(r["IE"], o["IE"]), p_Fmax=wil(r["Fmax"], o["Fmax"]),
                          dVR=float(100 * (r["viol"].mean() - o["viol"].mean())),
                          dSucc=float(100 * (r["success"].mean() - o["success"].mean())))
    for key in ["p_viol", "p_success", "p_RR", "p_IE", "p_Fmax"]:
        names = list(rows)
        adj = holm([rows[n][key] for n in names])
        for n, a in zip(names, adj):
            rows[n][key + "_holm"] = a
    return rows
