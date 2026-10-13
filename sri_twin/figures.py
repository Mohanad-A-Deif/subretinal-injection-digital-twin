"""Figures for the revised manuscript (dark academic palette, marker/hatch redundancy)."""
import json, pickle, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .analysis import load, boot_ci

COL = {"nominal": "#000000", "traj": "#00308F", "inf": "#006400", "joint": "#8B0000", "policy": "#5B2C6F", "lam0.00": "#7A4A00"}
MK = {"nominal": "s", "traj": "^", "inf": "v", "joint": "o", "policy": "D", "lam0.00": "P"}
LAB = {"nominal": "Nominal fixed", "traj": "Trajectory-only", "inf": "Infusion-only", "joint": "Joint risk-aware (MPC)",
       "policy": "Joint, distilled policy", "lam0.00": "Joint, risk-neutral"}
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.6, "savefig.dpi": 300})
OUT = "figs"
os.makedirs(OUT, exist_ok=True)


def rate(m, k):
    x = m[k].astype(float); lo, hi = boot_ci(x)
    return 100 * x.mean(), 100 * lo, 100 * hi


def fig_tradeoff():
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for k in ["nominal", "traj", "inf", "joint", "lam0.00", "policy"]:
        m = load(k)
        vr, vlo, vhi = rate(m, "viol"); su, slo, shi = rate(m, "success")
        ax.errorbar(vr, su, xerr=[[vr - vlo], [vhi - vr]], yerr=[[su - slo], [shi - su]], fmt=MK[k], color=COL[k],
                    ms=8, capsize=3, lw=1.2, label=LAB[k])
    ax.set_xlabel("Safety-violation rate (%)"); ax.set_ylabel("Success rate (%)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_safety_quality_tradeoff.png"); plt.close(fig)


def fig_risk():
    lams = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
    tags = ["lam0.00", "lam0.25", "lam0.50", "joint", "lam1.00", "lam1.50"]
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for key, ax, yl in [("viol", axs[0], "Safety-violation rate (%)"), ("success", axs[1], "Success rate (%)")]:
        r = [rate({k: v[:200] for k, v in load(t).items() if np.ndim(v)}, key) for t in tags]
        y = np.array([a[0] for a in r]); lo = np.array([a[1] for a in r]); hi = np.array([a[2] for a in r])
        ax.errorbar(lams, y, yerr=[y - lo, hi - y], fmt="o-", color="#8B0000", capsize=3, label="Medium regime")
        hi_t = {0.0: "lam0.00_high", 0.75: "joint_high"}
        if all(os.path.exists(f"results/{t}.npz") for t in hi_t.values()):
            rr = [rate(load(hi_t[l]), key) for l in hi_t]
            yy = np.array([a[0] for a in rr])
            ax.errorbar(list(hi_t), yy, yerr=[yy - np.array([a[1] for a in rr]), np.array([a[2] for a in rr]) - yy],
                        fmt="s--", color="#00308F", capsize=3, label="High regime")
        ax.set_xlabel(r"Risk weight $\lambda$"); ax.set_ylabel(yl)
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig2_risk_settings_tradeoff.png"); plt.close(fig)


def fig_ablation():
    tags = ["joint", "lam0.00", "abl_nomap", "abl_decoupled", "abl_adaptive"]
    labs = ["Full framework", "w/o CVaR term", "w/o online MAP", "Decoupled", "+ adaptive covariance"]
    keys = [("v_depth", "Depth", "#00308F", ""), ("v_pressure", "Pressure", "#006400", "//"), ("v_reflux", "Reflux", "#8B0000", "\\\\")]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(tags)); bottom = np.zeros(len(tags))
    for k, lab, c, h in keys:
        v = np.array([100 * load(t)[k].mean() for t in tags])
        ax.bar(x, v, 0.6, bottom=bottom, color=c, hatch=h, edgecolor="white", linewidth=1.5, label=lab + " violations")
        bottom += v
    succ = [100 * load(t)["success"].mean() for t in tags]
    for i, s in enumerate(succ):
        ax.text(i, bottom[i] + 0.6, f"success {s:.1f}%", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8); ax.set_ylabel("Violations by type (% of episodes)")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(f"{OUT}/fig3_ablation_tradeoff.png"); plt.close(fig)


def fig_compute():
    lat = json.load(open("results/logs/latency.json"))
    lc = pickle.load(open("results/logs/surrogate_learning_curve.pkl", "rb"))
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.5))
    names = ["full_twin", "hybrid_1_4", "surrogate", "policy"]; labs = ["Full-twin MPC", "Hybrid 1:4", "Surrogate MPC", "Distilled policy"]
    cols = ["#8B0000", "#7A4A00", "#00308F", "#5B2C6F"]
    ax = axs[0]
    for i, (n, c) in enumerate(zip(names, cols)):
        ax.bar(i, lat[n]["p50"], 0.6, color=c)
        ax.errorbar(i, lat[n]["p50"], yerr=[[0], [lat[n]["p99"] - lat[n]["p50"]]], color="k", capsize=3)
    ax.axhline(100, color="k", ls="--", lw=1); ax.text(3.4, 110, "planning period 100 ms", ha="right", fontsize=8)
    ax.axhline(20, color="k", ls=":", lw=1); ax.text(3.4, 22, "control period 20 ms", ha="right", fontsize=8)
    ax.set_yscale("log"); ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=8, rotation=15)
    ax.set_ylabel("Latency per decision (ms)")
    ax = axs[1]
    n = [r["n"] / 1e3 for r in lc]
    ax.plot(n, [r["spearman"] for r in lc], "o-", color="#8B0000", label="Spearman rank correlation")
    ax.plot(n, [r["top1"] for r in lc], "s--", color="#00308F", label="Top-1 agreement")
    ax.set_xscale("log"); ax.set_xticks(n); ax.set_xticklabels([f"{int(v)}" for v in n]); ax.minorticks_off(); ax.set_xlabel("Training transitions (thousands)"); ax.set_ylabel("Planning fidelity")
    ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig4_surrogate_speed_fidelity.png"); plt.close(fig)


def fig_regimes():
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    for k in ["nominal", "traj", "inf", "joint"]:
        ys, lo, hi = [], [], []
        for t in [f"{k}_low", k, f"{k}_high"]:
            m = load(t); m = {a: b[:200] for a, b in m.items() if np.ndim(b)}
            r = rate(m, "viol"); ys.append(r[0]); lo.append(r[1]); hi.append(r[2])
        ys = np.array(ys)
        ax.errorbar([0, 1, 2], ys, yerr=[ys - np.array(lo), np.array(hi) - ys], fmt=MK[k] + "-", color=COL[k], capsize=3, label=LAB[k])
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["Low", "Medium", "High"]); ax.set_xlabel("Uncertainty regime")
    ax.set_ylabel("Safety-violation rate (%)"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig5_robustness_uncertainty_regimes.png"); plt.close(fig)


def fig_ood():
    vs = ["puncture", "compliance", "saccade", "heavydelay", "driftdelay", "dropout", "shift"]
    vl = ["Puncture\ncontact", "Compliance\nbleb", "Saccadic\njumps", "Heavy-tailed\ndelay", "Drifting\ndelay", "10%\ndropout", "Abrupt\nshift"]
    ms = ["nominal", "inf", "joint"]
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    w = 0.26
    for i, k in enumerate(ms):
        v = [100 * load(f"{k}_ood_{x}")["viol"].mean() for x in vs]
        ax.bar(np.arange(len(vs)) + (i - 1) * w, v, w, color=COL[k], label=LAB[k], edgecolor="white", linewidth=1)
    ax.set_xticks(range(len(vs))); ax.set_xticklabels(vl, fontsize=8); ax.set_ylabel("Safety-violation rate (%)")
    ax.legend(frameon=False, fontsize=8, ncol=2); fig.tight_layout(); fig.savefig(f"{OUT}/fig6_structural_mismatch.png"); plt.close(fig)


def fig_estimation():
    est = pickle.load(open("results/logs/joint_est_est.pkl", "rb"))
    t = np.array([e["t"] for e in est[0]])
    def stack(key):
        return np.concatenate([np.array([e[key] for e in ep])[: len(t)] for ep in est], 1)
    done = stack("done")
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.4))
    for key, lab, c in [("p_err", r"penetration $p$ ($\mu$m)", "#8B0000"), ("h_err", r"thickness $h$ ($\mu$m)", "#00308F")]:
        e = np.where(done, np.nan, stack(key)); r = np.sqrt(np.nanmean(e ** 2, 1))
        axs[0].plot(t, r, color=c, label=lab)
    axs[0].set_xlabel("Time (s)"); axs[0].set_ylabel("RMSE"); axs[0].legend(frameon=False, fontsize=8)
    for key, lab, c, ls in [("kk_err", r"$\kappa_k$", "#000000", "-"), ("ar_err", r"$\alpha_r$", "#8B0000", "--"),
                            ("bq_err", r"$\beta_q$", "#006400", "-."), ("pt_err", r"$P_{\mathrm{tract}}$", "#00308F", ":")]:
        e = np.where(done, np.nan, stack(key)); r = 100 * np.sqrt(np.nanmean(e ** 2, 1))
        axs[1].plot(t, r, color=c, ls=ls, label=lab)
    axs[1].set_xlabel("Time (s)"); axs[1].set_ylabel("Relative RMSE (%)"); axs[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig7_estimation_errors.png"); plt.close(fig)


if __name__ == "__main__":
    import sys
    for f in sys.argv[1:]:
        globals()["fig_" + f]()
