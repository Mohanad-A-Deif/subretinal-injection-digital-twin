"""Nominal parameters, uncertainty sets and scenario sampling for the
robotic subretinal-injection digital twin.

Units: length um, time s, force mN, volume uL, area mm^2,
pressure normalized (dimensionless surrogate), angle rad.
"""
import numpy as np

# ---------------------------------------------------------------- timing
DT_INT = 1e-3        # fine integration step of the full twin (truth plant)
DT_CTRL = 0.02       # estimation / observation / low-level control step
BLOCK = 5            # control steps per planning block  (100 ms)
DT_PLAN = DT_CTRL * BLOCK
N_PLAN = 10          # planning horizon in blocks (1.0 s)
DT_SUB = 0.02        # integration step of the planner's full-twin rollouts
T_MAX = 10.0         # maximum episode duration

# ---------------------------------------------------------------- task / safety
V_STAR = 30.0        # target endpoint bleb volume (uL)
V_MIN = 0.85 * V_STAR
A_STAR = 10.0        # target bleb area (mm^2)
A_MIN = 6.0
F_SAFE = 12.0        # hard force limit (mN)
F_SOFT = 8.0         # soft penalty onset (mN)
DELTA_RPE = 35.0     # minimum tip-to-RPE margin (um)
D_TARGET = 70.0      # planned tip-to-RPE distance in the subretinal target zone (um)
P_MAX = 1.25         # hard pressure-surrogate limit
RR_MAX = 0.20        # maximum reflux ratio
DDELTA_SAFE = 400.0  # safe compression rate (um/s)

# ---------------------------------------------------------------- input bounds
U_LO = np.array([-800.0, -800.0, -0.5, 0.0])
U_HI = np.array([800.0, 800.0, 0.5, 8.0])

# ---------------------------------------------------------------- nominal model constants
NOM = dict(
    # needle--tissue interaction (Kelvin--Voigt, Eq. normal force)
    k0=0.020, k1=5.0e-5, k2=5.0e-8,        # mN/um, mN/um^2, mN/um^3
    c0=2.0e-3, c1=5.0e-6,                   # mN s/um, mN s/um^2
    mu=0.15, v_eps=20.0,
    g_c=50.0,                               # force-to-acceleration coupling (um/s^2 per mN)
    tau_r=0.03, tau_phi=0.05,               # robot translational / rotational lag (s)
    h=220.0,                                # local retinal thickness (um)
    # tract and delivery geometry
    beta_d=2.0, beta_phi=8.0, beta_v=0.5,  # tract-openness sensitivities
    d_zone=130.0, w_zone=10.0,              # delivery-zone logistic (um)
    # bleb dynamics
    alpha_r=12.0, P_tract=0.85,             # reflux conductance (uL/s), opening pressure
    alpha_l=2.0, P_l=1.00, sig_l=0.05,      # leakage
    eta_A=8.0, P_sep=0.50, A_max=25.0, lam_A=0.05,
    tau_P=0.40, beta_q=0.18, beta_A=0.03, beta_V=0.04, P_base=0.20,
    gamma_d=0.5, p_pen=2.0,
    r_c=0.02,                               # cannula line-pressure drop per uL/s (normalized)
)

# identified online (log-scale multipliers on nominal values) + retinal thickness
ID_NAMES = ["log_kk", "log_ar", "log_bq", "log_pt", "h"]

# ---------------------------------------------------------------- uncertainty regimes
REGIME_SCALE = {"low": 0.5, "medium": 1.0, "high": 1.6}


def _trunc_lognormal(rng, sigma, lo, hi, size):
    x = np.exp(rng.normal(0.0, sigma, size))
    bad = (x < lo) | (x > hi)
    while bad.any():
        x[bad] = np.exp(rng.normal(0.0, sigma, bad.sum()))
        bad = (x < lo) | (x > hi)
    return x


def _trunc_normal_pos(rng, mean, cv, size, lo_frac=0.3):
    x = rng.normal(mean, cv * mean, size)
    lo = lo_frac * mean
    bad = x < lo
    while bad.any():
        x[bad] = rng.normal(mean, cv * mean, bad.sum())
        bad = x < lo
    return x


def sample_theta(rng, B, regime="medium"):
    """Sample B truth parameter vectors theta ~ P_Theta (Table: uncertainty blocks)."""
    s = REGIME_SCALE[regime]
    th = {}
    kk = _trunc_lognormal(rng, 0.20 * s, 0.6 ** s, 1.6 ** s, B)
    th["kk"] = kk
    mu_half = 0.125 * s
    th["mu"] = rng.uniform(max(0.01, 0.175 - mu_half), 0.175 + mu_half, B)
    th["h"] = rng.uniform(220 - 40 * s, 220 + 40 * s, B)
    for k, cv in [("alpha_r", 0.15), ("P_tract", 0.15), ("alpha_l", 0.15), ("P_l", 0.15)]:
        th[k] = _trunc_normal_pos(rng, NOM[k], cv * s, B)
    for k, cv in [("eta_A", 0.20), ("lam_A", 0.20), ("tau_P", 0.20), ("beta_q", 0.20),
                  ("beta_A", 0.20), ("beta_V", 0.20)]:
        th[k] = _trunc_normal_pos(rng, NOM[k], cv * s, B)
    # retinal micro-motion: one respiratory + one pulsatile harmonic per axis
    # depth amplitudes anchored to supine heartbeat-induced axial retinal motion
    # (~21 um peak-to-peak, de Kinkelder et al. 2011); lateral amplitudes are design ranges
    th["a_s"] = np.stack([rng.uniform(10, 10 + 50 * s, B), rng.uniform(3, 3 + 12 * s, B)], 1)
    th["a_n"] = np.stack([rng.uniform(2, 2 + 8 * s, B), rng.uniform(3, 3 + 12 * s, B)], 1)
    th["f_s"] = np.stack([rng.uniform(0.15, 0.40, B), rng.uniform(0.8, 1.8, B)], 1)
    th["f_n"] = np.stack([rng.uniform(0.15, 0.40, B), rng.uniform(0.8, 1.8, B)], 1)
    th["ph_s"] = rng.uniform(0, 2 * np.pi, (B, 2))
    th["ph_n"] = rng.uniform(0, 2 * np.pi, (B, 2))
    th["ou_lam"] = 2.0 * (1 + rng.uniform(-0.25, 0.25, B))
    th["ou_sig"] = 6.0 * s * (1 + rng.uniform(-0.25, 0.25, B))
    # timing
    th["tau_a"] = rng.uniform(0.020, 0.020 + 0.040 * s, B)
    th["tau_F"] = rng.uniform(0.005, 0.005 + 0.015 * s, B)
    th["tau_D"] = rng.uniform(0.010, 0.010 + 0.025 * s, B)
    th["tau_P"] = rng.uniform(0.010, 0.010 + 0.020 * s, B)
    th["tau_S"] = rng.uniform(0.020, 0.020 + 0.040 * s, B)
    th["tau_V"] = rng.uniform(0.030, 0.030 + 0.040 * s, B)
    th["rho_F"] = rng.uniform(0.97, 0.995, B)
    th["sig_bF"] = 0.03 * s * np.ones(B)       # drift innovation std (mN)
    # initial state
    th["p0"] = -rng.uniform(40, 120, B)
    th["es0"] = rng.normal(0, 25, B)
    th["phi0"] = rng.uniform(-0.15, 0.15, B)
    return th


# sensor noise (Table: sensing configuration)
SENS = dict(sig_F=0.25, sig_D=6.0, sig_P=0.01, sig_S=8.0, sig_V0=0.3, sig_Vrel=0.05)


def prior_id():
    """Prior mean and covariance of the identified parameter block."""
    mean = np.array([0.0, 0.0, 0.0, 0.0, 220.0])
    std = np.array([0.20, 0.15, 0.20, 0.15, 40.0 / np.sqrt(3)])
    return mean, np.diag(std ** 2)
