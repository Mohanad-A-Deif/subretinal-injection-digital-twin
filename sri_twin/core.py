"""Shared mechanistic core of the digital twin (needle micro-dynamics in the
retina-attached frame, Kelvin--Voigt interaction, tract openness, reduced bleb
dynamics).  Used by the truth plant (1 ms) and by the planner rollouts (10 ms).

State layout (relative frame), last axis:
 0 p    needle-tip penetration below the retinal surface (um)
 1 es   lateral tracking error (um)
 2 vn   normal needle velocity (um/s, robot frame)
 3 vs   tangential needle velocity (um/s)
 4 phi  orientation deviation from preferred insertion angle (rad)
 5 om   angular velocity (rad/s)
 6 V    bleb volume (uL)
 7 A    bleb spread area (mm^2)
 8 P    pressure surrogate (normalized)
 9 D    cumulative tissue-stress surrogate
10 Qin  cumulative injected volume (uL)
11 Qrf  cumulative lost volume (reflux + vitreous loss, uL)
Inputs u = [us, un, uphi, q].
"""
import numpy as np
from .params import NOM, F_SAFE, F_SOFT, DDELTA_SAFE

NX = 12
PHI_LIM = 0.35
IP, IES, IVN, IVS, IPHI, IOM, IV, IA, IPR, ID, IQIN, IQRF = range(NX)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def contact_force(p, pdot, kk, variant="kv"):
    d = np.maximum(p, 0.0)
    dd = np.where(p > 0, pdot, 0.0)
    if variant == "kv":
        F = kk * ((NOM["k0"] + NOM["k1"] * d + NOM["k2"] * d * d) * d
                  + (NOM["c0"] + NOM["c1"] * d) * dd)
    elif variant == "puncture":
        # structural mismatch: membrane rupture at d_rup, then friction-dominated reaction
        d_rup = 60.0
        pre = kk * (0.06 * d + 3e-4 * d * d) * (d < d_rup)
        post = kk * (2.0 + 0.012 * np.maximum(d - d_rup, 0) + 6e-5 * np.maximum(d - d_rup, 0) ** 2) * (d >= d_rup)
        F = pre + post + kk * NOM["c0"] * dd
    else:
        raise ValueError(variant)
    return np.maximum(F, 0.0)


def deriv(x, u, th, mdn, mds, variant=None, Fextra=None):
    """Time derivative of the twin state.

    x: (..., 12); u: (..., 4); th: dict of arrays broadcastable to x[...,0];
    mdn, mds: retinal motion velocities (normal, tangential).
    variant: dict selecting structural alternatives (truth plant only).
    Returns (xdot, aux) where aux holds algebraic outputs (forces, flows).
    """
    variant = variant or {}
    p, es, vn, vs, phi, om, V, A, P = (x[..., i] for i in range(9))
    us, un, uphi, q = (u[..., i] for i in range(4))
    h = th["h"]
    pdot = vn - mdn
    esdot = vs - mds
    Fn = contact_force(p, pdot, th["kk"], variant.get("contact", "kv"))
    if Fextra is not None:
        Fn = Fn + Fextra
    Ft = th["mu"] * Fn * np.tanh(esdot / NOM["v_eps"])
    vndot = (un - vn) / NOM["tau_r"] - NOM["g_c"] * Fn * (p > 0)
    vsdot = (us - vs) / NOM["tau_r"] - NOM["g_c"] * Ft
    omdot = (uphi - om) / NOM["tau_phi"]
    # mechanical limit of the insertion-angle stage
    omdot = np.where((phi > PHI_LIM) & (uphi > 0) | (phi < -PHI_LIM) & (uphi < 0), -om / NOM["tau_phi"] * 5, omdot)
    # delivery zone and tract openness
    psi = sigmoid((p - (h - NOM["d_zone"])) / NOM["w_zone"])
    dseat = np.clip(p / h, 0.0, 1.0)
    vrel = np.sqrt(pdot * pdot + esdot * esdot) / 100.0
    g = np.exp(-NOM["beta_d"] * dseat) * (1 + NOM["beta_phi"] * phi * phi) * (1 + NOM["beta_v"] * vrel)
    vol_guard = np.clip(V / 1.0, 0.0, 1.0)
    q_ref = th["alpha_r"] * np.maximum(P - th["P_tract"], 0.0) * g * vol_guard
    q_leak = th["alpha_l"] * np.maximum(P - th["P_l"], 0.0) * vol_guard
    q_in = psi * q
    Vdot = q_in - q_ref - q_leak
    if variant.get("bleb", "dyn") == "dyn":
        Adot = th["eta_A"] * np.maximum(P - NOM["P_sep"], 0.0) * (1 - A / NOM["A_max"]) - th["lam_A"] * A
        Pdot = (th["beta_q"] * q_in - th["beta_A"] * Adot - th["beta_V"] * Vdot - (P - NOM["P_base"])) / th["tau_P"]
    else:
        # structural mismatch: compliance-type bleb, pressure set by stretched volume
        # and flow resistance, area slaved to volume (thin-bleb geometry)
        A_eq = NOM["A_max"] * (1 - np.exp(-V / 18.0))
        Adot = (A_eq - A) / 0.3
        P_eq = NOM["P_base"] + 0.9 * (V / 30.0) ** 2 / (1 + A / 10.0) + 0.11 * th["beta_q"] / NOM["beta_q"] * q_in
        Pdot = (P_eq - P) / (0.5 * th["tau_P"])
    ddel = np.where(p > 0, pdot, 0.0)
    Ddot = ((np.maximum(Fn - F_SOFT, 0.0) / (F_SAFE - F_SOFT)) ** NOM["p_pen"]
            + NOM["gamma_d"] * (np.maximum(ddel - DDELTA_SAFE, 0.0) / DDELTA_SAFE) ** 2)
    xd = np.stack([pdot, esdot, vndot, vsdot, om, omdot, Vdot, Adot, Pdot, Ddot,
                   q, q_ref + (1 - psi) * q], axis=-1)
    aux = dict(Fn=Fn, Ft=Ft, q_ref=q_ref, q_leak=q_leak, psi=psi, q_lost=q_ref + (1 - psi) * q)
    return xd, aux


def clamp_state(x):
    x[..., IV] = np.maximum(x[..., IV], 0.0)
    x[..., IA] = np.maximum(x[..., IA], 0.0)
    return x


def deriv_fast(S, us, un, uphi, q, th, mdn, mds, c):
    """Allocation-light version of deriv() for planner rollouts (nominal structure only).
    S: list of 12 state arrays; returns list of 12 derivative arrays and aux tuple.
    c: dict of scalar constants (from NOM). Numerically identical to deriv()."""
    p, es, vn, vs, phi, om, V, A, P = S[0], S[1], S[2], S[3], S[4], S[5], S[6], S[7], S[8]
    h = th["h"]; kk = th["kk"]
    pdot = vn - mdn
    esdot = vs - mds
    pos = p > 0
    d = np.maximum(p, 0.0)
    dd = np.where(pos, pdot, 0.0)
    Fn = np.maximum(kk * ((c["k0"] + c["k1"] * d + c["k2"] * d * d) * d + (c["c0"] + c["c1"] * d) * dd), 0.0)
    Ft = th["mu"] * Fn * np.tanh(esdot / c["v_eps"])
    vndot = (un - vn) / c["tau_r"] - c["g_c"] * Fn * pos
    vsdot = (us - vs) / c["tau_r"] - c["g_c"] * Ft
    omdot = (uphi - om) / c["tau_phi"]
    lim = ((phi > PHI_LIM) & (uphi > 0)) | ((phi < -PHI_LIM) & (uphi < 0))
    omdot = np.where(lim, -om / c["tau_phi"] * 5, omdot)
    psi = 1.0 / (1.0 + np.exp(-np.clip((p - (h - c["d_zone"])) / c["w_zone"], -40, 40)))
    dseat = np.clip(p / h, 0.0, 1.0)
    vrel = np.sqrt(pdot * pdot + esdot * esdot) / 100.0
    g = np.exp(-c["beta_d"] * dseat) * (1 + c["beta_phi"] * phi * phi) * (1 + c["beta_v"] * vrel)
    vg = np.clip(V, 0.0, 1.0)
    q_ref = th["alpha_r"] * np.maximum(P - th["P_tract"], 0.0) * g * vg
    q_leak = th["alpha_l"] * np.maximum(P - th["P_l"], 0.0) * vg
    q_in = psi * q
    Vdot = q_in - q_ref - q_leak
    Adot = th["eta_A"] * np.maximum(P - c["P_sep"], 0.0) * (1 - A / c["A_max"]) - th["lam_A"] * A
    Pdot = (th["beta_q"] * q_in - th["beta_A"] * Adot - th["beta_V"] * Vdot - (P - c["P_base"])) / th["tau_P"]
    Ddot = ((np.maximum(Fn - F_SOFT, 0.0) / (F_SAFE - F_SOFT)) ** 2
            + c["gamma_d"] * (np.maximum(dd - DDELTA_SAFE, 0.0) / DDELTA_SAFE) ** 2)
    q_lost = q_ref + (1 - psi) * q
    return [pdot, esdot, vndot, vsdot, om, omdot, Vdot, Adot, Pdot, Ddot, q, q_lost], (Fn, q_lost)
