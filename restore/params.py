"""Optimiser variables u <-> physical x, and random starting designs.

x = [nd, vd] * NE + [c, A4/1e-5, A6/1e-8, A8/1e-11] * NA + [back gap]   (config.py)
u: glass columns through glassmap.u_to_glass (CODE V-like (nd, dN) or sigmoid box), asphere curvature and back gap
through sigmoids onto their ranges, asphere coefficients linear."""
import numpy as np, torch
from scipy.optimize import least_squares
import config as CF
import core as T
import glassmap as GMP
from glass import model_n

dev, DT = T.dev, T.DT
NE, NA, NP = CF.NE, CF.NA, CF.NP


def params_to_phys(u):
    """u (B,NP) -> dict nd, vd (B,NE); c (B,NA); A (B,NA,3); t16 (B,) (back gap: last surface -> cover glass)"""
    sg = torch.sigmoid
    nd, vd = GMP.u_to_glass(u[:, 0:2 * NE:2], u[:, 1:2 * NE:2])
    cs, As = [], []
    for s in CF.ASPH_VAR:
        i = CF.IA[s]; lo, hi = CF.C_RANGE[s]
        cs.append(lo + (hi - lo) * sg(u[:, i])); As.append(u[:, i + 1:i + 4] * T.ASC)
    B = u.shape[0]
    c = torch.stack(cs, 1) if cs else torch.zeros(B, 0, dtype=u.dtype, device=u.device)
    A = torch.stack(As, 1) if As else torch.zeros(B, 0, 3, dtype=u.dtype, device=u.device)
    t16 = CF.T_LO + (CF.T_HI - CF.T_LO) * sg(u[:, CF.IT])
    return dict(nd=nd, vd=vd, c=c, A=A, t16=t16)


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


def phys_to_u(x):
    """x numpy (NP,) -> u numpy (NP,)"""
    x = np.asarray(x, float); u = np.zeros(NP)
    u[0:2 * NE:2], u[1:2 * NE:2] = GMP.glass_to_u(x[0:2 * NE:2], x[1:2 * NE:2])
    for s in CF.ASPH_VAR:
        i = CF.IA[s]; lo, hi = CF.C_RANGE[s]
        u[i] = _logit((x[i] - lo) / (hi - lo)); u[i + 1:i + 4] = x[i + 1:i + 4]
    u[CF.IT] = _logit((x[CF.IT] - CF.T_LO) / (CF.T_HI - CF.T_LO))
    return u


def u_to_phys_np(u):
    p = params_to_phys(torch.tensor(np.asarray(u)[None], dtype=DT, device=dev))
    x = np.zeros(NP)
    x[0:2 * NE:2] = p['nd'][0].cpu().numpy(); x[1:2 * NE:2] = p['vd'][0].cpu().numpy()
    for j, s in enumerate(CF.ASPH_VAR):
        i = CF.IA[s]; x[i] = p['c'][0, j].item(); x[i + 1:i + 4] = (p['A'][0, j] / T.ASC).cpu().numpy()
    x[CF.IT] = p['t16'].item()
    return x


def radii(x):
    """vertex radii of the variable aspheres {surface: R}"""
    return {s: (round(1 / x[CF.IA[s]], 3) if abs(x[CF.IA[s]]) > 1e-12 else 1e12) for s in CF.ASPH_VAR}


# ------------------------------------------------------------------------------------------------ starts
# convex polygon in (dispersion = (nd-1)/vd, nd) used for random start glasses: FCD100, PCD51, TAC8, TAFD65, FDS16W
POLY = np.array([[0.437 / 95.1, 1.437], [0.5935 / 67.0, 1.5935], [0.729 / 54.67, 1.729],
                 [1.0509 / 26.94, 2.0509], [0.9861 / 16.48, 1.9861]])


def poly_excess(nd, vd):
    p = np.array([(nd - 1) / vd, nd]); worst = 0.0; c = POLY.mean(0)
    for i in range(len(POLY)):
        a, b = POLY[i], POLY[(i + 1) % len(POLY)]
        e = b - a; nrm = np.array([e[1], -e[0]])
        if np.dot(c - a, nrm) > 0: nrm = -nrm
        nrm = nrm / np.linalg.norm(nrm * np.array([20.0, 1.0]))
        worst = max(worst, np.dot(p - a, nrm))
    return worst


def sag(c, A, h):
    r2 = h * h
    return c * r2 / (1 + np.sqrt(max(1 - c * c * r2, 1e-12))) + A[0] * r2 ** 2 + A[1] * r2 ** 3 + A[2] * r2 ** 4


def fit_asph_start(x):
    """least-squares A4..A8 so every variable asphere sits mid-band (sag bands) for its vertex curvature"""
    for s, B in CF.BANDS.items():
        i = CF.IA[s]; c = x[i]
        hs = np.array(list(B.keys())); zt = np.array([(lo + hi) / 2 for lo, hi in B.values()])
        base = np.array([c * h * h / (1 + np.sqrt(max(1 - c * c * h * h, 1e-12))) for h in hs])
        M = np.stack([hs ** 4 * CF.ASC[0], hs ** 6 * CF.ASC[1], hs ** 8 * CF.ASC[2]], 1)
        x[i + 1:i + 4] = np.clip(np.linalg.lstsq(M, zt - base, rcond=None)[0], -59, 59)
    return x


def parax_np(x):
    """paraxial EFL and focus z (with the cover glass) of design x at the reference wavelength (numpy)"""
    iw = CF.IREF
    from glass import nidx, get
    nfix = {e: nidx(get(*k), CF.WL)[iw] for e, k in CF.FIXED.items()}
    y, u, npv = 1.0, 0.0, 1.0
    for k, s in enumerate(CF.ORDER):
        if s in CF.ASPH_VAR: c = x[CF.IA[s]]
        else: c = 1 / CF.R0[s] if CF.R0[s] else 0.0
        if s in CF.AFTER:
            e = CF.AFTER[s]
            n = nfix[e] if e in nfix else float(np.atleast_1d(model_n(x[2 * CF.ELEMS.index(e)], x[2 * CF.ELEMS.index(e) + 1]))[iw])
        else:
            n = 1.0
        u = (npv * u - y * c * (n - npv)) / n
        if k + 1 < CF.NS: y = y + (CF.ZS[k + 1] - CF.ZS[k]) * u
        npv = n
    u = u if abs(u) > 1e-4 else -1e-4
    ncg = float(T.NCG[iw])
    return -1.0 / u, CF.Z[CF.LAST] - y / u + CF.CG_T * (1 - 1 / ncg)


def parax_project(x):
    """adjust the glass nd's and asphere vertex curvatures so EFL = window centre and the paraxial focus is on the image"""
    gi = list(range(0, 2 * NE, 2)); ai = [CF.IA[s] for s in CF.ASPH_VAR]; idx = gi + ai
    x0 = x.copy()
    zimg = CF.Z[CF.LAST] + x[CF.IT] + CF.CG_T + CF.CG_GAP

    def f(pv):
        y = x.copy(); y[idx] = pv
        efl, zpar = parax_np(y)
        return np.r_[(efl - CF.EFL_NOM) / 0.05, (zpar - zimg + 0.05) / 0.05, (pv[:NE] - x0[gi]) / 0.08, (pv[NE:] - x0[ai]) / 0.003]
    lb = np.r_[[1.45] * NE, [CF.C_RANGE[s][0] for s in CF.ASPH_VAR]]
    ub = np.r_[[2.05] * NE, [CF.C_RANGE[s][1] for s in CF.ASPH_VAR]]
    sol = least_squares(f, np.clip(x[idx], lb, ub), bounds=(lb, ub))
    x = x.copy(); x[idx] = sol.x
    return fit_asph_start(x)


def random_start(rng):
    x = np.zeros(NP)
    for i, e in enumerate(CF.ELEMS):
        while True:
            nd = rng.uniform(1.48, 2.0); vd = rng.uniform(17, 82)
            if poly_excess(nd, vd) == 0: break
        x[2 * i], x[2 * i + 1] = nd, min(vd, CF.VD_MAX - 2.0)
    for s in CF.ASPH_VAR:
        av = CF.AV.get(s, {}); lo, hi = CF.C_RANGE[s]
        x[CF.IA[s]] = 1 / rng.uniform(*av['random_R']) if av.get('random_R') else rng.uniform(lo + 0.3 * (hi - lo), lo + 0.7 * (hi - lo))
    x[CF.IT] = rng.uniform(*CF.T_RANDOM)
    return x


def random_starts(n, seed, efl_tol=1.2):
    """n random designs projected onto the EFL / focus conditions -> u tensor (n, NP)"""
    rng = np.random.default_rng(seed); X = []; tries = 0
    while len(X) < n and tries < 50 * n:
        tries += 1
        x = parax_project(fit_asph_start(random_start(rng)))
        efl, _ = parax_np(x)
        if abs(efl - CF.EFL_NOM) > efl_tol: continue
        X.append(phys_to_u(x))
    return torch.tensor(np.array(X), dtype=DT, device=dev)
