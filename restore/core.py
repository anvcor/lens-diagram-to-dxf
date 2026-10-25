"""Batched ray-trace core (PyTorch, float64, CUDA or CPU) for the lens in the config.

build(p) -> per-surface (curvature (B,), asphere A4..A10 (B,4), index after (B,NW)); trace(); parax().
Device: env LENS_DEVICE (cpu|cuda); default cuda when available, cpu when the Rust engine is used."""
import os, numpy as np, torch
import config as CF
from glass import WL, WW, SH_C, SH_F, VD_C, VD_F, GLASS_MODEL, GM_DEG, GM_COEF, gm_terms, nidx, get

_want = os.environ.get('LENS_DEVICE', '').lower()
dev = torch.device('cuda' if (torch.cuda.is_available() and _want != 'cpu') else 'cpu')
if dev.type == 'cpu':
    torch.set_num_threads(int(os.environ.get('LENS_THREADS', max(1, (os.cpu_count() or 2) - 1))))
print('[lens solver] device: %s  engine: %s' % (dev, CF.ENGINE), flush=True)
DT = torch.float64
NW = len(WL)
IREF = CF.IREF
ORDER, ZS, NS, ISTOP = CF.ORDER, CF.ZS, CF.NS, CF.ISTOP
ELEMS = CF.ELEMS
WWt = torch.tensor(WW, dtype=DT, device=dev)
SHC = torch.tensor(SH_C, dtype=DT, device=dev); SHF = torch.tensor(SH_F, dtype=DT, device=dev)
GMC = torch.tensor(GM_COEF if GM_COEF is not None else np.zeros((1, NW)), dtype=DT, device=dev)
ASC = torch.tensor(CF.ASC, dtype=DT, device=dev)
CIRS = torch.tensor([CF.CIR[s] for s in ORDER], dtype=DT, device=dev)
N_FIXED = {e: nidx(get(*k), WL) for e, k in CF.FIXED.items()}
NCG_NP = nidx(get(*CF.CG_GLASS), WL) if CF.CG_GLASS else np.ones(NW)       # cover-glass index (exact, used by parax)
NCG = torch.tensor(NCG_NP, dtype=DT, device=dev)
NEWTON = set(CF.NEWTON)

# ---- glass map region: non-convex polygon in (vd, nd) around the real optical glasses, counter-clockwise
GMAP = np.array([
    (95.1, 1.437), (81.6, 1.497), (70.4, 1.487), (64.2, 1.517), (59.5, 1.522), (45.8, 1.548), (40.9, 1.581),
    (36.3, 1.620), (33.8, 1.648), (29.5, 1.717), (25.4, 1.805), (23.8, 1.847), (20.9, 1.923), (16.5, 1.986),
    (19.3, 2.002), (26.9, 2.051), (29.1, 2.001), (32.3, 1.954), (37.3, 1.900), (40.8, 1.883), (46.6, 1.816),
    (49.6, 1.773), (54.7, 1.729), (63.4, 1.618), (68.6, 1.593), (81.6, 1.497 + 0.0001), (95.1, 1.437 + 0.0001)])
GM = torch.tensor(np.c_[GMAP[:, 0], GMAP[:, 1] / 0.01], dtype=DT, device=dev)   # normalized units (vd, nd/0.01)
if os.environ.get('RESTORE_LEGACY32') == '1':
    # regression only: the original Sigma solver created these constants in float32 before casting to float64
    # (fictitious-glass coefficients ~1e-9 in n, asphere scaling ~1e-8 relative); emulate it to compare bit for bit
    WWt, SHC, SHF, GMC, ASC, GM, CIRS, NCG = (x.float().double() for x in (WWt, SHC, SHF, GMC, ASC, GM, CIRS, NCG))


def gmap_dist(vd, nd):
    """distance outside the glass-map polygon in normalized units (1 = 1 Abbe = 0.01 nd); 0 inside"""
    P = torch.stack([vd, nd / 0.01], -1)
    A = GM; Bv = torch.roll(GM, -1, 0)
    E = Bv - A
    AP = P[..., None, :] - A
    t = ((AP * E).sum(-1) / (E * E).sum(-1)).clamp(0, 1)
    D = torch.sqrt(((AP - t[..., None] * E) ** 2).sum(-1) + 1e-9)
    dmin = D.min(-1).values
    with torch.no_grad():
        y = P[..., 1:2]; x = P[..., 0:1]
        y1, y2 = A[:, 1], Bv[:, 1]; x1, x2 = A[:, 0], Bv[:, 0]
        cond = ((y1 > y) != (y2 > y))
        xint = x1 + (y - y1) * (x2 - x1) / (y2 - y1 + 1e-12)
        inside = ((cond & (x < xint)).sum(-1) % 2) == 1
    return torch.where(inside, torch.zeros_like(dmin), dmin)


def model_index(nd, vd):
    """(B,1) nd, vd -> (B,NW) model-glass indices"""
    if GLASS_MODEL == 'codev':
        shape = (torch.stack(torch.broadcast_tensors(*gm_terms(nd, vd, GM_DEG)), -1) @ GMC)[:, 0]
    else:
        shape = SHC + (SHF - SHC) * (vd - VD_C) / (VD_F - VD_C)
    return nd + (nd - 1) / vd * shape


def build(p):
    """per-surface (c (B,), A (B,4), n after (B,NW)).  p: nd, vd (B,NE); c (B,NA); A (B,NA,3); t16 (B,)"""
    B = p['t16'].shape[0]
    n_el = {}
    for i, e in enumerate(ELEMS):
        n_el[e] = model_index(p['nd'][:, i:i + 1], p['vd'][:, i:i + 1])
    for e, n in N_FIXED.items():
        n_el[e] = torch.tensor(n, dtype=DT, device=dev).expand(B, NW)
    one = torch.ones(B, NW, dtype=DT, device=dev)
    z4 = torch.zeros(B, 4, dtype=DT, device=dev)
    surf = []
    for s in ORDER:
        if s in CF.ASPH_VAR:
            j = CF.ASPH_VAR.index(s)
            c = p['c'][:, j]; A = torch.cat([p['A'][:, j], z4[:, :1]], 1)
        elif s in CF.ASPH_FIXED:
            c = torch.full((B,), 1 / CF.R0[s] if CF.R0[s] else 0.0, dtype=DT, device=dev)
            A = torch.tensor(CF.ASPH_FIXED[s], dtype=DT, device=dev).expand(B, 4)
        else:
            c = torch.full((B,), 1 / CF.R0[s] if CF.R0[s] else 0.0, dtype=DT, device=dev); A = z4
        n = n_el[CF.AFTER[s]] if s in CF.AFTER else one
        surf.append((c, A, n))
    return surf


def sag_g(c, A, r2):
    r2 = torch.clamp(r2, max=900.0)
    q = torch.sqrt(torch.clamp(1 - c * c * r2, min=1e-8))
    sag = c * r2 / (1 + q) + A[..., 0] * r2 ** 2 + A[..., 1] * r2 ** 3 + A[..., 2] * r2 ** 4 + A[..., 3] * r2 ** 5
    g = c / q + 4 * A[..., 0] * r2 + 6 * A[..., 1] * r2 ** 2 + 8 * A[..., 2] * r2 ** 3 + 10 * A[..., 3] * r2 ** 4
    return sag, g


def parax(surf, t16, iw=IREF):
    """y-nu at wavelength iw -> efl, paraxial focus z (with the cover glass), stop-defined epd, entrance-pupil z"""
    B = t16.shape[0]
    y = torch.ones(B, dtype=DT, device=dev); u = torch.zeros(B, dtype=DT, device=dev); npv = torch.ones(B, dtype=DT, device=dev)
    ystop = None
    for k, (c, A, n) in enumerate(surf):
        nn = n[:, iw]
        u = (npv * u - y * c * (nn - npv)) / nn
        if k == ISTOP: ystop = y
        if k + 1 < NS: y = y + (ZS[k + 1] - ZS[k]) * u
        npv = nn
    ncg = float(NCG_NP[iw])
    u = torch.where(u.abs() < 1e-4, torch.full_like(u, -1e-4), u)
    l = -y / u
    zpar = CF.Z[CF.LAST] + l + CF.CG_T * (1 - 1 / ncg)                  # physical focus with the plate
    efl = -1.0 / u
    ystop = torch.where(ystop.abs() < 1e-3, torch.full_like(ystop, 1e-3), ystop)
    epd = 2 * CF.CIR[CF.STOP] / ystop
    z0 = ZS[0] - 20.0
    yb = torch.zeros(B, dtype=DT, device=dev); ub = torch.full((B,), 0.01, dtype=DT, device=dev); npv = torch.ones(B, dtype=DT, device=dev)
    yb = yb + 20.0 * ub
    for k in range(ISTOP):
        c, A, n = surf[k]; nn = n[:, iw]
        ub = (npv * ub - yb * c * (nn - npv)) / nn
        yb = yb + (ZS[k + 1] - ZS[k]) * ub
        npv = nn
    beta = yb / 0.01
    zep = z0 + beta / ystop
    return efl, zpar, epd, zep


def trace(surf, P, D, iw_all):
    """P,D: (B,R,3); returns P, D at the last traced surface and the ray heights at every surface (B,R,ns)"""
    npv = torch.ones(P.shape[:2], dtype=DT, device=dev)
    rs = []
    for k, (c, A, n) in enumerate(surf):
        zk = ZS[k]
        cc = c[:, None]; AA = A[:, None, :]
        p = P - torch.tensor([0, 0, zk], dtype=DT, device=dev)
        Bq = D[..., 2] - cc * (p * D).sum(-1); Cq = cc * (p * p).sum(-1) - 2 * p[..., 2]
        disc = torch.clamp(Bq * Bq - cc * Cq, min=1e-9)
        den = Bq + torch.sqrt(disc)
        den = torch.where(den.abs() < 1e-4, torch.full_like(den, 1e-4), den)
        t = Cq / den
        if k in NEWTON:
            with torch.no_grad():
                tt = t.detach()
                for _ in range(5):
                    q = p.detach() + tt[..., None] * D.detach(); r2 = q[..., 0] ** 2 + q[..., 1] ** 2
                    sg, g = sag_g(cc.detach(), AA.detach(), r2)
                    f = q[..., 2] - sg; fp = D.detach()[..., 2] - g * (q[..., 0] * D.detach()[..., 0] + q[..., 1] * D.detach()[..., 1])
                    fp = torch.where(fp.abs() < 1e-3, torch.full_like(fp, 1e-3), fp)
                    tt = torch.nan_to_num(tt - f / fp, nan=0.0, posinf=0.0, neginf=0.0).clamp(-200, 200)
            q = p + tt[..., None] * D; r2 = q[..., 0] ** 2 + q[..., 1] ** 2
            sg, g = sag_g(cc, AA, r2)
            f = q[..., 2] - sg; fp = D[..., 2] - g * (q[..., 0] * D[..., 0] + q[..., 1] * D[..., 1])
            fp = torch.where(fp.abs() < 1e-3, torch.full_like(fp, 1e-3), fp)
            t = tt - f / fp                                              # one differentiable Newton step
        q = p + t[..., None] * D; r2 = q[..., 0] ** 2 + q[..., 1] ** 2
        _, g = sag_g(cc, AA, r2)
        N = torch.stack([-g * q[..., 0], -g * q[..., 1], torch.ones_like(r2)], -1)
        N = N / N.norm(dim=-1, keepdim=True)
        n2 = torch.gather(n, 1, iw_all)
        mu = npv / n2
        ci = (D * N).sum(-1)
        arg = torch.clamp(1 - mu * mu * (1 - ci * ci), min=1e-6)
        D = mu[..., None] * D + (torch.sqrt(arg) - mu * ci)[..., None] * N
        P = (q + torch.tensor([0, 0, zk], dtype=DT, device=dev)).clamp(-1e3, 1e3)
        D = D / D.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        rs.append(torch.sqrt(r2.clamp(max=1e6) + 1e-12))
        npv = n2
    return P, D, torch.stack(rs, -1)
