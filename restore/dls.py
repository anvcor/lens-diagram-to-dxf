"""dls.py -- CODE V AUT-style damped least squares for the restoration solver (batched; torch CUDA/CPU or the Rust engine).

Algorithm (restore/README.md; follows the CODE V Optimization Reference Manual's description of AUT):
  * error function f(u)  = spot residuals + official geometric-MTF residuals (exactly objective.resid's first rows)
  * constraints   c(u)<=0 = EFL window, focus, asphere sag bands, glass map, vd <= vd_max, axial
    clearance, CRA at the sensor corner, RI at the corner.  Same thresholds/units as the objective's penalty rows,
    so that  objective loss == ||f||^2 + ||relu(c)||^2  (checked by tests/check_sigma50.py).  They are NOT penalties here: each major cycle
    solves  min ||f + J d||^2 + lam ||S d||^2   s.t.  c_W + A_W d = 0  (Lagrange multipliers, active set W).
  * active set: violated / near-active constraints start in W; a constraint the step would violate is added; one
    whose multiplier is negative (it would move into the feasible region) is dropped; a constraint that is
    dropped and then re-added is locked (no cycling).  Iterated to a stable set (capped).
  * automatic damping: for a log grid of lam the constrained step d(lam) is solved for every candidate in one
    batched call; the error function is PREDICTED for every lam from the linearised ray-intercept features
    (exact MTF/spot of linearised intercepts -- the analogue of CODE V's wavefront-differential MTF model;
    a purely linear residual model is monotone in lam and cannot choose a damping).  The lam with the lowest
    predicted error function that satisfies the linearised constraints is verified by real traces (ntry lam
    values in one batch); if none improves, larger lam / shorter steps are tried (backtracking).
  * derivatives: deriv='ad'  forward-mode autodiff (torch.func.vmap(jvp)) of the ray-intercept features with
    the vignetted pupil fixed (CODE V fixed vignetting factors) and an implicit-function correction for the
    chief-ray aim;  deriv='fd' forward differences with per-variable steps re-calibrated every cycle so that
    every variable produces the same RMS intercept change (balanced diagonal);  deriv='fdc' central.
  * termination like CODE V: MXC, MNC, IMP (relative improvement < imp for impn cycles), TAR, TIM;
    optional BFG (structured BFGS second-order term) and CNV (allowed error-function rise).
  * variable mask per candidate (fixed glasses / shape-only polish).  Works with core.build replaced by
    snap.build_mixed (per-design catalog glasses; row bookkeeping handled here, original state restored).

Wrappers:  lm(u, iters) ; lm_masked(u, varmask, iters)  (return u, loss, pup with loss = objective loss).
Env: LENS_CRA_MAX, LENS_RI_MIN, LENS_DEVICE, LENS_ENGINE, ... (README)."""
import math, os, sys, time, numpy as np, torch
from torch.func import jvp, vmap
import config as CF
import objective as lm4
import params as L2
import core as T
import glassmap as GMP
from config import Z, LAST

dev, DT = lm4.dev, lm4.DT
NP = CF.NP
NF, NW, NR = lm4.NF, lm4.NW, lm4.NR
NPI = NF * NW * NR * 2                      # ray-intercept features (x,y per ray)
NS = T.NS
AXI = torch.nonzero(lm4.FI == 0)[:, 0]      # rays of the axial field
NFEAT = NPI + 3 + NF + NS                   # intercepts, efl, dz, cra, ri per field, axial max height per surface
MF = NPI + 2 * len(CF.FREQS) * NF                           # error-function rows: spot (NPI) + MTF (40)
# sag bands of the variable aspheres: per asphere (index j, heights, lo, hi)
SAGB = [(j, torch.tensor(sorted(CF.BANDS[s]), dtype=DT, device=dev),
         torch.tensor([CF.BANDS[s][h][0] for h in sorted(CF.BANDS[s])], dtype=DT, device=dev),
         torch.tensor([CF.BANDS[s][h][1] for h in sorted(CF.BANDS[s])], dtype=DT, device=dev))
        for j, s in enumerate(CF.ASPH_VAR) if s in CF.BANDS]
NOSTOP = torch.tensor([k for k in range(NS) if k != T.ISTOP], device=dev)
# constraint layout (names for reports)
CNAMES = (['efl_hi', 'efl_lo', 'dz_hi', 'dz_lo']
          + [nm for s in CF.ASPH_VAR if s in CF.BANDS for nm in (['%slo@%g' % (s, h) for h in sorted(CF.BANDS[s])] + ['%shi@%g' % (s, h) for h in sorted(CF.BANDS[s])])]
          + GMP.NAMES + ['vd_%s' % e for e in T.ELEMS]
          + ['clear_%s' % T.ORDER[k] for k in range(NS) if k != T.ISTOP] + ['cra', 'ri'])
MC = len(CNAMES)
CFG = dict(cra_margin=0.2, ri_margin=0.005)  # constraint thresholds = where lm4's penalties start
# sag-band rows of one surface at neighbouring heights are nearly parallel.  CODE V rule for such constraints
# (Optimization RM p99: one active general thickness constraint per surface, the most violated, per cycle):
# at most one member of each group is active in a cycle; the others are handled in later cycles.
GROUPS = [torch.tensor([nm.startswith(pfx) for nm in CNAMES], device=dev) for s in CF.ASPH_VAR if s in CF.BANDS for pfx in (s + 'lo@', s + 'hi@')]


def _one_per_group(mask, score, groups=None):
    """keep at most one True per constraint group (the one with the largest score); mask, score (..., m)"""
    for gm in (GROUPS if groups is None else groups):
        cand = mask & gm
        s = torch.where(cand, score, torch.full_like(score, -float('inf')))
        keep = torch.zeros_like(mask).scatter_(-1, s.argmax(-1, keepdim=True), True) & cand
        mask = (mask & ~gm) | keep
    return mask

DEFAULTS = dict(
    mxc=60, mnc=2, imp=0.05, impn=3, tar=None, tim=None,
    deriv='ad', fd_delta=1e-7, fd_h0=1e-5, fd_hmax=1e-4, chunk=32,
    lam_lo=-8.0, lam_hi=2.0, nlam=21, ntry=3, nback=3, lam_drop=6, alphas=(1.0, 0.5, 0.25),
    rho_mul=1.5, rho_keep=0.5, rho_min=1.0, rho_max=1e8, mdz=0.0, soft=False, pred_chunk=1024, cand_margin=float('inf'),   # e.g. 3.0 restricts the bordered solve to near-active rows
    pupil_every=1, pupil='fast', vtol=0.05, as_iters=10, dmax=2.0,
    bfg=False, cnv=0.0, cnv_start=3, log_every=1, tag='', verbose=True)


def sync():
    if dev.type == 'cuda': torch.cuda.synchronize()


# ----------------------------------------------------------------------------------------------- row bookkeeping
class RowHook:
    """core.build may be snap.build_mixed (catalog glasses fixed per design), which reads a per-design state through
    ST['rows']; here every evaluated row is mapped to its original design index."""
    def __init__(self):
        self.kind = None
        b = T.build
        mod = sys.modules.get(getattr(b, '__module__', ''), None)
        if getattr(b, '__name__', '') == 'build_mixed' and mod is not None and hasattr(mod, 'ST'):
            self.kind, self.mod = 'snap', mod
            self.rows0 = mod.ST.get('rows')

    def set(self, rows):
        if self.kind == 'snap':
            self.mod.ST['rows'] = rows

    def restore(self):
        if self.kind == 'snap':
            self.mod.ST['rows'] = self.rows0


# ----------------------------------------------------------------------------------------------- fast vignetted pupil
def pupil_fast(u, nsec=16, rounds=6):
    """same rim-ray definition as lm4.pupil_setup (setvig-like), multi-section search instead of 22 bisections:
    each round tests nsec-1 interior points at once (precision 16^-6 = 6e-8 of the EP radius; bisection 2^-22).
    For a monotone pass/fail predicate the result equals bisection; 12 small traces instead of 44."""
    with torch.no_grad():
        p = L2.params_to_phys(u); B = u.shape[0]
        surf = T.build(p)
        efl, zpar, epd, zep = T.parax(surf, p['t16'])
        ang = lm4.field_angles(surf, p['t16'], efl, zep)                 # (B,NF): YRI-like or fixed (LENS_FIELDS)
        R = 0.5 * efl / lm4.FNO_SPEC
        tth = torch.tan(ang)
        ycp = (lm4.Z0 - zep)[:, None] * tth
        cir_ns = T.CIRS.clone(); cir_ns[T.ISTOP] = 1e3
        D = torch.stack([torch.zeros_like(ang), torch.sin(ang), torch.cos(ang)], -1)   # (B,NF,3)

        def passes(x, y):                                          # (B,NF,K)
            K = x.shape[-1]
            P = torch.stack([x, y, torch.full_like(x, lm4.Z0)], -1).reshape(B, NF * K, 3)
            Dd = D[:, :, None, :].expand(B, NF, K, 3).reshape(B, NF * K, 3).contiguous()
            iw = torch.full((B, NF * K), lm4.IREF, device=dev, dtype=torch.long)
            _, _, rs = T.trace(surf, P, Dd, iw)
            rs = torch.nan_to_num(rs, nan=1e3, posinf=1e3, neginf=1e3)
            ok = (rs <= cir_ns).all(-1).view(B, NF, K)
            yep = y + (zep[:, None, None] - lm4.Z0) * tth[:, :, None]
            return ok & ((x ** 2 + yep ** 2) <= (R[:, None, None] ** 2) * (1 + 1e-9))
        fr = torch.linspace(0, 1, nsec + 1, dtype=DT, device=dev)[1:-1]          # interior fractions (nsec-1)

        def search(lo, hi, test):
            """lo passes / hi fails (assumed, as in bisection); returns the last pass below the first failure"""
            for _ in range(rounds):
                t = lo[..., None] + (hi - lo)[..., None] * fr                     # (...,nsec-1)
                ok = test(t)
                fail = ~ok
                idx = torch.arange(fr.numel(), device=dev).expand_as(fail)
                k = torch.where(fail, idx, torch.full_like(idx, fr.numel())).amin(-1)          # first failure (nsec-1 = none)
                tpad = torch.cat([lo[..., None], t, hi[..., None]], -1)
                nlo = tpad.gather(-1, k[..., None])[..., 0]                         # point before first failure
                nhi = tpad.gather(-1, (k + 1)[..., None])[..., 0]
                lo, hi = nlo, nhi
            return lo
        sgn = torch.tensor([1.0, -1.0], dtype=DT, device=dev)

        def test_m(t):                                             # t (B,NF,2,K)
            Kk = t.shape[-1]
            y = ycp[..., None, None] + sgn[None, None, :, None] * t * R[:, None, None, None]
            ok = passes(torch.zeros_like(y).reshape(B, NF, 2 * Kk), y.reshape(B, NF, 2 * Kk))
            return ok.view(B, NF, 2, Kk)
        lo = search(torch.zeros(B, NF, 2, dtype=DT, device=dev), torch.ones(B, NF, 2, dtype=DT, device=dev), test_m)
        yup = ycp + lo[..., 0] * R[:, None]; ylo = ycp - lo[..., 1] * R[:, None]
        yc = 0.5 * (yup + ylo); hy = 0.5 * (yup - ylo)

        def test_s(t):                                             # t (B,NF,K)
            return passes(t * R[:, None, None], yc[..., None].expand_as(t).clone())
        lo = search(torch.zeros(B, NF, dtype=DT, device=dev), torch.ones(B, NF, dtype=DT, device=dev), test_s)
        hx = lo * R[:, None]
        hy = hy.clamp(min=1e-3) * 0.999; hx = hx.clamp(min=1e-3) * 0.999
    return yc, hy, hx, ang


# ray engine: 'torch' (this file, GPU or CPU) or 'rust' (rcore.py -> rust/lenscore, CPU, all cores, same
# results to ~1e-13).  env LENS_ENGINE.
ENGINE = os.environ.get('LENS_ENGINE', 'torch')
if ENGINE == 'rust':
    import rcore


def pupil(u, kind='fast'):
    if ENGINE == 'rust':
        return rcore.pupil(u)
    return pupil_fast(u) if kind == 'fast' else lm4.pupil_setup(u)


# ----------------------------------------------------------------------------------------------- features
def cra_corner(u):
    """lm4.chief_cra with the aim made differentiable: secant aim (detached) + one implicit-function Newton step
    y = y* - s(y*, u)/s'(y*)  -> value unchanged, dy/du = -(ds/du)/s' (exact first-order aim correction)."""
    p = L2.params_to_phys(u); B = u.shape[0]
    surf = T.build(p)
    efl, zpar, epd, zep = T.parax(surf, p['t16'])
    K = len(lm4.CRA_ANG)
    th = lm4.CRA_ANG[None].expand(B, K)
    D = torch.stack([torch.zeros_like(th), torch.sin(th), torch.cos(th)], -1)
    iw = torch.full((B, K), lm4.IREF, dtype=torch.long, device=dev)
    sd = [(c.detach(), A.detach(), n.detach()) for c, A, n in surf[:T.ISTOP + 1]]

    def ystop(y0, sf, dd=D, ii=iw):
        P = torch.stack([torch.zeros_like(y0), y0, torch.full_like(y0, lm4.Z0)], -1)
        Pl, _, _ = T.trace(sf, P, dd, ii)
        return Pl[..., 1]
    with torch.no_grad():
        ya = (lm4.Z0 - zep.detach())[:, None] * torch.tan(th); yb = ya + 0.5
        sa, sb = ystop(ya, sd), ystop(yb, sd)
        for _ in range(4):
            yn = yb - sb * (yb - ya) / torch.where((sb - sa).abs() < 1e-9, torch.full_like(sb, 1e-9), sb - sa)
            ya, sa = yb, sb; yb = yn; sb = ystop(yb, sd)
        e = 1e-4
        sp = ystop(torch.cat([yb + e, yb - e], 1), sd, torch.cat([D, D], 1), torch.cat([iw, iw], 1))
        slope = (sp[:, :K] - sp[:, K:]) / (2 * e)
        slope = torch.where(slope.abs() < 1e-6, torch.full_like(slope, 1e-6), slope)
    yb = yb.detach(); slope = slope.detach()
    y0 = yb - ystop(yb, surf[:T.ISTOP + 1]) / slope
    P = torch.stack([torch.zeros_like(y0), y0, torch.full_like(y0, lm4.Z0)], -1)
    Pl, Dl, _ = T.trace(surf, P, D, iw)
    Dz = Dl[..., 2].clamp(min=0.05)
    cra = torch.rad2deg(torch.atan2(torch.sqrt(Dl[..., 0] ** 2 + Dl[..., 1] ** 2), Dz))
    zl = Z[LAST] + p['t16'][:, None]
    yl = Pl[..., 1] + (zl - Pl[..., 2]) / Dz * Dl[..., 1]
    ncg = float(T.NCG[lm4.IREF])
    spp = Dl[..., 1] / ncg; cp = torch.sqrt(torch.clamp(1 - spp ** 2, min=1e-6))
    yimg = yl + spp / cp * lm4.CG_T + Dl[..., 1] / Dz * lm4.CG_GAP
    return lm4.interp_h(yimg, cra, lm4.H_CORNER)


def phi_fn(u, pup):
    """features (B, NFEAT) at fixed vignetted pupil: intercepts, EFL, focus error, corner CRA, RI per field,
    max axial-beam height per surface"""
    if ENGINE == 'rust':
        return rcore.phi(u, pup)
    B = u.shape[0]
    p, efl, zpar, epd, Pi, rs = lm4.image_points(u, pup)
    Dl = lm4.image_points.last_dir
    ri = lm4.rel_illum(Dl, B)
    zimg = Z[LAST] + p['t16'] + lm4.CG_T + lm4.CG_GAP
    rsax = rs.index_select(1, AXI).amax(1)
    cra = cra_corner(u)
    return torch.cat([Pi.reshape(B, -1), efl[:, None], (zpar - zimg)[:, None], cra[:, None], ri, rsax], 1)


def gmap_signed(vd, nd):
    """core.gmap_dist with sign: negative inside the glass-map polygon (smooth constraint), relu() == gmap_dist"""
    P = torch.stack([vd, nd / 0.01], -1)
    A = T.GM; Bv = torch.roll(T.GM, -1, 0); E = Bv - A
    AP = P[..., None, :] - A
    t = ((AP * E).sum(-1) / (E * E).sum(-1)).clamp(0, 1)
    dmin = torch.sqrt(((AP - t[..., None] * E) ** 2).sum(-1) + 1e-9).min(-1).values
    with torch.no_grad():
        y = P[..., 1:2].detach(); x = P[..., 0:1].detach()
        y1, y2 = A[:, 1], Bv[:, 1]; x1, x2 = A[:, 0], Bv[:, 0]
        cond = ((y1 > y) != (y2 > y))
        xint = x1 + (y - y1) * (x2 - x1) / (y2 - y1 + 1e-12)
        inside = ((cond & (x < xint)).sum(-1) % 2) == 1
    return torch.where(inside, -dmin, dmin)


def _sag(c, A, h):
    h2 = h * h
    return c[:, None] * h2 / (1 + torch.sqrt(torch.clamp(1 - c[:, None] ** 2 * h2, min=1e-12))) + A[:, 0:1] * h2 ** 2 + A[:, 1:2] * h2 ** 3 + A[:, 2:3] * h2 ** 4


def G_fn(phi, u):
    """[f (MF) | c (MC)] from features + variables.  f, c in lm4 units."""
    B = phi.shape[0]
    Pf = phi[:, :NPI].view(B, NF, NW * NR, 2)
    w = lm4.WR[None].expand(B, -1); Wf = w.view(B, NF, NW * NR)
    sw = Wf.sum(-1)
    cen = (Wf[..., None] * Pf).sum(2) / sw[..., None]
    rr = (Pf - cen[:, :, None, :]) * torch.sqrt(Wf / sw[..., None])[..., None] / 0.010 * lm4.W_SPOT / math.sqrt(NR / 37)
    hreal = cen[..., 1].abs()
    ms_ = [m for f in CF.FREQS for m in lm4.geo_mtf(Pf, Wf, f)]
    tgt = {k: lm4.interp(hreal, lm4.OH, lm4.OM[k]) for k in lm4.MKEYS}
    mres = torch.cat([m - tgt[k] for m, k in zip(ms_, lm4.MKEYS)], 1) * lm4.W_MTF
    efl, dz, cra = phi[:, NPI], phi[:, NPI + 1], phi[:, NPI + 2]
    ri = phi[:, NPI + 3:NPI + 3 + NF]; rsax = phi[:, NPI + 3 + NF:]
    p = L2.params_to_phys(u)
    sg = [(_sag(p['c'][:, j], p['A'][:, j], H), LO, HI) for j, H, LO, HI in SAGB]
    ri_c = lm4.interp_h(hreal, ri, lm4.H_CORNER)
    c = torch.cat([((efl - lm4.EFL_HI) / 0.05)[:, None], ((lm4.EFL_LO - efl) / 0.05)[:, None], ((dz - CF.DZ_MAX) / 0.02)[:, None], ((-CF.DZ_MAX - dz) / 0.02)[:, None],
                   *[r for s_, LO, HI in sg for r in ((LO - s_) / 0.003, (s_ - HI) / 0.003)],
                   GMP.constraint_rows(p['nd'], p['vd']),          # glass map rows (glassmap.py, LENS_GMAP)
                   (p['vd'] - lm4.VD_MAX) / 0.5,
                   (rsax.index_select(1, NOSTOP) - T.CIRS[NOSTOP] + 0.05) / 0.02,
                   ((cra - (lm4.CRA_MAX - CFG['cra_margin'])) / 0.1)[:, None], ((lm4.RI_MIN + CFG['ri_margin'] - ri_c) / 0.005)[:, None]], 1)
    G = torch.cat([rr.reshape(B, -1), mres, c], 1)
    if EXTRA is not None:                       # extra error-function rows (e.g. catalog attraction), after c
        G = torch.cat([G, EXTRA(u)], 1)
    return G


# optional extra error-function rows: EXTRA(u) -> (B, k), differentiable in u (snap.py sets its catalog attraction)
EXTRA = None


def split(G):
    """G = [f | c | extra]  ->  error-function rows [f | extra], constraints c"""
    f = G[..., :MF] if G.shape[-1] == MF + MC else torch.cat([G[..., :MF], G[..., MF + MC:]], -1)
    return f, G[..., MF:MF + MC]


def evaluate(u, pup, hook=None, rows=None, chunk=96):
    """real evaluation (no derivatives): returns f (B,MF[+extra]), c (B,MC)"""
    out = []
    with torch.no_grad():
        for k in range(0, u.shape[0], chunk):
            if hook is not None: hook.set(rows[k:k + chunk])
            out.append(G_fn(phi_fn(u[k:k + chunk], tuple(x[k:k + chunk] for x in pup)), u[k:k + chunk]))
    return split(torch.cat(out))


def lm4_loss(f, c):
    return (f ** 2).sum(1) + (torch.relu(c) ** 2).sum(1)


# ----------------------------------------------------------------------------------------------- derivatives
def _tangents(cols, B):
    n = len(cols)
    E = torch.zeros(n, B, NP, dtype=DT, device=dev)
    E[torch.arange(n, device=dev), :, cols] = 1.0
    return E


def jac_ad(u, pup, cols, hook, rows, chunk=32):
    """forward-mode autodiff: phi0 (B,NFEAT), J (B,n,NFEAT); pupil fixed"""
    P0, JJ = [], []
    for k in range(0, u.shape[0], chunk):
        uk = u[k:k + chunk]; pk = tuple(x[k:k + chunk] for x in pup)
        hook.set(rows[k:k + chunk])
        Et = _tangents(cols, uk.shape[0])
        p0, Jt = vmap(lambda t: jvp(lambda uu: phi_fn(uu, pk), (uk,), (t,)))(Et)
        P0.append(p0[0].detach()); JJ.append(Jt.permute(1, 0, 2).detach())
        del p0, Jt
    return torch.cat(P0), torch.cat(JJ)


def jac_fd(u, pup, cols, h, hook, rows, central=False, chunk=32):
    """finite differences; h (B,NP) per-candidate per-variable steps (u space).  Rows are design-major as in lm4."""
    B = u.shape[0]; n = len(cols)
    colt = torch.tensor(cols, device=dev)
    hc = h[:, colt]                                                       # (B,n)
    P0, JJ = [], []
    with torch.no_grad():
        for k in range(0, B, chunk):
            uk = u[k:k + chunk]; bk = uk.shape[0]; pk = tuple(x[k:k + chunk] for x in pup); rk = rows[k:k + chunk]
            hook.set(rk); p0 = phi_fn(uk, pk)
            E = torch.zeros(bk, n, NP, dtype=DT, device=dev)
            E[:, torch.arange(n, device=dev), colt] = hc[k:k + chunk]
            sg = [1.0, -1.0] if central else [1.0]
            ph = []
            for s in sg:
                up = (uk[:, None, :] + s * E).reshape(bk * n, NP)
                pr = tuple(x.repeat_interleave(n, 0) for x in pk)
                hook.set(rk.repeat_interleave(n))
                ph.append(phi_fn(up, pr).view(bk, n, -1))
            if central:
                J = (ph[0] - ph[1]) / (2 * hc[k:k + chunk, :, None])
            else:
                J = (ph[0] - p0[:, None]) / hc[k:k + chunk, :, None]
            P0.append(p0); JJ.append(J)
    return torch.cat(P0), torch.cat(JJ)


def g_jac(phi0, u, Jphi, cols):
    """chain rule through the cheap feature->error/constraint map: G0 (B,MF+MC), JG (B,n,MF+MC)"""
    Et = _tangents(cols, u.shape[0])
    G0, JG = vmap(lambda tp, tu: jvp(G_fn, (phi0, u), (tp, tu)))(Jphi.permute(1, 0, 2).contiguous(), Et)
    return G0[0].detach(), JG.permute(1, 0, 2).detach()


# ----------------------------------------------------------------------------------------------- constrained DLS step
def dls_grid(H, g, A, c, fm, act0, lams, as_iters=10, ctol=1e-4, groups=None):
    """batched constrained damped least squares over a lam grid.
    H (B,n,n) = J^T J (+S), g (B,n) = J^T f, A (B,m,n), c (B,m) (c<=0 wanted), fm (B,n) free-variable mask,
    act0 (B,m) initial active set, lams (L,).  Returns d (B,L,n) in u units, mu (B,L,m), act (B,L,m), glin (B,L,m).
    Scaled variables v = S d with S = sqrt(diag H) (Marquardt/CODE V-like scaling) -> damping lam*I in v."""
    B, n = g.shape; m = c.shape[1]; L = lams.numel()
    fmd = fm.to(DT)
    dg = torch.diagonal(H, dim1=1, dim2=2).clamp(min=0)
    Sc = torch.sqrt(dg)
    Sc = torch.maximum(Sc, 1e-7 * Sc.amax(1, keepdim=True).clamp(min=1e-30))
    Sc = torch.where(fm, Sc, torch.ones_like(Sc))
    Ht = H / (Sc[:, :, None] * Sc[:, None, :]) * (fmd[:, :, None] * fmd[:, None, :]) + torch.diag_embed(1 - fmd)
    gt = g / Sc * fmd
    At = A / Sc[:, None, :] * fmd[:, None, :]
    an = At.norm(dim=-1)                                                   # (B,m)
    canact = an > 1e-10
    an1 = torch.where(canact, an, torch.ones_like(an))
    Ah = At / an1[..., None]; ch = c / an1
    # least-squares multiplier estimate at the current point, independent of the damping (used for the merit weight):
    # min || gt + Ah_W^T mu ||  over the initially active set  ->  (Ah_W Ah_W^T + r I) mu = -Ah_W gt
    # (one member per sag-band group; a small ridge only as a fallback)
    groups = GROUPS if groups is None else groups
    act0 = _one_per_group(act0 & canact, c, groups)
    aw = act0.to(DT)
    Gw = (Ah @ Ah.transpose(1, 2)) * (aw[:, :, None] * aw[:, None, :]) + torch.diag_embed(torch.where(aw > 0, torch.full_like(aw, 1e-4), torch.ones_like(aw)))
    mu_ls = torch.linalg.solve(Gw, -(Ah @ gt[..., None])[..., 0] * aw) * aw / an1
    # per-lam factorisation of M = Ht + lam I (PD) and Schur complement  Sm = Ah M^-1 Ah^T
    I = torch.eye(n, dtype=DT, device=dev)
    M = Ht[:, None] + lams[None, :, None, None] * I                         # (B,L,n,n)
    Lc, info = torch.linalg.cholesky_ex(M)
    bad = info != 0
    if bad.any():                                                          # indefinite (BFG): shift
        M = torch.where(bad[..., None, None], M + (1e-6 + Ht.abs().amax((1, 2))[:, None, None, None]) * I, M)
        Lc, _ = torch.linalg.cholesky_ex(M)
    rhs = torch.cat([gt[:, None, :, None].expand(B, L, n, 1), Ah.transpose(1, 2)[:, None].expand(B, L, n, m)], -1)
    X = torch.cholesky_solve(rhs, Lc)                                      # (B,L,n,1+m)
    xg, XA = X[..., 0], X[..., 1:]
    Sm = Ah[:, None] @ XA                                                  # (B,L,m,m)
    Axg = (Ah[:, None] @ xg[..., None])[..., 0]                            # (B,L,m)
    act = (act0 & canact)[:, None].expand(B, L, m).clone()
    locked = torch.zeros_like(act); dropped = torch.zeros_like(act)
    eps = 1e-9 * torch.diagonal(Sm, dim1=-2, dim2=-1).amax(-1, keepdim=True).clamp(min=1e-30)   # relative regularisation
    for it in range(as_iters):
        a = act.to(DT)
        Smm = Sm * (a[..., :, None] * a[..., None, :]) + torch.diag_embed(torch.where(act, eps.expand_as(a), torch.ones_like(a)))
        r = torch.where(act, ch[:, None] - Axg, torch.zeros_like(a))
        mu = torch.linalg.solve(Smm, r[..., None])[..., 0] * a
        vt = -xg - (XA @ mu[..., None])[..., 0]
        glin = (ch[:, None] + (Ah[:, None] @ vt[..., None])[..., 0]) * an1[:, None]      # c units
        add = ~act & canact[:, None] & (glin > ctol)
        for gm in groups:                                                  # a group with an active member takes no other
            add &= ~(gm & (act & gm).any(-1, keepdim=True))
        add = _one_per_group(add, glin, groups)
        drop = act & ~locked & (mu < -1e-12)
        if not (add.any() or drop.any()): break
        locked |= add & dropped
        dropped |= drop
        act = (act | add) & ~drop
    d = vt / Sc[:, None, :] * fmd[:, None, :]
    return d, mu / an1[:, None, :], act, glin, mu_ls                       # multipliers in c units (w.r.t. Phi/2)


# ----------------------------------------------------------------------------------------------- driver
def optimize(u, varmask=None, **kw):
    """CODE V-style DLS on a batch of candidates.  u (B,25) raw variables; varmask (B,25) bool free variables.
    Returns dict(u, loss (lm4-equivalent), phi (error function ||f||^2), viol (max constraint violation, lm4 units),
    pup, hist, cycles (per candidate), time)."""
    o = dict(DEFAULTS); o.update(kw)
    t0 = time.time()
    u = u.detach().clone().to(dev, DT)
    B0 = u.shape[0]
    vm = torch.ones(B0, NP, dtype=torch.bool, device=dev) if varmask is None else varmask.to(dev).bool().clone()
    hook = RowHook()
    try:
        return _optimize(u, vm, hook, o, t0)
    finally:
        hook.restore()


def _optimize(u, vm, hook, o, t0):
    B0 = u.shape[0]
    ids_all = torch.arange(B0, device=dev)
    lams = 10.0 ** torch.linspace(o['lam_lo'], o['lam_hi'], o['nlam'], dtype=DT, device=dev)
    L = lams.numel()
    hfd = torch.full((B0, NP), o['fd_h0'], dtype=DT, device=dev)
    active = torch.ones(B0, dtype=torch.bool, device=dev)
    nsmall = torch.zeros(B0, dtype=torch.long, device=dev)
    ncyc = torch.zeros(B0, dtype=torch.long, device=dev)
    kmin = torch.zeros(B0, dtype=torch.long, device=dev)                    # lam floor index (trust safeguard)
    act_prev = torch.zeros(B0, MC, dtype=torch.bool, device=dev)
    phi_cur = torch.full((B0,), float('inf'), dtype=DT, device=dev); v_cur = phi_cur.clone()
    best_u = u.clone(); best_phi = phi_cur.clone(); best_v = phi_cur.clone()
    lam_used = torch.full((B0,), float('nan'), dtype=DT, device=dev)
    rho = torch.full((B0,), o['rho_min'], dtype=DT, device=dev)            # l1 merit weight per candidate
    ctl_all = torch.ones(B0, MC, dtype=torch.bool, device=dev)              # controllable constraints
    rho_floor = torch.full((B0,), o['rho_min'], dtype=DT, device=dev)      # raised while a candidate is stuck infeasible
    pup_full = [torch.zeros(B0, NF, dtype=DT, device=dev) for _ in range(4)]          # yc, hy, hx, field angles
    Sbfg = torch.zeros(B0, NP, NP, dtype=DT, device=dev) if o['bfg'] else None
    bfg_prev = {}
    hist = []
    nevals = dict(jac=0, trial=0, pupil=0)
    for cyc in range(o['mxc']):
        ids = ids_all[active]
        if ids.numel() == 0: break
        if o['tim'] is not None and time.time() - t0 > o['tim']: break
        ua = u[ids]; B = ids.numel()
        tc = time.time()
        if cyc % o['pupil_every'] == 0:
            hook.set(ids); pp = pupil(ua, o['pupil']); nevals['pupil'] += B
            for k in range(4): pup_full[k][ids] = pp[k]
        pa = tuple(x[ids] for x in pup_full)
        vma = vm[ids]
        cols = torch.nonzero(vma.any(0))[:, 0].tolist(); n = len(cols)
        colt = torch.tensor(cols, device=dev)
        fm = vma[:, colt]
        # ---- derivatives of the features, then chain rule to f and c
        if ENGINE == 'rust':                                          # exact forward-mode derivatives in Rust
            hook.set(ids); phi0, Jphi = rcore.jac(ua, pa, cols)
        elif o['deriv'] == 'ad':
            phi0, Jphi = jac_ad(ua, pa, cols, hook, ids, o['chunk'])
        else:
            phi0, Jphi = jac_fd(ua, pa, cols, hfd[ids], hook, ids, central=(o['deriv'] == 'fdc'), chunk=o['chunk'])
            cn = Jphi[:, :, :NPI].norm(dim=-1) / math.sqrt(NPI)                  # RMS intercept change per unit u
            hnew = (o['fd_delta'] / cn.clamp(min=1e-30)).clamp(1e-9, o['fd_hmax'])
            hh = hfd[ids]; hh[:, colt] = torch.where(cn > 1e-20, hnew, hh[:, colt]); hfd[ids] = hh
        nevals['jac'] += B * n
        Jphi = Jphi * fm[..., None].to(DT)
        hook.set(ids)
        G0, JG = g_jac(phi0, ua, Jphi, cols)
        f0, c0 = split(G0)
        Jf, A = split(JG); A = A.transpose(1, 2)                             # (B,n,MF), (B,MC,n)
        if o['soft']:            # CODE V WTC/PTC-like: constraints become error-function rows relu(c) (= lm4's objective)
            on = (c0 > 0).to(DT)
            f0 = torch.cat([f0, torch.relu(c0)], 1)
            Jf = torch.cat([Jf, A.transpose(1, 2) * on[:, None, :]], 2)
            A = torch.zeros_like(A); c0 = torch.full_like(c0, -1.0)
        # constraints with no derivative w.r.t. any free variable (e.g. glass map of a fixed catalog glass) cannot be
        # controlled; like CODE V's frozen-thickness violations they are reported but not counted (RM p167)
        ctl = A.abs().amax(-1) > 0                                           # (B,MC)
        ctl_all[ids] = ctl
        Phi0 = (f0 ** 2).sum(1); V0 = torch.relu(torch.where(ctl, c0, -1.0)).amax(1)
        phi_cur[ids] = Phi0; v_cur[ids] = V0
        bv, bp = best_v[ids], best_phi[ids]                                   # the current design may be the best one
        b0 = ((V0 <= o['vtol']) & ((bv > o['vtol']) | (Phi0 < bp))) | ((bv > o['vtol']) & (V0 < bv))
        bi = ids[b0]; best_u[bi] = ua[b0]; best_phi[bi] = Phi0[b0]; best_v[bi] = V0[b0]
        H = Jf @ Jf.transpose(1, 2); g = (Jf @ f0[..., None])[..., 0]
        if Sbfg is not None:
            Hb = _bfg_update(Sbfg, bfg_prev, ids, colt, Jf, f0, g, ua)
            H = H + Hb
        # ---- constrained DLS over the lam grid
        act0 = (c0 > -1e-3) | act_prev[ids]
        # candidate constraints (CODE V handles only constraints that can become active): near the bound or active
        # last cycle; after the solve every constraint is checked on the linearised step and newly violated ones are
        # added for one more solve.  Results are unchanged, the bordered systems shrink from ~100 rows to a few tens.
        cand = ((c0 > -o['cand_margin']) | act_prev[ids]) & ctl
        for _pass in range(3):
            cidx = torch.nonzero(cand.any(0))[:, 0]
            if cidx.numel() == 0:
                cidx = torch.zeros(1, dtype=torch.long, device=dev)
            d, mu_s, act_s, glin_s, mu_ls_s = dls_grid(H, g, A[:, cidx], c0[:, cidx], fm, act0[:, cidx], lams, o['as_iters'],
                                                       groups=[gm[cidx] for gm in GROUPS])
            gl_all = c0[:, None, :] + (A[:, None] @ d[..., None].clamp(-o['dmax'], o['dmax']))[..., 0]   # (B,L,MC)
            new = ((gl_all > 1e-4) & ctl[:, None, :]).any(1) & ~cand
            new[:, cidx] = False
            if not bool(new.any()): break
            cand |= new
        mu = torch.zeros(B, L, MC, dtype=DT, device=dev); mu[..., cidx] = mu_s
        actf = torch.zeros(B, L, MC, dtype=torch.bool, device=dev); actf[..., cidx] = act_s
        mu_ls = torch.zeros(B, MC, dtype=DT, device=dev); mu_ls[:, cidx] = mu_ls_s
        d = d.clamp(-o['dmax'], o['dmax'])
        dfull = torch.zeros(B, L, NP, dtype=DT, device=dev); dfull[..., colt] = d
        # ---- predicted merit for every (lam, step length): exact f, c of the linearised features
        # l1 exact-penalty merit  Psi = Phi/2 + rho * sum relu(c), rho = 1.5 x the largest least-squares multiplier
        # at the current point (not the damped-step multipliers: heavy damping inflates those and then any step that
        # restores feasibility wins regardless of the error function).  A step that trades error function for
        # feasibility at the price the multipliers say is accepted (CODE V's exact constraints).
        AL = torch.tensor(o['alphas'], dtype=DT, device=dev); NA = AL.numel()
        ar = torch.arange(B, device=dev)
        with torch.no_grad():
            dz = d[:, :, None, :] * AL[None, None, :, None]                      # (B,L,NA,n)
            dfz = torch.zeros(B, L, NA, NP, dtype=DT, device=dev); dfz[..., colt] = dz
            php = phi0[:, None, None] + torch.einsum('blan,bnf->blaf', dz, Jphi)
            pf, uf, rows_p = php.reshape(B * L * NA, -1), (ua[:, None, None] + dfz).reshape(-1, NP), ids.repeat_interleave(L * NA)
            CH = o['pred_chunk']                                                 # rows per call: bounds memory (dense pupils)
            Gl = []
            for k0 in range(0, pf.shape[0], CH):
                hook.set(rows_p[k0:k0 + CH]); Gl.append(G_fn(pf[k0:k0 + CH], uf[k0:k0 + CH]))
            Gp = torch.cat(Gl).view(B, L, NA, -1); del Gl, pf
            fp, cp = split(Gp)
            Phip = (fp ** 2).sum(-1)
            if o['soft']: Phip = Phip + (torch.relu(cp) ** 2).sum(-1)
            Sp = torch.relu(torch.where(ctl[:, None, None], cp, -1.0) - o['mdz']).sum(-1)
            S0 = torch.relu(torch.where(ctl, c0, -1.0) - o['mdz']).sum(1)
            mumax = torch.nan_to_num(mu_ls, nan=0.0, posinf=0.0).clamp(min=0).amax(-1)      # (B,) c units
            rho_c = torch.maximum(torch.maximum(o['rho_mul'] * mumax, o['rho_keep'] * rho[ids]), rho_floor[ids]).clamp(o['rho_min'], o['rho_max'])
            rho_k = rho_c[:, None].expand(B, L)                                  # same weight for every lam
            Psi0 = 0.5 * Phi0[:, None] + rho_k * S0[:, None]                    # (B,L)
            Psip = 0.5 * Phip + rho_k[..., None] * Sp                           # (B,L,NA)
            redp = (Psi0[..., None] - Psip) / Psi0[..., None].clamp(min=1e-30)  # predicted relative merit reduction
            okk = torch.isfinite(redp) & (torch.arange(L, device=dev)[None, :, None] >= kmin[ids][:, None, None])
            score = torch.where(okk, -redp, torch.full_like(redp, float('inf'))).view(B, L * NA)
            order = score.argsort(1)
            k1 = order[:, 0] // NA; a1 = order[:, 0] % NA                       # best predicted (lam, alpha)
        # ---- verification by real traces + backtracking along the best direction / heavier damping
        acc = torch.zeros(B, dtype=torch.bool, device=dev)
        u_new = ua.clone(); phi_new = Phi0.clone(); v_new = V0.clone(); k_acc = k1.clone()
        rho_acc = rho_k[ar, k1].clone()
        rel = torch.zeros(B, dtype=DT, device=dev)
        cnv_on = o['cnv'] > 0 and cyc >= o['cnv_start']
        nb = 0
        for bt in range(1 + o['nback']):
            todo = torch.nonzero(~acc)[:, 0]
            if todo.numel() == 0: break
            if bt == 0:
                nt = min(o['ntry'], L * NA)
                fl = order[todo, :nt]
                ks = fl // NA; As = AL[fl % NA]
            else:
                nt = 2
                ks = torch.stack([k1[todo], (k1[todo] + 4 * bt).clamp(max=L - 1)], 1)
                As = torch.stack([AL[a1[todo]] * 0.25 ** bt, torch.ones_like(AL[a1[todo]])], 1)
            dt = dfull[todo[:, None], ks] * As[..., None]                       # (todo,nt,NP)
            ut = (ua[todo][:, None] + dt).reshape(-1, NP)
            pr = tuple(x[todo].repeat_interleave(nt, 0) for x in pa)
            ft, ct = evaluate(ut, pr, hook, ids[todo].repeat_interleave(nt), chunk=max(o['chunk'], 64))
            nevals['trial'] += ut.shape[0]
            Pt = (ft ** 2).sum(1)
            if o['soft']: Pt = Pt + (torch.relu(ct) ** 2).sum(1)
            Pt = Pt.view(-1, nt)
            ct = torch.where(ctl[todo].repeat_interleave(nt, 0), ct, -1.0)
            Vt = torch.relu(ct).amax(1).view(-1, nt)
            St = torch.relu(ct - o['mdz']).sum(1).view(-1, nt)
            rk = rho_k[todo[:, None], ks]                                       # each trial is judged on its own merit
            Ps0 = Psi0[todo[:, None], ks]
            Pst = 0.5 * Pt + rk * St
            lim = Ps0 * ((1 + 0.02 * o['cnv']) if cnv_on else (1 - 1e-12))
            good = torch.isfinite(Pst) & (Pst < lim)
            gain = torch.where(good, (Ps0 - Pst) / Ps0.clamp(min=1e-30), torch.full_like(Pst, -float('inf')))
            j = gain.argmax(1); g1 = good.any(1)
            sel = todo[g1]; jj = j[g1]
            u_new[sel] = ut.view(-1, nt, NP)[g1, jj]
            phi_new[sel] = Pt[g1, jj]; v_new[sel] = Vt[g1, jj]; k_acc[sel] = ks[g1, jj]
            rho_acc[sel] = rk[g1, jj]; rel[sel] = gain[g1, jj]
            acc[sel] = True
            nb += int(bt > 0 and bool(g1.any()))
        # ---- bookkeeping, termination tests (IMP on the per-cycle merit change, as CODE V prints it)
        u[ids] = u_new
        ncyc[ids] += 1
        rho[ids] = torch.where(acc, rho_acc, rho[ids])
        # infeasible and no step accepted: the merit price of the violation is too low to leave this point.
        # Raise it (x10 per stuck cycle) so the constraints end up enforced, as CODE V's exact constraints are.
        stuck = ~acc & (V0 > o['vtol'])
        rho_floor[ids] = torch.where(stuck, (10 * rho_c).clamp(max=o['rho_max']), rho_floor[ids])
        lam_used[ids] = torch.where(acc, lams[k_acc], torch.full_like(Phi0, float('nan')))
        kmin[ids] = torch.where(acc, (k_acc - o['lam_drop']).clamp(min=0), (k1 + 2).clamp(max=L - 1))
        act_prev[ids] = actf[ar, k_acc]
        if Sbfg is not None:
            bfg_prev.update(ids=ids, s=torch.where(acc[:, None], u_new - ua, torch.zeros_like(ua)), Jf=Jf, cols=colt, g=g)
        bv, bp = best_v[ids], best_phi[ids]
        better = acc & (((v_new <= o['vtol']) & ((bv > o['vtol']) | (phi_new < bp))) | ((bv > o['vtol']) & (v_new < bv)))
        bi = ids[better]
        best_u[bi] = u_new[better]; best_phi[bi] = phi_new[better]; best_v[bi] = v_new[better]
        small = rel < o['imp']
        nsmall[ids] = torch.where(small, nsmall[ids] + 1, torch.zeros_like(nsmall[ids]))
        done = (ncyc[ids] >= o['mnc']) & (nsmall[ids] >= o['impn'])
        if o['tar'] is not None: done |= (phi_new <= o['tar']) & (v_new <= o['vtol'])
        active[ids[done]] = False
        sync()
        hist.append(dict(cyc=cyc, nact=int(B), best=float(phi_new.min()), median=float(phi_new.median()), vmax=float(v_new.max()),
                         acc=int(acc.sum()), nback=nb, lam_med=float(torch.nanmedian(lam_used[ids])) if acc.any() else float('nan'),
                         pred_gain=float(redp[ar, k1, a1].median()), rho_med=float(rho[ids].median()),
                         real_gain=float(rel.median()), t=time.time() - t0, dt=time.time() - tc))
        if o['verbose'] and (cyc % o['log_every'] == 0):
            h = hist[-1]
            print('  %s cyc %3d act %3d  best %.5g  med %.5g  Vmax %.3g  acc %d/%d bt %d  lam %.1e rho %.2g  gain pred %.3f real %.3f  (%.1fs, %.1fs)' % (
                o['tag'], cyc, B, h['best'], h['median'], h['vmax'], h['acc'], B, nb, h['lam_med'], h['rho_med'], h['pred_gain'], h['real_gain'], h['t'], h['dt']), flush=True)
    # ---- final: best design per candidate, fresh pupil, lm4-equivalent loss
    hook.set(ids_all)
    pupF = pupil(best_u, o['pupil'])
    f, c = evaluate(best_u, pupF, hook, ids_all)
    hook.set(ids_all)
    return dict(u=best_u, loss=lm4_loss(f, c), phi=(f ** 2).sum(1), viol=torch.relu(torch.where(ctl_all, c, -1.0)).amax(1),
                viol_raw=torch.relu(c).amax(1), c=c, pup=pupF, hist=hist,
                cycles=ncyc, time=time.time() - t0, nevals=nevals)


def _bfg_update(Sb, prev, ids, colt, Jf, f0, g, ua):
    """structured BFGS (Dennis-Gay-Welsch) of the second-order part S of the Hessian, H = J^T J + S"""
    if prev and torch.equal(prev['ids'], ids) and torch.equal(prev['cols'], colt):
        s = prev['s'][:, colt]
        y = g - prev['g']                                                 # gradient change
        ysh = g - (prev['Jf'] @ f0[..., None])[..., 0]                     # (J_new - J_old)^T f_new
        Sc = Sb[ids][:, colt][:, :, colt]
        ys = (y * s).sum(1)
        Ss = (Sc @ s[..., None])[..., 0]
        tau = ((s * ysh).sum(1).abs() / (s * Ss).sum(1).abs().clamp(min=1e-30)).clamp(max=1.0)
        Sc = Sc * tau[:, None, None]
        v = ysh - (Sc @ s[..., None])[..., 0]
        okk = ys > 1e-12 * (y.norm(dim=1) * s.norm(dim=1)).clamp(min=1e-30)
        ysd = torch.where(okk, ys, torch.ones_like(ys))
        U = (v[:, :, None] * y[:, None, :] + y[:, :, None] * v[:, None, :]) / ysd[:, None, None] \
            - ((v * s).sum(1) / ysd ** 2)[:, None, None] * y[:, :, None] * y[:, None, :]
        Sc = torch.where(okk[:, None, None], Sc + U, Sc)
        full = Sb[ids]; full[:, colt[:, None], colt[None, :]] = Sc; Sb[ids] = full
    Sc = Sb[ids][:, colt][:, :, colt]
    # keep J^T J + S positive semi-definite
    H0 = Jf @ Jf.transpose(1, 2)
    ev, Q = torch.linalg.eigh(H0 + Sc)
    ev = ev.clamp(min=0)
    return Q @ torch.diag_embed(ev) @ Q.transpose(1, 2) - H0


# ----------------------------------------------------------------------------------------------- drop-in wrappers
def lm(u, iters=60, tag='', **kw):
    """drop-in for lm4.lm(u, iters): returns (u, lm4-equivalent loss, pupil)"""
    kw.setdefault('mxc', iters)
    r = optimize(u, None, tag=tag, **kw)
    return r['u'], r['loss'], r['pup']


def lm_masked(u, varmask, iters, tag='', **kw):
    """masked DLS (varmask (B,NP) bool): returns (u, objective loss, pupil)"""
    kw.setdefault('mxc', iters)
    r = optimize(u, varmask, tag=tag, **kw)
    return r['u'], r['loss'], r['pup']
