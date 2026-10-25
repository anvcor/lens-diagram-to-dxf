"""rcore -- the Rust ray-trace engine (rust/lenscore, C ABI, loaded with ctypes) behind dls.py.

Same quantities as dls.phi_fn / dls.jac_ad / dls.pupil_fast, on the CPU (all cores, float64):
  phi(u, pup)            features (B, NFEAT): image intercepts, EFL, focus error, corner CRA, RI per field,
                         max axial-beam height per surface
  jac(u, pup, cols)      phi0 and Jphi (B, n, NFEAT): exact forward-mode derivatives (dual numbers in Rust)
  pupil(u)               (yc, hy, hx, field angles), same search as dls.pupil_fast
All ray traces and their derivatives run in Rust; the cheap per-design parts (variables -> surface data and its small
Jacobian, paraxial EFL / focus, RI from the rim directions, CRA interpolation) stay in torch, so a replaced
core.build (fixed catalog glasses, catalog attraction) works unchanged.
Local dual parameters per ray: for every variable asphere c, A4, A6, A8; the back gap; the index of every variable
element at the ray's wavelength; then start x, y.  Library: env LENSCORE_DLL, default lenscore.dll / liblenscore.so
next to this file (rust/build.sh)."""
import ctypes, os, numpy as np, torch
from torch.func import jvp, vmap
import config as CF
import objective as OB
import params as P
import core as T

_L = ctypes.CDLL(CF._DLL)
_vp, _sz, _i32, _f = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int32, ctypes.c_double


class _Args(ctypes.Structure):
    _fields_ = [('nb', _sz), ('nr', _sz), ('ns', _sz), ('nw', _sz), ('nu', _sz), ('nq', _sz),
                ('zs', _vp), ('asph', _vp), ('c', _vp), ('a', _vp), ('n', _vp),
                ('cslot', _vp), ('aslot', _vp), ('nslot', _vp), ('m', _vp),
                ('p0', _vp), ('d0', _vp), ('iw', _vp), ('seed', _vp),
                ('k0', _sz), ('k1', _sz), ('mode', _i32),
                ('t16', _vp), ('t16slot', _i32), ('zlast', _f), ('cg_t', _f), ('cg_gap', _f), ('ncg', _vp),
                ('group', _vp), ('ngroups', _sz), ('ndd', _sz), ('dd_idx', _vp),
                ('p_out', _vp), ('d_out', _vp), ('dp_out', _vp), ('dd_out', _vp),
                ('rs_out', _vp), ('rsmax', _vp), ('drsmax', _vp)]


_L.lc_trace.restype = _i32
_L.lc_trace.argtypes = [ctypes.POINTER(_Args)]
_L.lc_nthreads.restype = _sz
_L.lc_version.restype = _i32
assert _L.lc_version() >= 2, 'lenscore library too old: rebuild with rust/build.sh'

DT = torch.float64
NS, NW, NF, NR = T.NS, T.NW, OB.NF, OB.NR
NRAY = NF * NW * NR
NE, NA = CF.NE, CF.NA
NQ = 4 * NA + 1 + NE                                   # local lens parameters (start x, y follow)
T16SLOT = 4 * NA
ZS = np.ascontiguousarray(np.array(T.ZS, float))
ASPH = np.ascontiguousarray(np.array([1 if k in T.NEWTON else 0 for k in range(NS)], np.int32))
CSLOT = np.full(NS, -1, np.int32); ASLOT = np.full((NS, 3), -1, np.int32); NSLOT = np.full(NS, -1, np.int32)
for _j, _s in enumerate(CF.ASPH_VAR):
    _k = CF.ORDER.index(_s); CSLOT[_k] = 4 * _j; ASLOT[_k] = [4 * _j + 1, 4 * _j + 2, 4 * _j + 3]
for _k, _s in enumerate(CF.ORDER):
    if _s in CF.AFTER and CF.AFTER[_s] in CF.ELEMS:
        NSLOT[_k] = 4 * NA + 1 + CF.ELEMS.index(CF.AFTER[_s])
ELEM_SURF = [int(np.nonzero(NSLOT == 4 * NA + 1 + i)[0][0]) for i in range(NE)]
NCG = np.ascontiguousarray(T.NCG.detach().cpu().numpy().astype(float))
ZLAST, Z0, IREF = float(CF.Z[CF.LAST]), float(OB.Z0), OB.IREF
CIRS_NS = T.CIRS.detach().cpu().numpy().copy(); CIRS_NS[T.ISTOP] = 1e3
FI = OB.FI.cpu().numpy(); IWR = OB.IW.cpu().numpy().astype(np.int32); HGR = OB.HGR.cpu().numpy(); ANG = OB.ANG.cpu().numpy()
AXIS_GROUP = np.where(FI == 0, 0, -1).astype(np.int32)
RING = np.array([(f * NW + IREF) * NR + j for f in range(NF) for j in range(NR - OB.NRING, NR)], np.int32)


def _p(a):
    return None if a is None else a.ctypes.data


def _c(a, dt=float):
    return np.ascontiguousarray(a, dtype=dt)


def nthreads():
    return int(_L.lc_nthreads())


def trace(sysv, P0, D0, iw, M=None, seed=None, k0=0, k1=NS, mode=0, t16=None, group=None, dd_idx=None, want_rs=False):
    """lc_trace wrapper.  sysv = (c (B,NS), A (B,NS,4), n (B,NS,NW)) numpy.  Returns a dict of numpy arrays."""
    c, A, n = (_c(x) for x in sysv)
    B, nr = P0.shape[0], P0.shape[1]
    nu = 0 if M is None else M.shape[-1]
    P0, D0, iw = _c(P0), _c(D0), _c(iw, np.int32)
    M = None if M is None else _c(M)
    seed = None if seed is None else _c(seed)
    out = dict(P=np.empty((B, nr, 3)), D=np.empty((B, nr, 3)))
    ndd = 0 if dd_idx is None else len(dd_idx)
    if nu:
        out['dP'] = np.zeros((B, nr, 2 if mode == 1 else 3, nu))
        out['dD'] = np.zeros((B, ndd, 3, nu)) if mode == 1 else np.zeros((B, nr, 3, nu))
    if want_rs: out['rs'] = np.empty((B, nr, NS))
    ng = 0
    if group is not None:
        ng = int(group.max()) + 1; group = _c(group, np.int32)
        out['rsmax'] = np.zeros((B, ng, NS))
        if nu: out['drsmax'] = np.zeros((B, ng, NS, nu))
    t16 = None if t16 is None else _c(t16)
    dd_idx = None if dd_idx is None else _c(dd_idx, np.int32)
    a = _Args(B, nr, NS, NW, nu, NQ, _p(ZS), _p(ASPH), _p(c), _p(A), _p(n), _p(CSLOT), _p(ASLOT), _p(NSLOT), _p(M),
              _p(P0), _p(D0), _p(iw), _p(seed), k0, k1, mode, _p(t16), T16SLOT, ZLAST, float(CF.CG_T), float(CF.CG_GAP),
              _p(NCG), _p(group), ng, ndd, _p(dd_idx), _p(out['P']), _p(out['D']), _p(out.get('dP')), _p(out.get('dD')),
              _p(out.get('rs')), _p(out.get('rsmax')), _p(out.get('drsmax')))
    rc = _L.lc_trace(ctypes.byref(a))
    if rc != 0:
        raise RuntimeError('lc_trace failed (%d)' % rc)
    return out


# ------------------------------------------------------------------------------------------------ per-design data
def _local(u):
    """variables -> (geo (B,4NA+1): c A4 A6 A8 per asphere, back gap; element indices (B,NE,NW); EFL; focus error)"""
    p = P.params_to_phys(u)
    surf = T.build(p)
    c = torch.stack([s[0] for s in surf], 1); A = torch.stack([s[1] for s in surf], 1); n = torch.stack([s[2] for s in surf], 1)
    parts = []
    for s in CF.ASPH_VAR:
        k = CF.ORDER.index(s); parts += [c[:, k:k + 1], A[:, k, :3]]
    geo = torch.cat(parts + [p['t16'][:, None]], 1)
    nel = n[:, ELEM_SURF, :] if NE else torch.zeros(u.shape[0], 0, NW, dtype=u.dtype, device=u.device)
    efl, zpar, epd, zep = T.parax(surf, p['t16'])
    dz = zpar - (CF.Z[CF.LAST] + p['t16'] + CF.CG_T + CF.CG_GAP)
    return geo, nel, efl, dz, zep, (c, A, n)


def _system(u, cols=None):
    with torch.no_grad():
        geo, nel, efl, dz, zep, (c, A, n) = _local(u)
    res = dict(sys=tuple(x.detach().cpu().numpy() for x in (c, A, n)), geo=geo.detach().cpu().numpy(), efl=efl, dz=dz,
               zep=zep.detach().cpu().numpy())
    if cols is not None:
        B = u.shape[0]; nc = len(cols)
        E = torch.zeros(nc, B, u.shape[1], dtype=u.dtype, device=u.device)
        E[torch.arange(nc), :, torch.tensor(cols)] = 1.0
        f = lambda uu: tuple(_local(uu)[:4])
        _, (dgeo, dnel, defl, ddz) = vmap(lambda t: jvp(f, (u,), (t,)))(E)
        dgeo = dgeo.permute(1, 2, 0).detach().cpu().numpy()                       # (B,4NA+1,n)
        dnel = dnel.permute(1, 3, 2, 0).detach().cpu().numpy()                    # (B,NW,NE,n)
        M = np.zeros((B, NW, NQ, nc)); M[:, :, :4 * NA + 1, :] = dgeo[:, None]; M[:, :, 4 * NA + 1:, :] = dnel
        res.update(M=M, dgeo=dgeo, defl=defl.detach(), ddz=ddz.detach())
    return res


# ------------------------------------------------------------------------------------------------ fields, pupil
def _chief_height(S, th):
    B, K = th.shape
    D = np.stack([np.zeros_like(th), np.sin(th), np.cos(th)], -1)
    iw = np.full((B, K), IREF, np.int32)

    def ystop(y):
        return trace(S['sys'], np.stack([np.zeros_like(y), y, np.full_like(y, Z0)], -1), D, iw, k1=T.ISTOP + 1)['P'][..., 1]
    ya = (Z0 - S['zep'])[:, None] * np.tan(th); yb = ya + 0.5
    sa, sb = ystop(ya), ystop(yb)
    for _ in range(4):
        den = sb - sa; den = np.where(np.abs(den) < 1e-9, 1e-9, den)
        yn = yb - sb * (yb - ya) / den
        ya, sa = yb, sb; yb = yn; sb = ystop(yb)
    o = trace(S['sys'], np.stack([np.zeros_like(yb), yb, np.full_like(yb, Z0)], -1), D, iw)
    Pl, Dl = o['P'], o['D']
    Dz = np.maximum(Dl[..., 2], 0.05)
    yl = Pl[..., 1] + (ZLAST + S['geo'][:, T16SLOT][:, None] - Pl[..., 2]) / Dz * Dl[..., 1]
    ncg = float(T.NCG[IREF])
    sp = Dl[..., 1] / ncg; cp = np.sqrt(np.maximum(1 - sp ** 2, 1e-6))
    return yl + sp / cp * CF.CG_T + Dl[..., 1] / Dz * CF.CG_GAP


def field_angles(S, B, iters=6):
    if OB.FIELDS_MODE == 'ang':
        return np.broadcast_to(ANG[None], (B, NF)).copy()
    h = np.broadcast_to(np.array(OB.HTS, float)[None], (B, NF))
    efl = S['efl'].detach().cpu().numpy()
    ta = np.arctan(h / efl[:, None]); tb = ta * 1.02
    fa = _chief_height(S, ta) - h; fb = _chief_height(S, tb) - h
    for _ in range(iters):
        den = fb - fa; den = np.where(np.abs(den) < 1e-12, 1e-12, den)
        tn = np.clip(tb - fb * (tb - ta) / den, 0.0, 0.6)
        ta, fa = tb, fb; tb = tn; fb = _chief_height(S, tb) - h
    return np.where(h > 0, tb, 0.0)


def pupil(u, nsec=16, rounds=6):
    S = _system(u)
    B = u.shape[0]
    efl = S['efl'].detach().cpu().numpy(); zep = S['zep']
    ang = field_angles(S, B)
    R = 0.5 * efl / OB.FNO_SPEC
    tth = np.tan(ang); ycp = (Z0 - zep)[:, None] * tth
    Dv = np.stack([np.zeros_like(ang), np.sin(ang), np.cos(ang)], -1)

    def passes(x, y):
        K = x.shape[-1]
        P0 = np.stack([x, y, np.full_like(x, Z0)], -1).reshape(B, NF * K, 3)
        D0 = np.broadcast_to(Dv[:, :, None, :], (B, NF, K, 3)).reshape(B, NF * K, 3)
        o = trace(S['sys'], P0, D0, np.full((B, NF * K), IREF, np.int32), want_rs=True)
        ok = (o['rs'] <= CIRS_NS).all(-1).reshape(B, NF, K)
        yep = y + (zep[:, None, None] - Z0) * tth[:, :, None]
        return ok & ((x ** 2 + yep ** 2) <= (R[:, None, None] ** 2) * (1 + 1e-9))
    fr = np.linspace(0, 1, nsec + 1)[1:-1]

    def search(lo, hi, test):
        for _ in range(rounds):
            t = lo[..., None] + (hi - lo)[..., None] * fr
            fail = ~test(t)
            idx = np.broadcast_to(np.arange(fr.size), fail.shape)
            k = np.where(fail, idx, fr.size).min(-1)
            tpad = np.concatenate([lo[..., None], t, hi[..., None]], -1)
            lo, hi = np.take_along_axis(tpad, k[..., None], -1)[..., 0], np.take_along_axis(tpad, (k + 1)[..., None], -1)[..., 0]
        return lo
    sgn = np.array([1.0, -1.0])

    def test_m(t):
        Kk = t.shape[-1]
        y = ycp[..., None, None] + sgn[None, None, :, None] * t * R[:, None, None, None]
        return passes(np.zeros_like(y).reshape(B, NF, 2 * Kk), y.reshape(B, NF, 2 * Kk)).reshape(B, NF, 2, Kk)
    lo = search(np.zeros((B, NF, 2)), np.ones((B, NF, 2)), test_m)
    yup = ycp + lo[..., 0] * R[:, None]; ylo = ycp - lo[..., 1] * R[:, None]
    yc = 0.5 * (yup + ylo); hy = 0.5 * (yup - ylo)
    lo = search(np.zeros((B, NF)), np.ones((B, NF)), lambda t: passes(t * R[:, None, None], np.broadcast_to(yc[..., None], t.shape).copy()))
    hx = lo * R[:, None]
    hy = np.maximum(hy, 1e-3) * 0.999; hx = np.maximum(hx, 1e-3) * 0.999
    to = lambda a: torch.tensor(a, dtype=DT, device=u.device)
    return to(yc), to(hy), to(hx), to(ang)


# ------------------------------------------------------------------------------------------------ features
def _rays(pup):
    yc, hy, hx = (x.detach().cpu().numpy() for x in pup[:3])
    B = yc.shape[0]
    ang = pup[3].detach().cpu().numpy() if len(pup) > 3 else np.broadcast_to(ANG[None], (B, NF))
    gx = HGR[None, :, 0] * hx[:, FI]; gy = yc[:, FI] + HGR[None, :, 1] * hy[:, FI]
    P0 = np.stack([gx, gy, np.full_like(gx, Z0)], -1)
    th = ang[:, FI]
    D0 = np.stack([np.zeros_like(th), np.sin(th), np.cos(th)], -1)
    return P0, D0, np.broadcast_to(IWR[None], (B, NRAY))


def _ri(Dr):
    B = Dr.shape[0]
    D = Dr.reshape(B, NF, OB.NRING, 3)
    L, Mm = D[..., 0], D[..., 1]
    area = 0.5 * (L * torch.roll(Mm, -1, -1) - torch.roll(L, -1, -1) * Mm).sum(-1).abs()
    return area / area[:, :1].clamp(min=1e-12)


def _cra_post(Pl, Dl, t16):
    yimg, Dz = OB._to_image_y(Pl, Dl, t16)
    cra = torch.rad2deg(torch.atan2(torch.sqrt(Dl[..., 0] ** 2 + Dl[..., 1] ** 2), Dz))
    return OB.interp_h(yimg, cra, OB.H_CORNER)


def _cra(S, B, M=None):
    """dls.cra_corner: secant aim of the stop-centre chief rays (values), one implicit Newton step for the derivative"""
    th = OB.CRA_ANG.cpu().numpy(); K = th.size
    Dv = np.broadcast_to(np.stack([np.zeros_like(th), np.sin(th), np.cos(th)], -1)[None], (B, K, 3))
    iw = np.full((B, K), IREF, np.int32)
    k1 = T.ISTOP + 1

    def ystop(y, D=Dv, ii=iw):
        return trace(S['sys'], np.stack([np.zeros_like(y), y, np.full_like(y, Z0)], -1), D, ii, k1=k1)['P'][..., 1]
    ya = (Z0 - S['zep'])[:, None] * np.tan(th)[None]; yb = ya + 0.5
    sa, sb = ystop(ya), ystop(yb)
    for _ in range(4):
        den = sb - sa; den = np.where(np.abs(den) < 1e-9, 1e-9, den)
        yn = yb - sb * (yb - ya) / den
        ya, sa = yb, sb; yb = yn; sb = ystop(yb)
    e = 1e-4
    sp = ystop(np.concatenate([yb + e, yb - e], 1), np.concatenate([Dv, Dv], 1), np.concatenate([iw, iw], 1))
    slope = (sp[:, :K] - sp[:, K:]) / (2 * e); slope = np.where(np.abs(slope) < 1e-6, 1e-6, slope)
    P0 = np.stack([np.zeros_like(yb), yb, np.full_like(yb, Z0)], -1)
    t16 = torch.tensor(S['geo'][:, T16SLOT], dtype=DT)
    if M is None:
        y0 = yb - ystop(yb) / slope
        o = trace(S['sys'], np.stack([np.zeros_like(y0), y0, np.full_like(y0, Z0)], -1), Dv, iw)
        return _cra_post(torch.tensor(o['P'], dtype=DT), torch.tensor(o['D'], dtype=DT), t16), None
    nu = M.shape[-1]
    os_ = trace(S['sys'], P0, Dv, iw, M=M, k1=k1)
    s, ds = os_['P'][..., 1], os_['dP'][..., 1, :]
    y0 = yb - s / slope; dy0 = -ds / slope[..., None]
    seed = np.zeros((B, K, 2, nu)); seed[:, :, 1, :] = dy0
    o = trace(S['sys'], np.stack([np.zeros_like(y0), y0, np.full_like(y0, Z0)], -1), Dv, iw, M=M, seed=seed)
    Pl, Dl = torch.tensor(o['P'], dtype=DT), torch.tensor(o['D'], dtype=DT)
    tP = torch.tensor(o['dP'], dtype=DT).permute(3, 0, 1, 2); tD = torch.tensor(o['dD'], dtype=DT).permute(3, 0, 1, 2)
    tt = torch.tensor(S['dgeo'][:, T16SLOT, :], dtype=DT).T
    val, d = vmap(lambda a, b, c: jvp(_cra_post, (Pl, Dl, t16), (a, b, c)))(tP, tD, tt)
    return val[0], d


def phi(u, pup):
    """features (B, NFEAT), values only (== dls.phi_fn)"""
    B = u.shape[0]
    S = _system(u)
    P0, D0, iw = _rays(pup)
    o = trace(S['sys'], P0, D0, iw, mode=1, t16=S['geo'][:, T16SLOT], group=np.broadcast_to(AXIS_GROUP[None], (B, NRAY)))
    Pi = torch.tensor(o['P'][..., :2], dtype=DT)
    ri = _ri(torch.tensor(o['D'][:, RING, :], dtype=DT))
    cra, _ = _cra(S, B)
    rsax = torch.tensor(o['rsmax'][:, 0, :], dtype=DT)
    f = torch.cat([Pi.reshape(B, -1), S['efl'].detach().cpu()[:, None], S['dz'].detach().cpu()[:, None], cra[:, None], ri, rsax], 1)
    return f.to(u.device)


def jac(u, pup, cols):
    """phi0 (B, NFEAT) and Jphi (B, n, NFEAT) w.r.t. the variables in cols (== dls.jac_ad)"""
    B = u.shape[0]; nc = len(cols)
    S = _system(u, cols)
    P0, D0, iw = _rays(pup)
    o = trace(S['sys'], P0, D0, iw, M=S['M'], mode=1, t16=S['geo'][:, T16SLOT], group=np.broadcast_to(AXIS_GROUP[None], (B, NRAY)), dd_idx=RING)
    Pi = torch.tensor(o['P'][..., :2], dtype=DT)
    dPi = torch.tensor(o['dP'], dtype=DT)
    Dr = torch.tensor(o['D'][:, RING, :], dtype=DT); dDr = torch.tensor(o['dD'], dtype=DT).permute(3, 0, 1, 2)
    ri, dri = vmap(lambda t: jvp(_ri, (Dr,), (t,)))(dDr)
    ri = ri[0]
    cra, dcra = _cra(S, B, S['M'])
    rsax = torch.tensor(o['rsmax'][:, 0, :], dtype=DT); drsax = torch.tensor(o['drsmax'][:, 0, :, :], dtype=DT)
    efl, dz = S['efl'].detach().cpu(), S['dz'].detach().cpu()
    phi0 = torch.cat([Pi.reshape(B, -1), efl[:, None], dz[:, None], cra[:, None], ri, rsax], 1)
    J = torch.cat([dPi.permute(0, 3, 1, 2).reshape(B, nc, -1), S['defl'].cpu().T[..., None], S['ddz'].cpu().T[..., None],
                   dcra.T[..., None], dri.permute(1, 0, 2), drsax.permute(0, 2, 1)], 2)
    return phi0.to(u.device), J.to(u.device)
