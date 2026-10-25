"""Objective: fit the official geometric MTF chart (config targets) of the lens directly.

Geometric MTF from ray intercepts: MTF(nu) = |sum w exp(2 pi i nu x)| / sum w  (x: sagittal blur for S, y for M).
Fields at real image heights (config image.fields_mm; env LENS_FIELDS=yri, default) -- the field angles are solved per
design so the chief ray (reference wavelength, through the stop centre, through the cover glass) lands there, like
CODE V YRI; fixed with the vignetted pupil (CODE V setvig-like) for an optimisation cycle.  Aperture by F-number
(config aperture.fno), the stop has no fixed radius.  Official curves are interpolated at the real heights.
Constraints (rows of resid as penalties; dls.py uses them as exact constraints): EFL window, focus, asphere sag bands,
glass map, vd max, axial-beam clearance of every aperture, CRA at the image corner, relative illumination there.
CLI = continuous search with the CODE V-style DLS (dls.py):
  python objective.py --config cfg.json --nstart 512 --quick 15 --keep 96 --iters 60 --out cont.jsonl
  python objective.py --config cfg.json --starts previous.jsonl --keep 96 --iters 60 --out cont.jsonl
env LENS_NRING: pupil rings per field and wavelength (5 -> 91 rays; 8 -> 217; geometric MTF converges slowly)."""
import argparse, json, math, os, time, numpy as np, torch
import config as CF
import core as T
import params as P
import glassmap as GMP

dev, DT = T.dev, T.DT
NP, NW = CF.NP, T.NW
Z, LAST = CF.Z, CF.LAST
HTS = CF.FIELDS_MM
NF = len(HTS)
ANG = torch.tensor([math.atan(h / CF.EFL_NOM) for h in HTS], dtype=DT, device=dev)
FIELDS_MODE = os.environ.get('LENS_FIELDS', 'yri')
HT = torch.tensor(HTS, dtype=DT, device=dev)
EFL_LO, EFL_HI = CF.EFL_LO, CF.EFL_HI
_OFF = json.load(open(CF.MTF_FILE))[CF.MTF_KEY]
MKEYS = ['%g%s' % (f, a) for f in CF.FREQS for a in 'SM']
OH = torch.tensor(_OFF[MKEYS[0]]['h'], dtype=DT, device=dev)
OM = {k: torch.tensor(np.interp(_OFF[MKEYS[0]]['h'], _OFF[k]['h'], _OFF[k]['mtf']), dtype=DT, device=dev) for k in MKEYS}
W_MTF, W_SPOT = CF.W_MTF, CF.W_SPOT
VD_MAX = CF.VD_MAX
Z0 = T.ZS[0] - 20.0
CG_T, CG_GAP = CF.CG_T, CF.CG_GAP
FNO_SPEC = CF.FNO
RI_MIN, CRA_MAX, H_CORNER = CF.RI_MIN, CF.CRA_MAX, CF.H_CORNER
CRA_ANG = torch.tensor([math.radians(a) for a in CF.CRA_ANG_DEG], dtype=DT, device=dev)
IREF = CF.IREF


def hexgrid(nr):
    pts = [(0.0, 0.0)]
    for i in range(1, nr + 1):
        r = i / nr; n = 6 * i
        for j in range(n):
            a = 2 * math.pi * j / n
            pts.append((r * math.sin(a), r * math.cos(a)))
    return torch.tensor(pts, dtype=DT, device=dev)


NRINGS = int(os.environ.get('LENS_NRING', '5'))
HG = hexgrid(NRINGS)
NR = HG.shape[0]
FI = torch.arange(NF, device=dev)[:, None, None].expand(NF, NW, NR).reshape(-1)
IW = torch.arange(NW, device=dev)[None, :, None].expand(NF, NW, NR).reshape(-1)
HGR = HG[None, None].expand(NF, NW, NR, 2).reshape(-1, 2)
WR = T.WWt[IW]
NRING = 6 * NRINGS                                  # points of the outer pupil ring (relative illumination)


def interp(x, xp, fp):
    """batched linear interpolation of a 1-D table (xp increasing) at x (any shape)"""
    i = torch.searchsorted(xp, x.clamp(xp[0], xp[-1]).contiguous()).clamp(1, len(xp) - 1)
    x0, x1 = xp[i - 1], xp[i]; f0, f1 = fp[i - 1], fp[i]
    return f0 + (f1 - f0) * (x.clamp(xp[0], xp[-1]) - x0) / (x1 - x0)


def _to_image_y(Pl, Dl, t16):
    """height on the image of rays leaving the last surface (cover-glass propagation), as image_points"""
    Dz = Dl[..., 2].clamp(min=0.05)
    zl = Z[LAST] + t16[:, None]
    yl = Pl[..., 1] + (zl - Pl[..., 2]) / Dz * Dl[..., 1]
    ncg = float(T.NCG[IREF])
    sp = Dl[..., 1] / ncg; cp = torch.sqrt(torch.clamp(1 - sp ** 2, min=1e-6))
    return yl + sp / cp * CG_T + Dl[..., 1] / Dz * CG_GAP, Dz


def chief_image_height(surf, t16, zep, th):
    """real image height of the stop-centre chief ray at field angles th (B,K): 4-step secant aim, full trace"""
    B, K = th.shape
    D = torch.stack([torch.zeros_like(th), torch.sin(th), torch.cos(th)], -1)
    iw = torch.full((B, K), IREF, dtype=torch.long, device=dev)

    def ystop(y0):
        Pp = torch.stack([torch.zeros_like(y0), y0, torch.full_like(y0, Z0)], -1)
        return T.trace(surf[:T.ISTOP + 1], Pp, D, iw)[0][..., 1]
    ya = (Z0 - zep)[:, None] * torch.tan(th); yb = ya + 0.5
    sa, sb = ystop(ya), ystop(yb)
    for _ in range(4):
        yn = yb - sb * (yb - ya) / torch.where((sb - sa).abs() < 1e-9, torch.full_like(sb, 1e-9), sb - sa)
        ya, sa = yb, sb; yb = yn; sb = ystop(yb)
    Pp = torch.stack([torch.zeros_like(yb), yb, torch.full_like(yb, Z0)], -1)
    Pl, Dl, _ = T.trace(surf, Pp, D, iw)
    return _to_image_y(Pl, Dl, t16)[0]


def field_angles(surf, t16, efl, zep, iters=6):
    """(B,NF): fixed angles (LENS_FIELDS=ang) or solved so the chief ray lands at the real heights (yri)"""
    B = t16.shape[0]
    if FIELDS_MODE == 'ang':
        return ANG[None].expand(B, NF).clone()
    h = HT[None].expand(B, NF)
    ta = torch.atan(h / efl[:, None]); tb = ta * 1.02
    fa = chief_image_height(surf, t16, zep, ta) - h; fb = chief_image_height(surf, t16, zep, tb) - h
    for _ in range(iters):
        den = torch.where((fb - fa).abs() < 1e-12, torch.full_like(fb, 1e-12), fb - fa)
        tn = (tb - fb * (tb - ta) / den).clamp(0.0, 0.6)
        ta, fa = tb, fb; tb = tn; fb = chief_image_height(surf, t16, zep, tb) - h
    return torch.where(h > 0, tb, torch.zeros_like(tb))


def pupil_setup(u, nbis=22):
    """vignetted pupil per field like CODE V setvig (bisection of the meridional and sagittal rim rays, reference
    wavelength).  Beam = paraxial entrance-pupil disc of diameter EFL/FNO, clipped by every clear aperture except the
    stop.  Returns (yc, hy, hx, field angles) in the start plane z = Z0."""
    with torch.no_grad():
        p = P.params_to_phys(u); B = u.shape[0]
        surf = T.build(p)
        efl, zpar, epd, zep = T.parax(surf, p['t16'])
        ang = field_angles(surf, p['t16'], efl, zep)
        R = 0.5 * efl / FNO_SPEC
        tth = torch.tan(ang)
        ycp = (Z0 - zep)[:, None] * tth
        cir_ns = T.CIRS.clone(); cir_ns[T.ISTOP] = 1e3
        D = torch.stack([torch.zeros_like(ang), torch.sin(ang), torch.cos(ang)], -1)

        def passes(x, y):
            K = x.shape[-1]
            Pp = torch.stack([x, y, torch.full_like(x, Z0)], -1).reshape(B, NF * K, 3)
            Dd = D[:, :, None, :].expand(B, NF, K, 3).reshape(B, NF * K, 3).contiguous()
            iw = torch.full((B, NF * K), IREF, device=dev, dtype=torch.long)
            _, _, rs = T.trace(surf, Pp, Dd, iw)
            rs = torch.nan_to_num(rs, nan=1e3, posinf=1e3, neginf=1e3)
            ok = (rs <= cir_ns).all(-1).view(B, NF, K)
            yep = y + (zep[:, None, None] - Z0) * tth[:, :, None]
            return ok & ((x ** 2 + yep ** 2) <= (R[:, None, None] ** 2) * (1 + 1e-9))
        lo = torch.zeros(B, NF, 2, dtype=DT, device=dev); hi = torch.ones(B, NF, 2, dtype=DT, device=dev)
        sgn = torch.tensor([1.0, -1.0], dtype=DT, device=dev)
        for _ in range(nbis):
            m = 0.5 * (lo + hi)
            ok = passes(torch.zeros_like(m), ycp[..., None] + sgn * m * R[:, None, None])
            lo = torch.where(ok, m, lo); hi = torch.where(ok, hi, m)
        yup = ycp + lo[..., 0] * R[:, None]; ylo = ycp - lo[..., 1] * R[:, None]
        yc = 0.5 * (yup + ylo); hy = 0.5 * (yup - ylo)
        lo = torch.zeros(B, NF, 1, dtype=DT, device=dev); hi = torch.ones(B, NF, 1, dtype=DT, device=dev)
        for _ in range(nbis):
            m = 0.5 * (lo + hi)
            ok = passes(m * R[:, None, None], yc[..., None].expand_as(m).clone())
            lo = torch.where(ok, m, lo); hi = torch.where(ok, hi, m)
        hx = lo[..., 0] * R[:, None]
        hy = hy.clamp(min=1e-3) * 0.999; hx = hx.clamp(min=1e-3) * 0.999
    return yc, hy, hx, ang


def image_points(u, pup):
    yc, hy, hx = pup[:3]
    p = P.params_to_phys(u); B = u.shape[0]
    surf = T.build(p)
    efl, zpar, epd, zep = T.parax(surf, p['t16'])
    ang = pup[3] if len(pup) > 3 else ANG[None].expand(B, NF)
    th = ang[:, FI].detach()
    gx = HGR[None, :, 0] * hx[:, FI]; gy = yc[:, FI] + HGR[None, :, 1] * hy[:, FI]
    Pp = torch.stack([gx, gy, torch.full_like(gx, Z0)], -1)
    D = torch.stack([torch.zeros_like(th), torch.sin(th), torch.cos(th)], -1).contiguous()
    iw = IW[None].expand(B, -1)
    Pl, Dl, rs = T.trace(surf, Pp, D, iw)
    rs = torch.nan_to_num(rs, nan=1e3, posinf=1e3, neginf=1e3)
    zl = Z[LAST] + p['t16']
    Dz = Dl[..., 2].clamp(min=0.05)
    t = (zl[:, None] - Pl[..., 2]) / Dz
    Pc = Pl[..., :2] + t[..., None] * Dl[..., :2]
    ncg = T.NCG[iw]
    Dc = Dl[..., :2] / ncg[..., None]
    Dcz = torch.sqrt(torch.clamp(1 - (Dc ** 2).sum(-1), min=1e-6))
    Pi = Pc + Dc / Dcz[..., None] * CG_T + Dl[..., :2] / Dz[..., None] * CG_GAP
    Pi = torch.nan_to_num(Pi, nan=0.0, posinf=0.0, neginf=0.0).clamp(-50, 50)
    image_points.last_dir = Dl
    return p, efl, zpar, efl / FNO_SPEC, Pi, rs


def geo_mtf(Pf, Wf, nu):
    """Pf (B,NF,R,2) intercepts, Wf (B,NF,R) weights -> MTF_x, MTF_y at frequency nu (cycles/mm)"""
    sw = Wf.sum(-1)
    out = []
    for k in (0, 1):
        ph = 2 * math.pi * nu * Pf[..., k]
        c = (Wf * torch.cos(ph)).sum(-1) / sw; s = (Wf * torch.sin(ph)).sum(-1) / sw
        out.append(torch.sqrt(c * c + s * s + 1e-12))
    return out


def interp_h(hreal, val, h0):
    K = hreal.shape[1]
    j = (hreal < h0).sum(1).clamp(1, K - 1)
    y0 = hreal.gather(1, (j - 1)[:, None])[:, 0]; y1 = hreal.gather(1, j[:, None])[:, 0]
    v0 = val.gather(1, (j - 1)[:, None])[:, 0]; v1 = val.gather(1, j[:, None])[:, 0]
    return v0 + (v1 - v0) * (h0 - y0) / torch.where((y1 - y0).abs() < 1e-6, torch.full_like(y1, 1e-6), y1 - y0)


def chief_cra(u):
    """CRA (deg, = CODE V AOI R1 SI) interpolated at the image corner H_CORNER from stop-centre chief rays at CRA_ANG"""
    p = P.params_to_phys(u); B = u.shape[0]
    surf = T.build(p)
    efl, zpar, epd, zep = T.parax(surf, p['t16'])
    K = len(CRA_ANG)
    th = CRA_ANG[None].expand(B, K)
    D = torch.stack([torch.zeros_like(th), torch.sin(th), torch.cos(th)], -1)
    iw = torch.full((B, K), IREF, dtype=torch.long, device=dev)
    with torch.no_grad():
        ya = (Z0 - zep)[:, None] * torch.tan(th); yb = ya + 0.5

        def ystop(y0):
            Pp = torch.stack([torch.zeros_like(y0), y0, torch.full_like(y0, Z0)], -1)
            return T.trace(surf[:T.ISTOP + 1], Pp, D, iw)[0][..., 1]
        sa, sb = ystop(ya), ystop(yb)
        for _ in range(4):
            yn = yb - sb * (yb - ya) / torch.where((sb - sa).abs() < 1e-9, torch.full_like(sb, 1e-9), sb - sa)
            ya, sa = yb, sb; yb = yn; sb = ystop(yb)
    Pp = torch.stack([torch.zeros_like(yb), yb, torch.full_like(yb, Z0)], -1)
    Pl, Dl, _ = T.trace(surf, Pp, D, iw)
    yimg, Dz = _to_image_y(Pl, Dl, p['t16'])
    cra = torch.rad2deg(torch.atan2(torch.sqrt(Dl[..., 0] ** 2 + Dl[..., 1] ** 2), Dz))
    return interp_h(yimg, cra, H_CORNER), yimg, cra


def rel_illum(Dl, B):
    """RI per field = projected solid angle (area in image-space direction cosines L, M) of the vignetted beam relative
    to the axis, from the outer pupil ring at the reference wavelength"""
    D = Dl.view(B, NF, NW, NR, 3)[:, :, IREF, NR - NRING:, :]
    L, M = D[..., 0], D[..., 1]
    area = 0.5 * (L * torch.roll(M, -1, -1) - torch.roll(L, -1, -1) * M).sum(-1).abs()
    return area / area[:, :1].clamp(min=1e-12)


def sag_rows(p):
    """per variable asphere: list of (sag at each band height, lo, hi) tensors"""
    out = []
    for j, s in enumerate(CF.ASPH_VAR):
        if s not in CF.BANDS: continue
        c, A = p['c'][:, j], p['A'][:, j]
        for h, (lo, hi) in CF.BANDS[s].items():
            h2 = h * h
            out.append((c * h2 / (1 + torch.sqrt(torch.clamp(1 - c * c * h2, min=1e-12))) + A[:, 0] * h2 ** 2 + A[:, 1] * h2 ** 3 + A[:, 2] * h2 ** 4, lo, hi))
    return out


def resid(u, pup, detail=False):
    B = u.shape[0]
    p, efl, zpar, epd, Pi, rs = image_points(u, pup)
    w = WR[None].expand(B, -1)
    Wf = w.view(B, NF, NW * NR); Pf = Pi.view(B, NF, NW * NR, 2)
    sw = Wf.sum(-1)
    cen = (Wf[..., None] * Pf).sum(2) / sw[..., None]
    rr = (Pf - cen[:, :, None, :]) * torch.sqrt(Wf / sw[..., None])[..., None] / 0.010 * W_SPOT / math.sqrt(NR / 37)
    hreal = cen[..., 1].abs()
    ms_ = []
    for f in CF.FREQS:
        mx, my = geo_mtf(Pf, Wf, f); ms_ += [mx, my]
    tgt = {k: interp(hreal, OH, OM[k]) for k in MKEYS}
    mres = torch.cat([m - tgt[k] for m, k in zip(ms_, MKEYS)], 1) * W_MTF
    pens = [(torch.relu(efl - EFL_HI) + torch.relu(EFL_LO - efl)) / 0.05]
    zimg = Z[LAST] + p['t16'] + CG_T + CG_GAP
    pens.append(torch.relu((zpar - zimg).abs() - CF.DZ_MAX) / 0.02)
    for s, lo, hi in sag_rows(p):
        pens.append((torch.relu(lo - s) + torch.relu(s - hi)) / 0.003)
    gpen = GMP.penalty_rows(p['nd'], p['vd'])
    vdx = torch.relu(p['vd'] - VD_MAX) / 0.5
    ax = (FI == 0)
    margin = torch.full((T.NS,), 0.05, dtype=DT, device=dev); margin[T.ISTOP] = -1e3
    ex = torch.relu(rs[:, ax] - T.CIRS + margin).amax(1) / 0.02
    cra, _, _ = chief_cra(u)
    ri = rel_illum(image_points.last_dir, B)
    ri_c = interp_h(hreal, ri, H_CORNER)
    pcra = torch.stack([torch.relu(cra - (CRA_MAX - 0.2)) / 0.1, torch.relu(RI_MIN + 0.005 - ri_c) / 0.005], 1)
    r = torch.cat([rr.reshape(B, -1), mres, torch.stack(pens, 1), gpen, vdx, ex, pcra], 1)
    if detail:
        msq = ((Wf * ((Pf - cen[:, :, None, :]) ** 2).sum(-1)).sum(-1) / sw)
        return r, dict(efl=efl, fno=efl / epd, zpar=zpar, zimg=zimg, spots=torch.sqrt(msq) * 1000, hreal=hreal,
                       mtf=torch.stack(ms_, -1), tgt=torch.stack([tgt[k] for k in MKEYS], -1),
                       mtf_rms=torch.sqrt((mres / W_MTF) ** 2).mean(1), cra=cra, epd=epd, ri=ri_c, ri_f=ri)
    return r


def report(u, pup):
    with torch.no_grad():
        r, info = resid(u, pup, detail=True)
    out = []
    for b in range(u.shape[0]):
        x = P.u_to_phys_np(u[b].detach().cpu().numpy())
        out.append(dict(loss=float((r[b] ** 2).sum()), mtf_rms=float(info['mtf_rms'][b]), cra=float(info['cra'][b]),
                        ri=float(info['ri'][b]), epd=float(info['epd'][b]), efl=float(info['efl'][b]), fno=float(info['fno'][b]),
                        dz=float(info['zpar'][b] - info['zimg'][b]), spots_um=[round(float(v), 1) for v in info['spots'][b]],
                        mtf=info['mtf'][b].cpu().numpy().round(3).tolist(), tgt=info['tgt'][b].cpu().numpy().round(3).tolist(),
                        mtf_keys=MKEYS, hreal=info['hreal'][b].cpu().numpy().round(2).tolist(),
                        glass={e: [round(x[2 * i], 4), round(x[2 * i + 1], 2)] for i, e in enumerate(CF.ELEMS)},
                        R=P.radii(x), t16=round(x[CF.IT], 3), x=x.tolist()))
    return out


def load_starts(files, n):
    R = []
    for f in files.split(','):
        R += [json.loads(l) for l in open(f)]
    R.sort(key=lambda r: r['loss'])
    return torch.tensor(np.array([P.phys_to_u(np.array(r['x'])) for r in R[:n]]), dtype=DT, device=dev)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='continuous search (model glasses) with the CODE V-style DLS')
    ap.add_argument('--config'); ap.add_argument('--nstart', type=int, default=512); ap.add_argument('--quick', type=int, default=15)
    ap.add_argument('--keep', type=int, default=96); ap.add_argument('--iters', type=int, default=60)
    ap.add_argument('--chunk', type=int, default=48); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', required=True); ap.add_argument('--starts', default=None)
    a = ap.parse_args()
    import dls
    t0 = time.time()

    def run(u, iters, tag, log_every):
        r = dls.optimize(u, None, mxc=iters, imp=0.01, impn=3, soft=False, tag=tag, log_every=log_every)
        return r['u'], r['loss'], r['pup']
    if a.starts:
        U = load_starts(a.starts, a.keep)
    else:
        U = P.random_starts(a.nstart, a.seed)
        print('starts %d (%.0fs)' % (U.shape[0], time.time() - t0), flush=True)
        us, cs = [], []
        for k in range(0, U.shape[0], a.chunk):
            u, c, _ = run(U[k:k + a.chunk], a.quick, 'Q%d' % (k // a.chunk), a.quick)
            us.append(u); cs.append(c)
        U = torch.cat(us); C = torch.cat(cs)
        idx = torch.argsort(C)[:a.keep]; U = U[idx]
        print('QUICK kept %d  loss %.4g .. %.4g  (%.0fs)' % (len(idx), C[idx[0]].item(), C[idx[-1]].item(), time.time() - t0), flush=True)
    outs = []
    for k in range(0, U.shape[0], a.chunk):
        u, c, pup = run(U[k:k + a.chunk], a.iters, 'R%d' % (k // a.chunk), 10)
        outs += report(u, pup)
    outs.sort(key=lambda r: r['loss'])
    with open(a.out, 'a') as f:
        for r in outs:
            r['seed'] = a.seed; f.write(json.dumps(r) + '\n')
    for r in outs[:10]:
        print('   loss %.4g mtf %.4f CRA %.2f RI %.3f EFL %.2f dz %.2f | %s | R %s t %.2f' % (
            r['loss'], r['mtf_rms'], r['cra'], r['ri'], r['efl'], r['dz'],
            ' '.join('%s %.3f/%.1f' % (e, *v) for e, v in r['glass'].items()), r['R'], r['t16']), flush=True)
    print('total %.0fs  (EFL %.1f..%.1f, CRA <= %.1f, RI >= %.3f, back gap %.1f..%.1f, %d rays/field/wl)' % (
        time.time() - t0, EFL_LO, EFL_HI, CRA_MAX, RI_MIN, CF.T_LO, CF.T_HI, NR))
