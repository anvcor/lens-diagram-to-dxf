"""deliver.py -- CODE V deliverables from an all-catalog solver design (no CODE V needed).

  <out>.seq      CODE V lens, INF only: catalog glasses, the config's radii/thicknesses, solved aspheres and back gap,
                 FNO aperture with a free stop (no stop CIR), fields at real image heights (YRI), cover glass
  <out>_opt.seq  optimisation macro (CODE V AUT): EFL window, sphere radii +-rdy_tol, asphere vertex |R| range,
                 CRA at the corner, back-gap range; variables are NOT set (the designer sets them)
  <out>.xlsx     surface table
usage: python deliver.py run_tight.jsonl --rank 0 --out E:/.../Lens_INF      (rank among feasible designs by mtf_rms)
       python deliver.py run_tight.jsonl --idx 3 --out ...                    (line index in the file)
Config keys used (section "codev"): title, notes, gla, asph_R_range, rdy_tol, yri_fractions, opt_lines."""
import argparse, json, os, numpy as np, torch
import config as CF
import objective as OB
import params as P
import core as T
import snap as SN
from glass import CAT_ALL, available, nidx, WL

CV = CF.CODEV


def cvname(v, n):
    return n.replace('-', '') + '_' + v


def load_records(path):
    if path.endswith('.jsonl'):
        return [json.loads(l) for l in open(path)]
    J = json.load(open(path))
    return J['results'] if isinstance(J, dict) else J


def pick(R, rank=None, idx=None):
    if idx is not None:
        return R[idx]
    R = sorted(R, key=lambda r: (not SN.feasible(r), r['mtf_rms']))
    return R[rank or 0]


def glasses_of(rec):
    G = {e: tuple(rec['glasses'][e].split(':')) for e in CF.ELEMS}
    G.update(CF.FIXED)
    return G


def real_epd(rec):
    """diameter of the axial beam (reference wavelength) that just fills the stop, from a real trace"""
    keys = [tuple(rec['glasses'][e].split(':')) for e in CF.ELEMS]
    u = np.array(P.phys_to_u(np.array(rec['x'])))
    for i, k in enumerate(keys): SN.set_u_glass(u, i, k)
    U = torch.tensor(u[None], dtype=T.DT, device=T.dev)
    SN.init_state(1, [keys])
    surf = T.build(P.params_to_phys(U))[:T.ISTOP + 1]
    iw = torch.full((1, 1), CF.IREF, dtype=torch.long, device=T.dev)

    def ystop(y0):
        Pp = torch.tensor([[[0.0, y0, CF.ZS[0] - 20.0]]], dtype=T.DT, device=T.dev)
        D = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=T.DT, device=T.dev)
        with torch.no_grad():
            Pe, _, _ = T.trace(surf, Pp, D, iw)
        return float(Pe[0, 0, 1]) - CF.CIR[CF.STOP]
    a = 0.5 * rec['epd']; b = 1.02 * a
    for _ in range(40):
        fa, fb = ystop(a), ystop(b)
        if abs(fb - fa) < 1e-14: break
        a, b = b, b - fb * (b - a) / (fb - fa)
    T.build = SN.ORIG_BUILD
    return 2 * b


def surfaces(rec, cg=True):
    """CODE V surface list (solver-only planes dropped): dict name, R, t, glass key, A (4), cir, stop"""
    x = np.array(rec['x']); G = glasses_of(rec)
    OO = [s for s in CF.ORDER if s not in CF.SOLVER_ONLY]
    back = float(x[CF.IT])
    out = []
    for i, s in enumerate(OO):
        if i + 1 < len(OO):
            t = round(CF.Z[OO[i + 1]] - CF.Z[s], 4)
        elif cg or not CF.CG_T:
            t = round(back, 4)
        else:                                             # no cover glass: air-equivalent distance to the image
            ncg = nidx(CAT_ALL[CF.CG_GLASS], 0.5876)
            t = round(back + CF.CG_T / ncg + CF.CG_GAP, 4)
        A = None
        if s in CF.ASPH_VAR:
            i0 = CF.IA[s]; c = x[i0]
            R = 1 / c if abs(c) > 1e-12 else 0.0
            A = list(x[i0 + 1:i0 + 4] * CF.ASC) + [0.0]
        elif s in CF.ASPH_FIXED:
            R = CF.R0[s] or 0.0; A = list(CF.ASPH_FIXED[s])
        else:
            R = CF.R0[s] or 0.0
        e = CF.AFTER.get(s)
        out.append(dict(name=s, R=R, t=t, glass=G[e] if e else None, A=A, cir=CF.CIR[s], stop=(s == CF.STOP), elem=e))
    return out


def write_seq(rec, out, cg=True, fno=None):
    S = surfaces(rec, cg)
    G = glasses_of(rec)
    for e, (v, n) in G.items():
        if not available(v, n): print('WARNING: %s %s:%s is not a currently available catalog glass' % (e, v, n))
    fr = CV.get('yri_fractions', [0, 0.3, 0.5, 0.7, 0.85, 1.0])
    yri = [round(f * CF.H_CORNER, 3) for f in fr]
    L = ["RDM;LEN", "TITLE '%s'" % CV.get('title', CF.TITLE + ' - restoration, INF').replace("'", '')]
    L += ['! ' + n for n in CV.get('notes', [])]
    L += ["FNO %g" % (fno or CF.FNO), "DIM M", "WL " + ' '.join('%g' % w for w in CF.WL_NM), "REF %d" % (CF.IREF + 1),
          "WTW " + ' '.join('%g' % w for w in CF.WW_RAW), "YRI " + ' '.join('%g' % y for y in yri),
          "WTF " + ' '.join('1' for _ in yri), "SO 0.0 0.1e11"]
    for s in S:
        if s['stop']:
            L += ["S 0.0 %g" % s['t'], "  STO"]           # free stop: the aperture is set by FNO
            continue
        gl = cvname(*s['glass']) if s['glass'] else ''
        L.append(("S %.6f %g %s" % (s['R'], s['t'], gl)).rstrip())
        if s['A'] is not None:
            L += ["  ASP", "  K 0.0", "  A %.8e; B %.8e; C %.8e; D %.8e" % tuple(s['A'])]
        L.append("  CIR %g" % s['cir'])
    if cg and CF.CG_T:
        L += ["S 0.0 %g %s" % (CF.CG_T, cvname(*CF.CG_GLASS)), "S 0.0 %g" % CF.CG_GAP]
    L += ["SI 0.0 0.0", "GO"]
    open(out + '.seq', 'w').write('\n'.join(L) + '\n')
    return S, len(yri)


def write_opt(S, out, nfield, cg=True):
    lo_R, hi_R = CV.get('asph_R_range', [30, 500])
    tol = float(CV.get('rdy_tol', 0.05))
    nlast = len(S)                                        # CODE V number of the last lens surface (back-gap thickness)
    nimg = nlast + (2 if (cg and CF.CG_T) else 0)
    M = ["! %s -- optimization macro (INF only)" % CF.TITLE,
         "! Run after the lens is loaded (IN %s.seq)." % os.path.basename(out),
         "! Spheres: radius kept within +-%g%% of the start value (RDY)." % (100 * tol),
         "! Aspheres: vertex R same sign as start, |R| %g..%g." % (lo_R, hi_R),
         "! Variables are NOT set by this macro -- set them in the lens before running.", "", "AUT", "",
         "EFL Z1 > %g" % CF.EFL_LO, "EFL Z1 < %g" % CF.EFL_HI, "", "! ---- asphere vertex radius ----"]
    sph = []
    for k, s in enumerate(S, 1):
        if s['stop'] or not s['R']: continue
        if s['A'] is not None:
            M += (["RDY S%d > %g" % (k, lo_R), "RDY S%d < %g" % (k, hi_R)] if s['R'] > 0 else
                  ["RDY S%d > %g" % (k, -hi_R), "RDY S%d < %g" % (k, -lo_R)])
        else:
            sph.append((k, s['R']))
    M += ["", "! ---- sphere radius, +-%g%% of start value ----" % (100 * tol)]
    for k, R in sph:
        lo, hi = sorted((R * (1 - tol), R * (1 + tol)))
        M += ["RDY S%d < %.4f" % (k, hi), "RDY S%d > %.4f" % (k, lo)]
    M += ["", "! chief ray angle at the sensor corner (fields are real image heights YRI; F%d = %g mm)" % (nfield, CF.H_CORNER),
          "AOI R1 SI F%d Z1 < %g" % (nfield, CF.CRA_MAX), "", "! back focus: last lens surface -> cover glass / image",
          "THI S%d > %g" % (nlast, CF.T_LO), "THI S%d < %g" % (nlast, CF.T_HI)]
    if nimg != nlast:
        M.append("THI S%d > 0.3" % nimg)
    M += [""] + CV.get('opt_lines', ["INT NO", "MXC 30", "MNC 30", "!EFT WFR", "EFT TA", "DRA S1..I YES", "",
                                     "EFP ALL Y", "MNT 0.8", "MNE 0.5", "MNA 0.05"])
    if CV.get('gla'):
        M += ["", "GLA S1..I " + CV['gla']]
    M += ["", "GO"]
    open(out + '_opt.seq', 'w').write('\r\n'.join(M) + '\r\n')


def write_xlsx(S, out, cg=True):
    try:
        from openpyxl import Workbook
    except ImportError:
        print('xlsx skipped: openpyxl not installed'); return
    wb = Workbook(); ws = wb.active; ws.title = 'Surfaces'
    ws.append(['Surface', 'Radius (mm)', 'Thickness (mm)', 'Glass', 'nd', 'vd', 'Clear semi-aperture (mm)', 'Asphere', 'Name'])
    ws.append(['OBJ', 'Infinity', 'Infinity', '', '', '', '', '', ''])
    for k, s in enumerate(S, 1):
        g = CAT_ALL.get(s['glass']) if s['glass'] else None
        ws.append(['STO' if s['stop'] else k, 'Infinity' if not s['R'] else round(s['R'], 4), s['t'],
                   '%s (%s)' % (s['glass'][1], s['glass'][0]) if s['glass'] else '', g['nd'] if g else '', g['vd'] if g else '',
                   s['cir'], ('A4..A10 = ' + ' '.join('%.4e' % c for c in s['A'])) if s['A'] is not None else '', s['name']])
    if cg and CF.CG_T:
        g = CAT_ALL.get(CF.CG_GLASS)
        ws.append(['CG', 'Infinity', CF.CG_T, '%s (%s)' % (CF.CG_GLASS[1], CF.CG_GLASS[0]), g['nd'] if g else '', g['vd'] if g else '', '', 'sensor cover glass', ''])
        ws.append(['', 'Infinity', CF.CG_GAP, '', '', '', '', '', ''])
    ws.append(['IMG', 'Infinity', '', '', '', '', '', '', ''])
    for col, w in zip('ABCDEFGHI', (9, 13, 15, 22, 9, 8, 24, 60, 8)):
        ws.column_dimensions[col].width = w
    wb.save(out + '.xlsx')


def make(rec, out, cg=True, fno=None):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    S, nf = write_seq(rec, out, cg, fno)
    write_opt(S, out, nf, cg)
    write_xlsx(S, out, cg)
    try:
        print('EPD (real, stop-filling) = %.4f   paraxial (solver) = %.4f' % (real_epd(rec), rec['epd']))
    except Exception as ex:
        print('real EPD skipped:', ex)
    print('wrote', out + '.seq', out + '_opt.seq', out + '.xlsx')
    print('glasses:', {e: '%s:%s' % g for e, g in glasses_of(rec).items()})


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('file'); ap.add_argument('--rank', type=int, default=None); ap.add_argument('--idx', type=int, default=None)
    ap.add_argument('--out', required=True); ap.add_argument('--fno', type=float, default=None)
    ap.add_argument('--nocg', action='store_true', help='no cover glass (air-equivalent back distance)')
    a = ap.parse_args()
    make(pick(load_records(a.file), a.rank, a.idx), a.out, cg=not a.nocg, fno=a.fno)
