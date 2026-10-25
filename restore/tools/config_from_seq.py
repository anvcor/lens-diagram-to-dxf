"""config_from_seq.py -- draft a restore/ lens config from a CODE V start structure (.seq).

The .seq is the lens as reconstructed from the diagram / patent (scripts/diag_build.py, make_seq, or by hand): radii,
thicknesses, clear apertures, glasses (CODE V names, e.g. MTAC80_HOYA), aspheres (start coefficients, e.g. measured
with scripts/asph_from_diagram.py), a stop, optionally a cover glass (flat plates after the last lens).
Written: config.json next to --out with
  surfaces       z (cumulative), R, cir, glass_after (elements L1..Ln in order), stop, asph ("var" or fixed coefficients)
  elements       all "var" (edit: "VENDOR:NAME" fixes an element, e.g. a known ED glass)
  asph_var       c_range around the start curvature; sag_table = sag of the start asphere at h = 2, 3, ... cir (the
                 measured profile -> sag bands keep the solution on the drawing; drop it if the coefficients are guesses)
  image          cover glass (detected), back-gap range start +-2 mm, image height (YRI or --image-height), 10 fields
  aperture       FNO (from the .seq or --fno);   stop radius from a paraxial trace if the .seq has no stop CIR
  limits         EFL window = paraxial EFL +-1.5 %, cra_max 32, ri_min 0.2 (edit for the camera system)
  targets        mtf_official.json (you supply it: official MTF chart, format in restore/README.md)
  glass          vendors of the start glasses (default HOYA, OHARA), aspheric elements moulded, others polished
Check every field afterwards -- this is a draft.  Paraxial numbers use nd from the AGF catalogs (--glass-dir).
usage: python tools/config_from_seq.py start.seq --out examples/mylens/config.json --name mylens [--image-height 21.6]"""
import os, re, sys, json, math, argparse
import numpy as np

NUM = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?'


def parse_seq(path):
    txt = open(path, encoding='latin-1').read().replace('&\n', ' ')
    L = dict(title='', wl=None, ref=None, wtw=None, fno=None, epd=None, yri=None, yan=None, surf=[])
    cur = None
    for raw in txt.splitlines():
        line = raw.split('!')[0].strip()
        if not line:
            continue
        if re.match(r'^[ABCDEFGH]\s', line) and cur is not None:          # asphere coefficients "A ..; B ..; C .."
            for k, v in re.findall(r'\b([A-H])\s+(' + NUM + ')', line):
                cur['A'][ord(k) - 65] = float(v.replace('D', 'E').replace('d', 'e'))
            continue
        for cmd in line.split(';'):
            t = cmd.split()
            if not t:
                continue
            k = t[0].upper()
            if k == 'TITLE':
                L['title'] = cmd.split(None, 1)[1].strip().strip("'\"") if len(t) > 1 else ''
            elif k == 'WL':
                L['wl'] = [float(v) for v in t[1:]]
            elif k == 'REF':
                L['ref'] = int(t[1])
            elif k == 'WTW':
                L['wtw'] = [float(v) for v in t[1:]]
            elif k == 'FNO':
                L['fno'] = float(t[1])
            elif k == 'EPD':
                L['epd'] = float(t[1])
            elif k == 'YRI':
                L['yri'] = [float(v) for v in t[1:]]
            elif k == 'YAN':
                L['yan'] = [float(v) for v in t[1:]]
            elif k == 'S' or re.match(r'^S\d+$', k):
                vals = t[1:] if k == 'S' else t[1:]
                R = float(vals[0]) if vals else 0.0
                T = float(vals[1]) if len(vals) > 1 else 0.0
                cur = dict(R=R, T=T, glass=vals[2] if len(vals) > 2 else None, cir=None, stop=False, asp=False, A=[0.0] * 8)
                L['surf'].append(cur)
            elif k == 'STO' and cur is not None:
                cur['stop'] = True
            elif k == 'ASP' and cur is not None:
                cur['asp'] = True
            elif k == 'CIR' and cur is not None and len(t) > 1:
                cur['cir'] = float(t[1])
            elif k in ('SI', 'GO'):
                cur = None
    return L


def load_catalogs(gdir, vendors):
    out = {}
    for v in vendors:
        p = os.path.join(gdir, v + '.AGF')
        if not os.path.exists(p):
            continue
        raw = open(p, 'rb').read()
        t = raw.decode('utf-16') if raw[:2] in (b'\xff\xfe', b'\xfe\xff') else raw.decode('latin-1')
        for l in t.splitlines():
            if l.startswith('NM '):
                q = l.split()
                out[(v, q[1].replace('-', '').replace(' ', '').upper())] = dict(name=q[1], nd=float(q[4]), vd=float(q[5]))
    return out


def resolve_glass(cvname, cat):
    """'MTAC80_HOYA' -> ('HOYA', 'M-TAC80', nd); unknown -> (vendor, raw, None)"""
    if '_' not in cvname:
        return None, cvname, None
    nm, v = cvname.rsplit('_', 1)
    g = cat.get((v.upper(), nm.upper()))
    return v.upper(), (g['name'] if g else nm), (g['nd'] if g else None)


def sag(R, A, h):
    c = 0.0 if not R else 1.0 / R
    return c * h * h / (1 + math.sqrt(max(1 - c * c * h * h, 1e-12))) + sum(A[i] * h ** (4 + 2 * i) for i in range(len(A)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('seq'); ap.add_argument('--out', required=True); ap.add_argument('--name', default=None)
    ap.add_argument('--image-height', type=float, default=None); ap.add_argument('--fno', type=float, default=None)
    ap.add_argument('--glass-dir', default=os.environ.get('LENS_GLASS_DIR') or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'glass_catalog'))
    ap.add_argument('--fixed-asph', default='', help='surface names (S<k>) whose asphere stays fixed')
    a = ap.parse_args()
    L = parse_seq(a.seq)
    S = L['surf']
    if not S:
        raise SystemExit('no surfaces found in %s' % a.seq)
    vendors = sorted({s['glass'].rsplit('_', 1)[1].upper() for s in S if s['glass'] and '_' in s['glass']}) or ['HOYA', 'OHARA']
    cat = load_catalogs(a.glass_dir, vendors)
    if not cat:
        print('WARNING: no AGF catalogs in %s -- glass names are not resolved, paraxial numbers skipped' % a.glass_dir)
    # cover glass = flat plates with glass after the last curved surface
    last_curved = max(i for i, s in enumerate(S) if s['R'] or s['asp'])
    cg = None
    tail = S[last_curved + 1:]
    if tail and tail[0]['glass'] and not tail[0]['R']:
        v, nm, _ = resolve_glass(tail[0]['glass'], cat)
        cg = dict(thickness=tail[0]['T'], glass='%s:%s' % (v, nm), gap=sum(s['T'] for s in tail[1:]))
        lens = S[:last_curved + 1]
    else:
        lens = S[:last_curved + 1]
    # surfaces
    surfaces, elements, z, ne, nd_after = [], {}, 0.0, 0, []
    fixed_asph = set(a.fixed_asph.split(',')) if a.fixed_asph else set()
    asph_var = {}
    for k, s in enumerate(lens, 1):
        name = 'STO' if s['stop'] else 'S%d' % k
        d = dict(name=name, z=round(z, 4), R=s['R'] if s['R'] else 0, cir=s['cir'])
        nd = 1.0
        if s['glass']:
            ne += 1; e = 'L%d' % ne; d['glass_after'] = e
            v, nm, nd = resolve_glass(s['glass'], cat)
            elements[e] = 'var'
            d['_start_glass'] = '%s:%s' % (v, nm) if v else s['glass']
            nd = nd or 1.6
        if s['stop']:
            d['stop'] = True
        if s['asp']:
            A = s['A'][:4]
            if name in fixed_asph:
                d['asph'] = A
            else:
                d['asph'] = 'var'
                R = s['R']
                if R and abs(R) < 150:
                    c0 = 1 / R; lo, hi = sorted((0.6 * c0, 1.5 * c0))
                    av = dict(c_range=[lo, hi], random_R=sorted([round(0.92 * R, 2), round(1.08 * R, 2)]))
                else:
                    av = dict(c_range=[-1 / 40, 1 / 40])
                if any(A) and s['cir']:
                    hs = [float(h) for h in range(2, int(s['cir']) + 1)] + ([s['cir']] if s['cir'] % 1 else [])
                    av['sag_table'] = {'%g' % h: round(sag(R, A, h), 4) for h in hs}
                asph_var[name] = av
        surfaces.append(d); nd_after.append(nd)
        z += s['T']
    back = lens[-1]['T']
    missing_cir = [d['name'] for d in surfaces if d['cir'] is None and not d.get('stop')]
    # paraxial (d-line, nd): EFL, stop radius for the FNO
    fno = a.fno or L['fno']
    efl = ystop_marg = None
    if cat:
        y, u, n0 = 1.0, 0.0, 1.0
        ys = None
        for i, d in enumerate(surfaces):
            c = 1 / d['R'] if d['R'] else 0.0
            n1 = nd_after[i]
            u = (n0 * u - y * c * (n1 - n0)) / n1
            if d.get('stop'): ys = y
            if i + 1 < len(surfaces): y += (surfaces[i + 1]['z'] - d['z']) * u
            n0 = n1
        efl = -1.0 / u
        if fno and ys is not None:
            ystop_marg = abs(ys) * efl / (2 * fno)                         # marginal ray of EPD = EFL/FNO at the stop
    for d in surfaces:
        if d.get('stop') and d['cir'] is None:
            d['cir'] = round(ystop_marg, 3) if ystop_marg else None
    # element carrying each asphere (moulded glass): the glass in front of the surface (rear surface), else the glass behind it
    asph_elems = []
    for i, d in enumerate(surfaces):
        if d.get('asph') is not None:
            left = surfaces[i - 1].get('glass_after') if i > 0 else None
            e = left or d.get('glass_after')
            if e and e not in asph_elems: asph_elems.append(e)
    h_img = a.image_height or (max(L['yri']) if L['yri'] else None)
    if h_img is None:
        print('WARNING: no YRI in the .seq and no --image-height: set image.image_height / fields_mm by hand')
    cfg = dict(
        name=a.name or os.path.splitext(os.path.basename(a.seq))[0],
        title=L['title'] or a.name or os.path.basename(a.seq),
        note='drafted by tools/config_from_seq.py from %s -- check every field' % os.path.basename(a.seq),
        wavelengths_nm=L['wl'] or [656.3, 587.6, 546.1, 486.1, 435.8],
        ref_wavelength_index=(L['ref'] - 1) if L['ref'] else 2,
        weights=L['wtw'] or [1.0] * len(L['wl'] or [0] * 5),
        surfaces=[{k: v for k, v in d.items() if not k.startswith('_')} for d in surfaces],
        start_glasses={d['glass_after']: d['_start_glass'] for d in surfaces if '_start_glass' in d},
        elements=elements,
        asph_var=asph_var,
        sag_band=dict(tol=0.035, edge_tol=0.01, hmin=3.0),
        image=dict(cover_glass=cg, back_gap_range=[round(back - 2, 2), round(back + 2, 2)],
                   back_gap_random=[round(back - 0.5, 2), round(back + 0.5, 2)], image_height=h_img,
                   fields_mm=[round(h_img * f, 2) for f in np.linspace(0, 1, 10)] if h_img else None),
        aperture=dict(fno=fno),
        targets=dict(mtf_file='mtf_official.json', mtf_key='geometric', freqs=[10, 30], w_mtf=50.0, w_spot=0.25),
        limits=dict(efl=[round(efl * 0.985, 2), round(efl * 1.015, 2)] if efl else None, cra_max=32.0, ri_min=0.2, vd_max=96.0, dz_max=0.35),
        glass=dict(vendors=vendors, rules=dict({'default': 'polished'}, **{e: 'moulded' for e in asph_elems}), modes={'default': {}}),
        codev=dict(asph_R_range=[30, 500], rdy_tol=0.05, notes=['start structure: %s' % os.path.basename(a.seq)]))
    if not cg:
        cfg['image'].pop('cover_glass')
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(cfg, open(a.out, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print('wrote %s: %d surfaces, %d elements, %d variable aspheres%s' % (a.out, len(surfaces), len(elements), len(asph_var),
                                                                          ', cover glass %s' % cg['glass'] if cg else ''))
    if efl: print('paraxial EFL (nd) %.3f -> limits.efl %s; stop radius %s (paraxial estimate from FNO when the .seq has no stop CIR:'
                  ' use the iris radius of the drawing if known)' % (efl, cfg['limits']['efl'], next(d['cir'] for d in surfaces if d.get('stop'))))
    if missing_cir: print('WARNING: no CIR for %s -- set the clear semi-apertures' % ', '.join(missing_cir))
    print('TODO: put the official MTF chart in %s (format: restore/README.md), check rules/limits/fields' %
          os.path.join(os.path.dirname(os.path.abspath(a.out)), 'mtf_official.json'))


if __name__ == '__main__':
    main()
