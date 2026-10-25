"""cvtools.py -- CODE V side of the delivery (Windows + CODE V + pywin32; uses scripts/cv_session.py).

  finalize <dir> <stem>   load <stem>.seq, vignetting with the CODE V macro setvig.seq, check reference rays R1..R5 of
                          every field, CRA per field, relative illumination (Relillum.seq), save <stem>.len, rewrite
                          <stem>.seq with the vignetting factors, ray paths for the layout -> <stem>_finalize.json
  mtf <lens.len|seq>...   CODE V MTF_1FLD (GEO and DIF, 64 grid) at the config's real image heights (YRI, SET VIG) vs
                          the official curves: mean / max |dev|  (GEO vs 'geometric', DIF vs 'diffraction' if present)
  rank --tag T --files f1.jsonl ... [--n 6]
                          CODE V re-ranking of solver candidates: the top-n feasible designs (distinct glass sets,
                          ranked by the solver's mtf_rms) -> deliver.make -> finalize -> mtf; pick = lowest CODE V GEO
                          mean |dev| with no ray failures, CRA < cra_max, RI >= ri_deliver, EFL in the window
  layout <dir> <stem>     layout PNG (lens profiles + vignetted R1/R2/R3 rays) from finalize output
CODE V macros: env CODEV_MACRO_DIR or config codev.macro_dir (folder with setvig.seq and Relillum.seq); otherwise
<drive>:/CODEV*/macro is searched."""
import os, sys, re, json, glob, argparse
import numpy as np
import config as CF

sys.path.insert(0, os.path.join(os.path.dirname(CF.HERE), 'scripts'))
CV = CF.CODEV


def CodeV(workdir):
    from cv_session import CodeV as _C
    return _C(workdir=workdir)


def macro_dir():
    d = os.environ.get('CODEV_MACRO_DIR') or CV.get('macro_dir')
    if d and os.path.exists(os.path.join(d, 'setvig.seq')):
        return d
    for p in sorted((q for dr in 'CDEFGH' for q in glob.glob(dr + ':/CODEV*/macro')), reverse=True):
        if os.path.exists(os.path.join(p, 'setvig.seq')):
            return p
    raise SystemExit('CODE V macro folder with setvig.seq not found: set CODEV_MACRO_DIR')


# ------------------------------------------------------------------------------------------------ finalize
def finalize(D, stem, verbose=True):
    M = macro_dir()
    seq = open(os.path.join(D, stem + '.seq')).read()
    with CodeV(D) as cv:
        out = cv.cmd('in "%s.seq"' % stem)
        errs = [l for l in out.splitlines() if 'rror' in l]
        if errs: print('load errors:', errs[:5])
        cv.cmd('run "%s" %s ;GO' % (os.path.join(M, 'setvig.seq'), CV.get('setvig_args', '1e-07 0.1 100 NO YES')))
        nf = int(cv.eva('(NUM F)'))
        rep = {'fields': []}
        for f in range(1, nf + 1):
            row = dict(F=f, YAN=cv.eva('(YRI F%d)' % f) if 'YRI' in seq else cv.eva('(YAN F%d)' % f),
                       VUY=cv.eva('(VUY F%d)' % f), VLY=cv.eva('(VLY F%d)' % f), VUX=cv.eva('(VUX F%d)' % f), VLX=cv.eva('(VLX F%d)' % f),
                       yreal=cv.eva('(Y R1 SI F%d)' % f), cra=cv.eva('(AOI R1 SI F%d)' % f))
            fails = []
            for r in range(1, 6):
                o = cv.cmd('EVA (Y R%d SI F%d)' % (r, f))
                if 'rror' in o or 'fail' in o.lower(): fails.append(r)
            row['ray_failures'] = fails
            rep['fields'].append(row)
        rep['EPD'] = cv.eva('(EPD)'); rep['EFL'] = cv.eva('(EFL)'); rep['FNO'] = cv.eva('(FNO)')
        o = cv.cmd('in "%s" %s' % (os.path.join(M, 'Relillum.seq'), CV.get('relillum_args', '11 48 1.25 0')))
        ri = re.findall(r'^\s*([0-9.]+)\s+([0-9.]+)\s*$', o, re.M)
        rep['RI'] = [(float(a), float(b) / 100) for a, b in ri]
        ns = int(cv.eva('(NUM S)'))
        zv = [0.0]
        for s in range(1, ns + 1):
            zv.append(zv[-1] + (cv.eva('(THI S%d)' % (s - 1)) if s > 1 else 0.0))
        paths = {}
        for f in range(1, nf + 1):
            for r in (1, 2, 3):
                pts = []
                for s in range(1, ns + 2):
                    sq = 'S%d' % s if s <= ns else 'SI'
                    y = cv.eva('(Y R%d %s F%d)' % (r, sq, f)); z = cv.eva('(Z R%d %s F%d)' % (r, sq, f))
                    zz = (zv[s] if s <= ns else zv[ns] + cv.eva('(THI S%d)' % ns)) + (z or 0.0)
                    pts.append((zz, y))
                paths['F%d_R%d' % (f, r)] = pts
        rep['paths'] = paths
        cv.cmd('SAV "%s.len"' % stem); cv.cmd('WRL "%s.seq"' % stem)
    json.dump(rep, open(os.path.join(D, stem + '_finalize.json'), 'w'), indent=1)
    if verbose:
        print('EPD %.4f  EFL %.4f  FNO %.4f  RI(edge) %s' % (rep['EPD'], rep['EFL'], rep['FNO'], rep['RI'][-1] if rep['RI'] else 'n/a'))
        for r in rep['fields']:
            print('F%d field %7.3f  y_real %7.3f  CRA %6.2f  VUY %.4f VLY %.4f VUX %.4f VLX %.4f  failures %s' % (
                r['F'], r['YAN'], r['yreal'], r['cra'], r['VUY'], r['VLY'], r['VUX'], r['VLX'], r['ray_failures'] or '-'))
    return rep


# ------------------------------------------------------------------------------------------------ MTF vs official
OFF = json.load(open(CF.MTF_FILE))
H = CF.FIELDS_MM
KEYS = ['%g%s' % (f, d) for f in CF.FREQS for d in ('S', 'M')]       # S = azimuth 0, M = azimuth 90 (CODE V MTF_1FLD)


def mtf_rows(cv, mode):
    cv.cmd('YRI ' + ' '.join('%g' % h for h in H)); cv.cmd('WTF ' + ' '.join('1' for _ in H))
    cv.cmd('XRI ' + ' '.join('0' for _ in H))
    for v in ('VUY', 'VLY', 'VUX', 'VLX'):
        cv.cmd(v + ' ' + ' '.join('0' for _ in H))
    cv.cmd('SET VIG'); cv.cmd('GBL NUM ^M(6)')
    rows = []
    for i, h in enumerate(H, 1):
        r = [h]
        for fr in CF.FREQS:
            for az in (0, 90):
                cv.cmd("EVA MTF_1FLD(1,%d,%g,%d,64,^M,'%s','SIN')" % (i, fr, az, mode))
                r.append(cv.eva('(^M(1))'))
        r.append(cv.eva('(Y R1 F%d SI)' % i))
        rows.append(r)
    return rows


def deviation(rows, ref):
    R = np.array(rows, float); y = np.abs(R[:, -1])
    return np.stack([R[:, 1 + j] - np.interp(y, OFF[ref][k]['h'], OFF[ref][k]['mtf']) for j, k in enumerate(KEYS)], 1)


def mtf_eval(cv, lens):
    res = {}
    for mode, ref in (('GEO', 'geometric'), ('DIF', 'diffraction')):
        if ref not in OFF: continue
        cv.cmd(('in "%s"' if lens.endswith('.seq') else 'RES "%s"') % lens)     # fresh copy; the file is not changed
        if 'EFL' not in res:
            res['EFL'] = cv.eva('(EFL)'); res['FNO'] = cv.eva('(FNO)')
        rows = mtf_rows(cv, mode); Dv = deviation(rows, ref)
        res[mode] = float(np.abs(Dv).mean()); res[mode + '_max'] = float(np.abs(Dv).max()); res[mode + '_rows'] = rows
    return res


def cmd_mtf(a):
    with CodeV(os.path.dirname(os.path.abspath(a.lenses[0]))) as cv:
        for f in a.lenses:
            r = mtf_eval(cv, os.path.abspath(f))
            print('%-40s EFL %.3f FNO %.3f  GEO mean|dev| %s  DIF mean|dev| %s' % (
                os.path.basename(f), r['EFL'], r['FNO'], '%.4f (max %.3f)' % (r['GEO'], r['GEO_max']) if 'GEO' in r else '-',
                '%.4f (max %.3f)' % (r['DIF'], r['DIF_max']) if 'DIF' in r else '-'), flush=True)
            json.dump(r, open(os.path.splitext(f)[0] + '_mtf.json', 'w'), indent=1)


# ------------------------------------------------------------------------------------------------ rank
def cmd_rank(a):
    import deliver as DV
    import snap as SN
    out = os.path.abspath(a.outdir); os.makedirs(out, exist_ok=True)
    R = []
    for f in a.files:
        if os.path.exists(f): R += [json.loads(l) for l in open(f)]
    R = [r for r in R if SN.feasible(r) and CF.EFL_LO <= r['efl'] <= CF.EFL_HI]
    R.sort(key=lambda r: r['mtf_rms'])
    seen, C = set(), []
    for r in R:
        sig = tuple(r['glasses'][e] for e in CF.ELEMS)
        if sig in seen: continue
        seen.add(sig); C.append(r)
        if len(C) >= a.n: break
    res = []
    for i, r in enumerate(C):
        stem = '%s_c%d' % (a.tag, i)
        DV.make(r, os.path.join(out, stem))
        F = finalize(out, stem, verbose=False)
        res.append(dict(stem=stem, solver_mtf=r['mtf_rms'], glasses=[r['glasses'][e].split(':')[1] for e in CF.ELEMS],
                        cra=F['fields'][-1]['cra'], ri=F['RI'][-1][1] if F['RI'] else float('nan'), efl=F['EFL'],
                        failures=sum(len(f['ray_failures']) for f in F['fields'])))
    ri_min = float(CV.get('ri_deliver', 0.2))
    with CodeV(out) as cv:
        for x in res:
            x.update(mtf_eval(cv, os.path.join(out, x['stem'] + '.len')))
            x['ok'] = x['failures'] == 0 and x['cra'] < CF.CRA_MAX and x['ri'] >= ri_min and CF.EFL_LO <= x['efl'] <= CF.EFL_HI
    res.sort(key=lambda x: (not x['ok'], x.get('GEO', x.get('DIF', 9))))
    json.dump(res, open(os.path.join(out, '%s_cvrank.json' % a.tag), 'w'), indent=1)
    print('%-12s %-7s %-7s %-7s %-6s %-6s %-6s %s' % ('lens', 'solver', 'CV GEO', 'CV DIF', 'CRA', 'RI', 'ok', 'glasses'))
    for x in res:
        print('%-12s %.4f  %.4f  %.4f  %.2f  %.3f  %-5s %s' % (x['stem'], x['solver_mtf'], x.get('GEO', float('nan')), x.get('DIF', float('nan')),
                                                             x['cra'], x['ri'], x['ok'], ' / '.join(x['glasses'])))
    print('pick:', res[0]['stem'] if res and res[0]['ok'] else 'none')


# ------------------------------------------------------------------------------------------------ layout
def layout(D, stem):
    import matplotlib
    matplotlib.use('Agg'); import matplotlib.pyplot as plt
    rep = json.load(open(os.path.join(D, stem + '_finalize.json')))
    txt = open(os.path.join(D, stem + '.seq')).read().replace('&\n', ' ')
    surf = []; cur = None
    for l in txt.splitlines():
        t = l.split()
        if not t: continue
        if t[0] == 'S' and len(t) >= 3:
            cur = dict(R=float(t[1]), T=float(t[2]), glass=(t[3] if len(t) > 3 else ''), A=[0, 0, 0, 0], cir=None); surf.append(cur)
        elif cur is not None and t[0] == 'CIR': cur['cir'] = float(t[1])
        elif cur is not None and t[0] == 'A':
            nums = [float(v) for v in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', re.sub(r'\b[ABCD]\b', ' ', l.replace(';', ' ')))]
            cur['A'] = (nums + [0, 0, 0, 0])[:4]

    def sag(s, h):
        c = 0 if s['R'] == 0 else 1 / s['R']
        return c * h * h / (1 + np.sqrt(max(1 - c * c * h * h, 0))) + sum(s['A'][k] * h ** (4 + 2 * k) for k in range(4))
    z = 0.0; zs = []
    for s in surf:
        zs.append(z); z += s['T']
    fig, ax = plt.subplots(figsize=(13, 6))
    for i, s in enumerate(surf):
        h = s['cir'] or 0
        if h:
            hh = np.linspace(-h, h, 120)
            ax.plot(zs[i] + np.array([sag(s, v) for v in hh]), hh, 'k', lw=0.9)
        if s['glass'] and i + 1 < len(surf) and h:
            h2 = min(h, surf[i + 1]['cir'] or h)
            for sg in (1, -1):
                ax.plot([zs[i] + sag(s, h2), zs[i + 1] + sag(surf[i + 1], h2)], [sg * h2, sg * h2], 'k', lw=0.6)
    cols = ['#d62728', '#2ca02c', '#1f77b4', '#9467bd', '#8c564b', '#e377c2']
    for k, f in enumerate(rep['fields']):
        for r in (1, 2, 3):
            p = np.array(rep['paths']['F%d_R%d' % (f['F'], r)], float)
            ax.plot(p[:, 0], p[:, 1], color=cols[k % 6], lw=0.8)
    ax.axhline(0, color='gray', lw=0.5, ls='-.')
    ax.set_aspect('equal'); ax.set_xlabel('z (mm)'); ax.set_ylabel('y (mm)')
    ax.set_title('%s  EPD %.3f  F/%.3f  EFL %.2f   fields: real h = %s mm   CRA(F%d) = %.2f deg' % (
        stem, rep['EPD'], rep['FNO'], rep['EFL'], ', '.join('%.1f' % f['yreal'] for f in rep['fields']), len(rep['fields']), rep['fields'][-1]['cra']), fontsize=9)
    fig.tight_layout(); p = os.path.join(D, stem + '_layout.png'); fig.savefig(p, dpi=120)
    print('saved', p)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('finalize'); p.add_argument('dir'); p.add_argument('stem')
    p = sub.add_parser('mtf'); p.add_argument('lenses', nargs='+')
    p = sub.add_parser('rank'); p.add_argument('--tag', required=True); p.add_argument('--files', nargs='+', required=True)
    p.add_argument('--n', type=int, default=6); p.add_argument('--outdir', default='cv')
    p = sub.add_parser('layout'); p.add_argument('dir'); p.add_argument('stem')
    a = ap.parse_args()
    if a.cmd == 'finalize':
        finalize(os.path.abspath(a.dir), a.stem)
    elif a.cmd == 'mtf':
        cmd_mtf(a)
    elif a.cmd == 'rank':
        cmd_rank(a)
    else:
        layout(os.path.abspath(a.dir), a.stem)
