"""glassmodel.py -- fit CODE V's fictitious-glass dispersion (GLA 'nd:vd' glass) for the solver (LENS_GLASSMODEL=codev).

CODE V is used as a black box: for a grid of (nd, vd) a one-surface lens gets the fictitious glass 'nd:vd' and the
indices at the config's wavelengths are read back (IND S1 Wj).  Normalised shape s_j = (n_j - nd) / dN, dN = (nd-1)/vd,
is fitted with one 2-D polynomial per wavelength in x = (nd - 1.75)/0.3, y = (vd - 45)/30 (degree chosen on held-out
points; CODE V's model turned out to be an exact degree-5 polynomial).  The fit is only valid for the wavelengths it
was sampled at; glass.py refuses a file fitted at other wavelengths.
  python tools/glassmodel.py sample            (Windows + CODE V)  -> <glass dir>/glassmodel_cv.json
  python tools/glassmodel.py fit [--deg 5]                          -> <glass dir>/glassmodel_cv_fit.json
Without the fit file the solver uses its own two-glass normal-dispersion model (LENS_GLASSMODEL=schott2)."""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('LENS_GLASSMODEL', 'schott2')         # the fit file may not exist yet
import config as CF

GD = CF.GLASS_DIR
ND = [round(1.44 + 0.02 * i, 2) for i in range(32)]        # 1.44 .. 2.06
VD = [16 + 2 * i for i in range(30)]                        # 16 .. 74


def sample(a):
    sys.path.insert(0, os.path.join(os.path.dirname(CF.HERE), 'scripts'))
    from cv_session import CodeV
    os.makedirs(GD, exist_ok=True)
    seq = os.path.join(GD, '_glassmodel_probe.seq')
    open(seq, 'w').write('\n'.join(['RDM;LEN', "TITLE 'fictitious glass probe'", 'DIM M',
                                     'WL ' + ' '.join('%g' % w for w in CF.WL_NM), 'REF %d' % (CF.IREF + 1),
                                     'EPD 10', 'SO 0 1e10', 'S 50 5', 'S 0 50', 'SI 0 0', 'GO']) + '\n')
    rows = []
    with CodeV(workdir=GD) as cv:
        cv.cmd('in "%s"' % seq)
        wl = [cv.eva('(WL W%d)' % j) for j in range(1, len(CF.WL_NM) + 1)]
        for nd in ND:
            for vd in VD:
                if vd > 95 - 40 * (nd - 1.45) or vd < 60 - 40 * (nd - 1.45) - 25:   # roughly where real glasses are
                    continue
                code = '%.6f:%.4f' % (nd, vd)               # 'index:V' form (the xxx.yyy code cannot express nd >= 2)
                if 'rror' in cv.cmd('GLA S1 %s' % code):
                    continue
                rows.append(dict(nd=nd, vd=vd, code=code, cv=[cv.eva('(IND S1 W%d)' % j) for j in range(1, len(wl) + 1)]))
    os.remove(seq)
    json.dump(dict(wl=wl, rows=rows), open(os.path.join(GD, 'glassmodel_cv.json'), 'w'), indent=1)
    print('%d samples at %s nm -> %s' % (len(rows), wl, os.path.join(GD, 'glassmodel_cv.json')))


def terms(x, y, deg):
    return [x ** i * y ** (k - i) for k in range(deg + 1) for i in range(k + 1)]


def design(nd, vd, deg):
    return np.stack(terms((nd - 1.75) / 0.3, (vd - 45) / 30, deg), -1)


def fit(a):
    J = json.load(open(os.path.join(GD, 'glassmodel_cv.json')))
    iref = int(np.argmin([abs(w - 587.6) for w in J['wl']]))
    R = [r for r in J['rows'] if abs(r['cv'][iref] - r['nd']) < 1e-5]
    nd = np.array([r['nd'] for r in R]); vd = np.array([r['vd'] for r in R]); N = np.array([r['cv'] for r in R])
    dN = (nd - 1) / vd
    S = (N - nd[:, None]) / dN[:, None]

    def fit_deg(deg, idx):
        C, *_ = np.linalg.lstsq(design(nd[idx], vd[idx], deg), S[idx], rcond=None)
        return C
    rng = np.random.default_rng(0); perm = rng.permutation(len(R)); test = perm[:len(R) // 5]; train = perm[len(R) // 5:]
    best = None
    for deg in ([a.deg] if a.deg else (2, 3, 4, 5, 6)):
        C = fit_deg(deg, train)
        err = (design(nd[test], vd[test], deg) @ C - S[test]) * dN[test, None]
        print('deg %d  held-out index error rms %s  max %s' % (deg, ' '.join('%.1e' % v for v in np.sqrt((err ** 2).mean(0))),
                                                               ' '.join('%.1e' % v for v in np.abs(err).max(0))))
        if best is None or np.abs(err).max() < best[1]: best = (deg, np.abs(err).max())
    deg = best[0]
    C = fit_deg(deg, np.arange(len(R)))
    out = os.path.join(GD, 'glassmodel_cv_fit.json')
    json.dump(dict(deg=deg, x='(nd-1.75)/0.3', y='(vd-45)/30', wl_nm=J['wl'], coef=C.tolist(),
                   note='s_j = sum_k coef[k][j] * term_k(x, y), term order: k = total degree, i = power of x; n_j = nd + (nd-1)/vd * s_j'),
              open(out, 'w'), indent=1)
    print('chosen degree %d -> %s' % (deg, out))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('sample')
    p = sub.add_parser('fit'); p.add_argument('--deg', type=int, default=None)
    a = ap.parse_args()
    (sample if a.cmd == 'sample' else fit)(a)
