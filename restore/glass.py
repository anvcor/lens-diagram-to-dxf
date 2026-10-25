"""Glass catalogs (Zemax AGF), availability filter, and the fictitious (model) glass of the optimiser.

Catalog directory: env LENS_GLASS_DIR, config glass.catalog_dir, or restore/glass_catalog/ (not in the repository:
copy <VENDOR>.AGF there; optionally CODE V's hoya.dat / ohara.dat to drop discontinued glasses).
Model glass n(lambda) for a free (nd, vd):
  codev   -- CODE V's own GLC fictitious glass: a degree-5 polynomial shape in (nd, vd) fitted from CODE V samples
             (tools/glassmodel_cv.py + tools/glassmodel_fit.py write glassmodel_cv_fit.json; needs CODE V once)
  schott2 -- shape interpolated between a crown (BSC7 / N-BK7) and a flint (E-F2 / F2) by vd (no CODE V needed)
env LENS_GLASSMODEL picks one; default codev when glassmodel_cv_fit.json exists."""
import json, os, numpy as np
import config as CF

WL = CF.WL
WW = CF.WW
WW_RAW = CF.WW_RAW
GDIR = CF.GLASS_DIR


def _dec(raw):
    return raw.decode('utf-16') if raw[:2] in (b'\xff\xfe', b'\xfe\xff') else raw.decode('latin-1')


def load(v):
    t = _dec(open(os.path.join(GDIR, v + '.AGF'), 'rb').read()).splitlines(); out = {}
    cur = None
    for l in t:
        p = l.split()
        if l.startswith('NM '):
            cur = dict(name=p[1], form=int(float(p[2])), nd=float(p[4]), vd=float(p[5]), vendor=v,
                       status=int(float(p[7])) if len(p) > 7 else 0); out[p[1]] = cur
        elif l.startswith('CD ') and cur is not None:
            cur['cd'] = [float(x) for x in p[1:]]
    return out


def nidx(g, lam):
    lam = np.asarray(lam, float); c = g['cd']; L2 = lam * lam
    if g['form'] == 1:      # Schott
        n2 = c[0] + c[1] * L2 + c[2] / L2 + c[3] / L2 ** 2 + c[4] / L2 ** 3 + c[5] / L2 ** 4
    elif g['form'] == 2:    # Sellmeier 1
        n2 = 1 + c[0] * L2 / (L2 - c[1]) + c[2] * L2 / (L2 - c[3]) + c[4] * L2 / (L2 - c[5])
    else:
        raise ValueError(g['form'])
    return np.sqrt(n2)


_need = set(CF.VENDORS) | {v for v, _ in CF.FIXED.values()} | ({CF.CG_GLASS[0]} if CF.CG_GLASS else set())
CAT = {}
for _v in sorted(_need):
    try:
        CAT.update({(_v, k): g for k, g in load(_v).items()})
    except Exception as ex:
        print('glass catalog %s not loaded (%s): put %s.AGF in %s' % (_v, ex, _v, GDIR))
CAT_ALL = dict(CAT)

# ---- availability (CODE V <vendor>.dat): HOYA 0/1/2 in production; OHARA 21..24 (8 = old lead Y-series etc.)
AVAIL = {}
for _v in ('HOYA', 'OHARA'):
    _f = os.path.join(GDIR, _v.lower() + '.dat')
    if not os.path.exists(_f):
        continue
    for _i, _l in enumerate(open(_f, encoding='latin-1').read().splitlines()):
        _p = _l.split('\t')
        if _i == 0 or len(_p) < 8:
            continue
        try:
            AVAIL[(_v, _p[1].strip())] = int(_p[7])
        except ValueError:
            pass
OK_CODES = {'HOYA': {0, 1, 2}, 'OHARA': {21, 22, 23, 24}}
KEEP_ALWAYS = set(CF.FIXED.values()) | ({CF.CG_GLASS} if CF.CG_GLASS else set())


def available(v, name):
    if (v, name) in KEEP_ALWAYS:
        return True
    if v not in OK_CODES or not AVAIL:
        return True
    code = AVAIL.get((v, name.replace('-', '')))
    return code is not None and code in OK_CODES[v]


if AVAIL and os.environ.get('LENS_ALL_GLASSES', '') != '1':
    CAT = {k: g for k, g in CAT_ALL.items() if available(*k)}


def get(v, name):
    return CAT_ALL[(v, name)]


def is_moulded(v, name):
    return any(name.startswith(p) for p in CF.MOULDED_PREFIX.get(v, []))


# ---- model glass
def _first(keys):
    for pool in (CAT, CAT_ALL):                        # prefer an available glass
        for k in keys:
            if k in pool:
                return pool[k]
    return None


_bk = _first([('HOYA', 'BSC7'), ('SCHOTT', 'N-BK7'), ('OHARA', 'S-BSL7'), ('CDGM', 'H-K9L')])
_f2 = _first([('HOYA', 'E-F2'), ('HOYA', 'F2'), ('SCHOTT', 'F2'), ('OHARA', 'PBM2Y'), ('CDGM', 'F2')])


def _shape(g):
    n = nidx(g, WL); nd = nidx(g, 0.5876); nF = nidx(g, 0.4861); nC = nidx(g, 0.6563)
    return (n - nd) / (nF - nC)


SH_C, SH_F = (_shape(_bk), _shape(_f2)) if (_bk and _f2) else (np.zeros(len(WL)), np.zeros(len(WL)))
VD_C, VD_F = (_bk['vd'], _f2['vd']) if (_bk and _f2) else (64.0, 36.0)

_FIT = next((p for p in (os.path.join(GDIR, 'glassmodel_cv_fit.json'), os.path.join(CF.HERE, 'glassmodel_cv_fit.json')) if os.path.exists(p)), None)
GLASS_MODEL = os.environ.get('LENS_GLASSMODEL', 'codev' if _FIT else 'schott2')
GM_DEG, GM_COEF = 0, None
if GLASS_MODEL == 'codev':
    if not _FIT:
        raise SystemExit('LENS_GLASSMODEL=codev but no glassmodel_cv_fit.json (run tools/glassmodel_cv.py + glassmodel_fit.py)')
    _gf = json.load(open(_FIT))
    if [round(w, 1) for w in _gf['wl_nm']] != [round(w, 1) for w in CF.WL_NM]:
        raise SystemExit('glassmodel_cv_fit.json was fitted at %s nm, the lens uses %s nm: refit' % (_gf['wl_nm'], CF.WL_NM))
    GM_DEG = _gf['deg']; GM_COEF = np.array(_gf['coef'])


def gm_terms(nd, vd, deg):
    """polynomial terms of the codev model (numpy or torch); order: total degree k, power of x i"""
    x = (nd - 1.75) / 0.3; y = (vd - 45) / 30
    return [x ** i * y ** (k - i) for k in range(deg + 1) for i in range(k + 1)]


def model_n(nd, vd):
    """indices at WL for a fictitious glass (nd, vd) with normal partial dispersion (GLASS_MODEL)"""
    if GLASS_MODEL == 'codev':
        nd = np.asarray(nd, float); vd = np.asarray(vd, float)
        s = np.stack(np.broadcast_arrays(*gm_terms(nd, vd, GM_DEG)), -1) @ GM_COEF
        return nd[..., None] + ((nd - 1) / vd)[..., None] * s
    s = SH_C + (SH_F - SH_C) * (vd - VD_C) / (VD_F - VD_C)
    return nd + (nd - 1) / vd * s


if __name__ == '__main__':
    print('catalog dir', GDIR, '|', len(CAT_ALL), 'glasses,', len(CAT), 'available | model', GLASS_MODEL)
