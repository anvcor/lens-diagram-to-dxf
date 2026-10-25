"""Lens definition for the restoration solver (restore/).

Every module reads the lens from ONE json file: env RESTORE_CONFIG=<config.json>, or `--config <config.json>` on any
command line (it is then exported so subprocesses inherit it).  Schema: restore/README.md; complete example:
examples/sigma50cl/config.json.

Runtime switches (environment, override the file):
  LENS_EFL='lo,hi'  LENS_CRA_MAX  LENS_RI_MIN  LENS_FNO  LENS_WW='w1,w2,..'  T16BAND='lo,hi' (back gap)
  GLASS_MODE (key of glass.modes)  LENS_ENGINE (rust|torch)  LENS_DEVICE (cpu|cuda)
Variable layout (u in the optimiser, x in result files), for NE variable glasses and NA variable aspheres:
  x = [nd, vd] * NE  +  [c, A4/1e-5, A6/1e-8, A8/1e-11] * NA  +  [back gap]           (NP = 2 NE + 4 NA + 1)"""
import json, os, sys, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _find():
    p = os.environ.get('RESTORE_CONFIG')
    for i, a in enumerate(sys.argv):
        if a == '--config' and i + 1 < len(sys.argv):
            p = sys.argv[i + 1]
    if not p:
        raise SystemExit('restore: no lens config -- set RESTORE_CONFIG or pass --config <config.json>')
    p = os.path.abspath(p)
    os.environ['RESTORE_CONFIG'] = p
    return p


PATH = _find()
C = json.load(open(PATH, encoding='utf-8'))
BASE = os.path.dirname(PATH)


def path(p):
    return p if os.path.isabs(p) else os.path.join(BASE, p)


def _env_pair(name, default):
    v = os.environ.get(name)
    return tuple(map(float, v.split(','))) if v else tuple(map(float, default))


NAME = C['name']
TITLE = C.get('title', NAME)
NOTE = C.get('note', '')

# ---- wavelengths
WL_NM = [float(w) for w in C.get('wavelengths_nm', [656.3, 587.6, 546.1, 486.1, 435.8])]
WL = np.array(WL_NM) / 1000.0
NW = len(WL)
IREF = int(C.get('ref_wavelength_index', 2))
WW_RAW = np.array([float(v) for v in os.environ['LENS_WW'].split(',')]) if os.environ.get('LENS_WW') \
    else np.array(C.get('weights', [1.0] * NW), float)
WW = WW_RAW / WW_RAW.sum()

# ---- surfaces (vertex z in mm, any origin; R = 0 or null: plane; cir = clear semi-aperture)
SURF = C['surfaces']
ORDER = [s['name'] for s in SURF]
NS = len(ORDER)
Z = {s['name']: float(s['z']) for s in SURF}
ZS = [Z[s] for s in ORDER]
R0 = {s['name']: (float(s['R']) if s.get('R') else 0.0) for s in SURF}
CIR = {s['name']: float(s['cir']) for s in SURF}
AFTER = {s['name']: s['glass_after'] for s in SURF if s.get('glass_after')}
STOP = next(s['name'] for s in SURF if s.get('stop'))
ISTOP = ORDER.index(STOP)
ASPH_FIXED = {s['name']: [float(a) for a in (list(s['asph']) + [0.0] * 4)[:4]] for s in SURF if isinstance(s.get('asph'), list)}
ASPH_VAR = [s['name'] for s in SURF if s.get('asph') == 'var']
NEWTON = [k for k, s in enumerate(SURF) if s.get('asph') is not None]     # aspheres: Newton intersection
SOLVER_ONLY = {s['name'] for s in SURF if s.get('solver_only')}            # not written to CODE V (e.g. a baffle plane)
LAST = ORDER[-1]

# ---- elements: 'var' (glass to identify) or 'VENDOR:NAME' (fixed, e.g. a known FLD/ED element)
ELEMENTS = C['elements']
ELEM_ORDER = []
for _s in ORDER:
    if _s in AFTER and AFTER[_s] not in ELEM_ORDER:
        ELEM_ORDER.append(AFTER[_s])
ELEMS = [e for e in ELEM_ORDER if ELEMENTS.get(e, 'var') == 'var']
FIXED = {e: tuple(v.split(':', 1)) for e, v in ELEMENTS.items() if v != 'var'}
NE, NA = len(ELEMS), len(ASPH_VAR)
NP = 2 * NE + 4 * NA + 1
IA = {s: 2 * NE + 4 * j for j, s in enumerate(ASPH_VAR)}                    # x index of the curvature of asphere s
IT = NP - 1                                                                  # x index of the back gap
SHAPE_COLS = list(range(2 * NE, NP))                                          # aspheres + back gap
ASC = np.array([1e-5, 1e-8, 1e-11])                                           # x scaling of A4, A6, A8

AV = C.get('asph_var', {})
C_RANGE = {s: tuple(float(v) for v in AV[s]['c_range']) for s in ASPH_VAR}
_SB = C.get('sag_band', {})


def _bands(tab):
    tol = float(_SB.get('tol', 0.035)); et = float(_SB.get('edge_tol', tol)); hmin = float(_SB.get('hmin', 0.0))
    items = sorted((float(k), float(v)) for k, v in tab.items())
    hmax = items[-1][0]
    return {h: (z - (et if h == hmax else tol), z + (et if h == hmax else tol)) for h, z in items if h >= hmin}


BANDS = {s: _bands(AV[s]['sag_table']) for s in ASPH_VAR if AV.get(s, {}).get('sag_table')}

# ---- image side
IMG = C['image']
_CG = IMG.get('cover_glass') or {}
CG_T = float(_CG.get('thickness', 0.0))
CG_GAP = float(_CG.get('gap', 0.0))
CG_GLASS = tuple(_CG['glass'].split(':', 1)) if _CG.get('glass') else None
T_LO, T_HI = _env_pair('T16BAND', IMG['back_gap_range'])
T_RANDOM = tuple(IMG.get('back_gap_random', [T_LO + 0.3 * (T_HI - T_LO), T_LO + 0.6 * (T_HI - T_LO)]))
H_CORNER = float(IMG['image_height'])
FIELDS_MM = [float(h) for h in IMG['fields_mm']]
FNO = float(os.environ.get('LENS_FNO', C['aperture']['fno']))

# ---- targets
_T = C['targets']
MTF_FILE = path(_T['mtf_file'])
MTF_KEY = _T.get('mtf_key', 'geometric')
FREQS = [float(f) for f in _T.get('freqs', [10, 30])]
W_MTF = float(_T.get('w_mtf', 50.0))
W_SPOT = float(_T.get('w_spot', 0.25))

# ---- limits
_L = C['limits']
EFL_LO, EFL_HI = _env_pair('LENS_EFL', _L['efl'])
EFL_NOM = 0.5 * (EFL_LO + EFL_HI)
CRA_MAX = float(os.environ.get('LENS_CRA_MAX', _L.get('cra_max', 90.0)))
RI_MIN = float(os.environ.get('LENS_RI_MIN', _L.get('ri_min', 0.0)))
VD_MAX = float(_L.get('vd_max', 96.0))
DZ_MAX = float(_L.get('dz_max', 0.35))
_th = np.degrees(np.arctan(H_CORNER / EFL_NOM))
CRA_ANG_DEG = [float(a) for a in (_L.get('cra_angles_deg') or (_th + np.array([-3.5, -2.5, -1.5, -0.5, 0.5])).tolist())]

# ---- glass rules: per element 'moulded' (HOYA M-, OHARA L-, ...), 'polished', or 'any'; modes override them
_G = C.get('glass', {})
VENDORS = list(_G.get('vendors', ['HOYA', 'OHARA']))
MODES = _G.get('modes', {'default': {}})
MODE = os.environ.get('GLASS_MODE', next(iter(MODES)))
_RULES = dict(_G.get('rules', {}))


def rule(e, mode=None):
    m = MODES.get(MODE if mode is None else mode, {})
    return m.get(e, _RULES.get(e, _RULES.get('default', 'any')))


MOULDED_PREFIX = _G.get('moulded_prefix', {'HOYA': ['M-'], 'OHARA': ['L-'], 'CDGM': ['D-'], 'SCHOTT': ['P-'], 'SUMITA': ['K-P']})
GLASS_DIR = os.environ.get('LENS_GLASS_DIR') or (path(_G['catalog_dir']) if _G.get('catalog_dir') else os.path.join(HERE, 'glass_catalog'))

CODEV = C.get('codev', {})

# ---- engine / device defaults: the Rust engine (CPU, all cores) when its library is present
_DLL = os.environ.get('LENSCORE_DLL', os.path.join(HERE, 'lenscore.dll' if os.name == 'nt' else 'liblenscore.so'))
ENGINE = os.environ.get('LENS_ENGINE') or ('rust' if os.path.exists(_DLL) else 'torch')
os.environ.setdefault('LENS_ENGINE', ENGINE)
if ENGINE == 'rust':
    os.environ.setdefault('LENS_DEVICE', 'cpu')
