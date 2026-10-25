"""Regression test: the generalised solver must reproduce the original Sigma 50/1.4 CL solver to round-off.

examples/sigma50cl/regression.json holds the original solver's numbers (report, vignetted pupil, features, Jacobian
column norms, dls error function) for 6 designs under a fixed environment.  This script sets the same environment,
recomputes everything with restore/ (torch engine and, if built, the Rust engine) and prints the worst differences.
Needs the glass catalogs (HOYA/OHARA AGF + CODE V hoya.dat/ohara.dat) and glassmodel_cv_fit.json in LENS_GLASS_DIR.
usage: LENS_GLASS_DIR=<catalog dir> python restore/tests/check_sigma50.py"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
REF = json.load(open(os.path.join(ROOT, 'examples', 'sigma50cl', 'regression.json')))
os.environ.update({k: v for k, v in REF['env'].items() if v is not None})
os.environ['RESTORE_CONFIG'] = os.path.join(ROOT, 'examples', 'sigma50cl', 'config.json')
os.environ['LENS_ENGINE'] = 'torch'; os.environ.setdefault('LENS_DEVICE', 'cpu')
os.environ['RESTORE_LEGACY32'] = '1'   # emulate the original solver's float32-built constants (see core.py)
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np, torch
import objective as OB, params as P, dls

U = torch.tensor(np.array([P.phys_to_u(np.array(x)) for x in REF['x']]), dtype=OB.DT, device=OB.dev)
B = U.shape[0]
worst = {}


def cmp(name, a, b, rel=False):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = np.abs(a - b) / (np.abs(b) + 1e-30) if rel else np.abs(a - b)
    worst[name] = max(worst.get(name, 0.0), float(d.max()))


pup = OB.pupil_setup(U)
for k, (x, y) in enumerate(zip(pup, REF['pupil'])):
    cmp('pupil', x.cpu().numpy(), y)
rep = OB.report(U, pup)
for r, q in zip(rep, REF['report']):
    for k in ('loss', 'mtf_rms', 'cra', 'ri', 'efl', 'dz', 'epd', 'mtf', 'hreal'):
        cmp('report.' + k, r[k], q[k], rel=(k == 'loss'))
hook = dls.RowHook(); ids = torch.arange(B)
with torch.no_grad():
    phi = dls.phi_fn(U, pup)
cmp('phi.norm', phi.norm(dim=1).cpu().numpy(), REF['phi_norm'], rel=True)
cmp('phi.head', phi[:, :8].cpu().numpy(), REF['phi_head']); cmp('phi.tail', phi[:, -30:].cpu().numpy(), REF['phi_tail'])
p0, J = dls.jac_ad(U, pup, list(range(OB.NP)), hook, ids)
cmp('jac.colnorm', J.norm(dim=2).cpu().numpy(), REF['jac_colnorm'], rel=True)
f, c = dls.evaluate(U, pup, hook, ids)
cmp('dls.phi', (f ** 2).sum(1).cpu().numpy(), REF['dls_phi'], rel=True)
cmp('dls.viol', torch.relu(c).amax(1).cpu().numpy(), REF['dls_viol'])
try:
    import rcore
    pr = rcore.pupil(U)
    for x, y in zip(pr, REF['pupil']):
        cmp('rust.pupil', x.cpu().numpy(), y)
    fr = rcore.phi(U, pup)
    cmp('rust.phi.norm', fr.norm(dim=1).cpu().numpy(), REF['phi_norm'], rel=True)
    _, Jr = rcore.jac(U, pup, list(range(OB.NP)))
    cmp('rust.jac.colnorm', Jr.norm(dim=2).cpu().numpy(), REF['jac_colnorm'], rel=True)
except OSError as ex:
    print('Rust engine not available (%s): build it with restore/rust/build.sh' % ex)
ok = True
for k, v in worst.items():
    # the Rust pupil search brackets the vignetting edge independently: it may stop one bisection step (~4e-6 mm) away
    lim = 2e-5 if k == 'rust.pupil' else 1e-8
    flag = 'ok' if v <= lim else 'DIFFERENT'
    ok &= v <= lim
    print('%-18s worst %.2e  %s' % (k, v, flag))
print('REGRESSION', 'PASSED' if ok else 'FAILED')
sys.exit(0 if ok else 1)
