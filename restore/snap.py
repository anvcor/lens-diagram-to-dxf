"""snap.py -- from fictitious (model) glasses to real catalog glasses, then shape polish and a glass-swap search.

Subcommands (each reads/writes jsonl files of objective.report() records; glass rules from the config / GLASS_MODE):
  attract  catalog attraction: the error function gets, per free element, w * soft-nearest distance of its (nd, vd)
           to the catalog glasses that element may use (normalised dnd/0.01, dvd/1; Boltzmann expectation over the
           catalog, >= the true nearest distance).  Annealed in stages (w up, tau down).  The model glass also gets
           the softmin-weighted anomalous partial dispersion of the nearby catalog glasses, so a glass that lands on a
           catalog point already has its real indices.  An element is frozen to its catalog glass as soon as it is
           within --dfreeze; the still-free glasses and the shapes keep compensating.  After the last stage the
           remaining elements are snapped one at a time (closest first), each followed by a re-optimisation.
             python snap.py attract --starts run_c.jsonl --top 16 --out run_a.jsonl
  polish   shape-only polish (aspheres + back gap) of all-catalog designs, DLS in soft-constraint mode.
             python snap.py polish run_a.jsonl --n 16 --outer 12 --lm 6 --out run_p.jsonl
  swap     one-element glass swaps (k nearest catalog neighbours) around the best designs, every trial polished,
           accepted when the loss drops by > 3 %.
             python snap.py swap run_p.jsonl --n 4 --k 3 --rounds 2 --outer 6 --final 10 --out run_s.jsonl
All optimisation steps use dls.optimize(soft=True).  Ported from the Sigma 50/1.4 CL round-3 scripts
(catalog attraction, structured polish, swap search)."""
import os, sys, json, time, argparse, math
import numpy as np, torch
import config as CF
import objective as OB
import params as P
import core as T
import glassmap as GMP
from glass import CAT, CAT_ALL, nidx, WL, model_n

dev, DT = OB.dev, OB.DT
NP, NE = CF.NP, CF.NE
ELEMS = CF.ELEMS
EIDX = {e: i for i, e in enumerate(ELEMS)}
NW = len(WL)
SHAPE = CF.SHAPE_COLS
ORIG_BUILD = T.build

# ---------------------------------------------------------------- catalog sets of the variable elements
CATS = {e: GMP.allowed(e) for e in ELEMS}
NMAX = max([len(c) for c in CATS.values()] + [1])
CND = torch.full((NE, NMAX), 9.0, dtype=DT, device=dev)        # padding far away
CVD = torch.full((NE, NMAX), 500.0, dtype=DT, device=dev)
CDEL = torch.zeros((NE, NMAX, NW), dtype=DT, device=dev)        # anomalous partial dispersion of each catalog glass
CNIDX = {}
for _i, _e in enumerate(ELEMS):
    for _k, _key in enumerate(CATS[_e]):
        _g = CAT[_key]; _n = nidx(_g, WL); CNIDX[_key] = _n
        CND[_i, _k] = _g['nd']; CVD[_i, _k] = _g['vd']
        CDEL[_i, _k] = torch.tensor(_n - model_n(_g['nd'], _g['vd']), dtype=DT)



def cat_n(key):
    """catalog indices at the system wavelengths (any catalog glass, also outside this mode's sets)"""
    if key not in CNIDX:
        CNIDX[key] = nidx(CAT_ALL[key], WL)
    return CNIDX[key]


CFG = dict(w=0.0, tau=1.0, pd=False, eps=0.02)
# per-design state: FM (B,NE) frozen mask, NF (B,NE,NW) frozen indices, rows = design index of every evaluated row,
# KEY = frozen catalog keys.  dls.RowHook finds this module through T.build.__module__ and the name 'build_mixed'.
ST = dict(rows=None, FM=None, NF=None, KEY=None)


def soft_weights(nd, vd, tau):
    """nd, vd (B,NE) -> distances d (B,NE,N), Boltzmann weights p (B,NE,N)"""
    d = torch.sqrt(((nd[..., None] - CND[None]) / 0.01) ** 2 + (vd[..., None] - CVD[None]) ** 2 + CFG['eps'] ** 2)
    return d, torch.softmax(-d / tau, -1)


def build_mixed(p):
    """core.build with frozen elements replaced by their catalog indices (+ partial-dispersion blending if CFG['pd'])"""
    surf = ORIG_BUILD(p)
    rows = ST['rows']
    FM, NF_ = ST['FM'][rows], ST['NF'][rows]
    corr = None
    if CFG['pd']:
        _, pw = soft_weights(p['nd'], p['vd'], CFG['tau'])
        corr = torch.einsum('bin,inw->biw', pw, CDEL)
    out = []
    for s, (c, A, n) in zip(CF.ORDER, surf):
        e = CF.AFTER.get(s)
        if e in EIDX:
            i = EIDX[e]
            if corr is not None:
                n = n + corr[:, i]
            n = torch.where(FM[:, i:i + 1], NF_[:, i, :], n)
        out.append((c, A, n))
    return out


def set_rows(r):
    ST['rows'] = r.to(dev) if torch.is_tensor(r) else r


def init_state(B, keys=None):
    """all free (keys None) or all frozen to keys[b][i]"""
    ST['FM'] = torch.zeros(B, NE, dtype=torch.bool, device=dev) if keys is None else torch.ones(B, NE, dtype=torch.bool, device=dev)
    ST['NF'] = torch.ones(B, NE, NW, dtype=DT, device=dev) if keys is None else \
        torch.tensor(np.array([[cat_n(k) for k in kk] for kk in keys]), dtype=DT, device=dev)
    ST['KEY'] = [[None] * NE for _ in range(B)] if keys is None else [list(kk) for kk in keys]
    set_rows(torch.arange(B))
    T.build = build_mixed


def attract_masked(u):
    ph = P.params_to_phys(u)
    d, p = soft_weights(ph['nd'], ph['vd'], CFG['tau'])
    return CFG['w'] * (p * d).sum(-1) * (~ST['FM'][ST['rows']]).to(DT)


def nearest_np(x):
    """x physical (NP,) -> [(catalog key, distance)] per element"""
    out = []
    for i, e in enumerate(ELEMS):
        nd, vd = x[2 * i], x[2 * i + 1]
        dd = [math.hypot((CAT[k]['nd'] - nd) / 0.01, CAT[k]['vd'] - vd) for k in CATS[e]]
        j = int(np.argmin(dd)); out.append((CATS[e][j], dd[j]))
    return out


def set_u_glass(u, i, key):
    g = CAT[key]
    a, b = GMP.glass_to_u(g['nd'], g['vd'], i)
    u[2 * i] = float(a); u[2 * i + 1] = float(b)


def freeze(U, b, i, key):
    ST['FM'][b, i] = True
    ST['NF'][b, i] = torch.tensor(cat_n(key), dtype=DT, device=dev)
    un = U[b].cpu().numpy().copy(); set_u_glass(un, i, key); U[b] = torch.tensor(un, dtype=DT, device=U.device)
    ST['KEY'][b][i] = key


def optimize(u, iters, cols=None, imp=0.01, tag='', log_every=5):
    """dls in soft mode; per-design mask = glass columns of frozen elements off; cols restricts the variables"""
    import dls
    B = u.shape[0]
    vm = torch.ones(B, NP, dtype=torch.bool, device=dev)
    FM = ST['FM'].to(dev)
    for i in range(NE):
        vm[:, 2 * i] = ~FM[:, i]; vm[:, 2 * i + 1] = ~FM[:, i]
    if cols is not None:
        m = torch.zeros(NP, dtype=torch.bool, device=dev); m[cols] = True; vm &= m
    set_rows(torch.arange(B, device=dev))
    if iters <= 0 or not bool(vm.any()):
        return u, OB.pupil_setup(u)
    dls.EXTRA = attract_masked if CFG['w'] > 0 else None
    try:
        r = dls.optimize(u, vm, mxc=iters, mnc=1, imp=imp, impn=2, soft=True, tag=tag, log_every=log_every)
    finally:
        dls.EXTRA = None
    set_rows(torch.arange(B, device=dev))
    return r['u'], OB.pupil_setup(r['u'])


def polish(U, outer, lm=0, tag='polish'):
    """shape-only polish: outer cycles (imp 0.005) then lm cycles (imp 0.01), all glasses frozen"""
    U, pup = optimize(U, outer, cols=SHAPE, imp=0.005, tag=tag, log_every=max(1, outer // 3))
    if lm > 0:
        U, pup = optimize(U, lm, cols=SHAPE, imp=0.01, tag=tag + '-lm', log_every=lm)
    return U, pup


def real_cost(U, pup):
    set_rows(torch.arange(U.shape[0]))
    with torch.no_grad():
        r = OB.resid(U, pup)
    return (r ** 2).sum(1)


def gdist(x, y):
    return max(math.hypot((x[2 * i] - y[2 * i]) / 0.01, x[2 * i + 1] - y[2 * i + 1]) for i in range(NE))


def load_continuous(files, top, dmin=0.3):
    """best continuous designs, dropping near-duplicates (all glasses within dmin units)"""
    R = []
    for f in files:
        if os.path.exists(f):
            R += [dict(json.loads(l), _src=os.path.basename(f)) for l in open(f)]
    R.sort(key=lambda r: r['loss'])
    keep = []
    for r in R:
        x = np.array(r['x'])
        if all(gdist(x, np.array(q['x'])) >= dmin for q in keep):
            keep.append(r)
        if len(keep) >= top: break
    return keep


def load_catalog_designs(files, n, key='mtf_rms'):
    """best all-catalog designs admissible in this glass mode, one per glass set"""
    R = []
    for f in files:
        if not os.path.exists(f): print('missing', f); continue
        R += [dict(json.loads(l), _file=os.path.basename(f)) for l in open(f)]
    R.sort(key=lambda r: r[key])
    seen, base = set(), []
    for r in R:
        sig = tuple(r['glasses'][e] for e in ELEMS)
        if any(tuple(r['glasses'][e].split(':')) not in CATS[e] for e in ELEMS): continue
        if sig in seen: continue
        seen.add(sig); base.append(r)
        if len(base) >= n: break
    return base


def catalog_U(base, keys):
    Un = np.array([P.phys_to_u(np.array(r['x'])) for r in base])
    for b in range(len(base)):
        for i, k in enumerate(keys[b]): set_u_glass(Un[b], i, k)
    return Un


def feasible(r):
    L = CF.C['limits']
    return r['cra'] <= float(L.get('cra_max', 90)) and r['ri'] >= float(L.get('ri_min', 0))


def gstr(r):
    return ' '.join(r['glasses'][e].split(':')[1] for e in ELEMS)


def finish(r, keys):
    r.pop('glass', None)
    r['mode'] = CF.MODE
    r['glasses'] = {e: '%s:%s' % keys[i] for i, e in enumerate(ELEMS)}
    r['cat_nd_vd'] = {e: [CAT[keys[i]]['nd'], CAT[keys[i]]['vd']] for i, e in enumerate(ELEMS)}
    return r


def dump(out, path):
    with open(path, 'a') as f:
        for r in out: f.write(json.dumps(r) + '\n')


def line(r):
    return 'mtf %.4f CRA %.2f RI %.3f EFL %.2f FNO %.3f dz %.2f back %.2f | %s' % (
        r['mtf_rms'], r['cra'], r['ri'], r['efl'], r['fno'], r['dz'], r['t16'], gstr(r))


# ---------------------------------------------------------------- attract
def dist_state(U):
    out = []
    for b in range(U.shape[0]):
        nn = nearest_np(P.u_to_phys_np(U[b].cpu().numpy()))
        out.append([(ST['KEY'][b][i], 0.0) if ST['FM'][b, i] else nn[i] for i in range(NE)])
    return out


def report_stage(U, pup, label):
    set_rows(torch.arange(U.shape[0]))
    rep = OB.report(U, pup)
    ds = dist_state(U)
    m = np.array([r['mtf_rms'] for r in rep]); nf = ST['FM'].sum(1).cpu().numpy()
    dfree = [d for row, fm in zip(ds, ST['FM']) for (k, d), f in zip(row, fm) if not f]
    print('[%s] mtf_rms best %.4f median %.4f | frozen per design: %s | free-element dist median %.3f max %.3f' % (
        label, m.min(), np.median(m), nf.tolist(), np.median(dfree) if dfree else 0, max(dfree) if dfree else 0), flush=True)
    return rep, ds


def cmd_attract(a):
    t0 = time.time()
    S0 = load_continuous(a.starts.split(','), a.top)
    U = torch.tensor(np.array([P.phys_to_u(np.array(r['x'])) for r in S0]), dtype=DT, device=dev)
    B = U.shape[0]
    print('mode %s | catalog sizes %s | %d starts' % (CF.MODE, {e: len(c) for e, c in CATS.items()}, B), flush=True)
    init_state(B)
    CFG.update(w=0.0, pd=False)
    pup = OB.pupil_setup(U)
    rep0, _ = report_stage(U, pup, 'start (continuous solutions)')
    log = [dict(start_mtf=rep0[b]['mtf_rms'], start_loss=rep0[b]['loss'], events=[]) for b in range(B)]
    # annealing with progressive freezing
    for si, st in enumerate(a.sched.split(',')):
        w, tau = map(float, st.split(':'))
        CFG.update(w=w, tau=tau, pd=True)
        U, pup = optimize(U, a.siters, tag='S%d w=%g tau=%g' % (si + 1, w, tau))
        rep, ds = report_stage(U, pup, 'S%d w=%g tau=%g before freeze' % (si + 1, w, tau))
        for b in range(B):
            for i in range(NE):
                if not ST['FM'][b, i] and ds[b][i][1] < a.dfreeze:
                    freeze(U, b, i, ds[b][i][0])
                    log[b]['events'].append('S%d freeze %s=%s (d=%.3f, mtf %.4f)' % (si + 1, ELEMS[i], ds[b][i][0][1], ds[b][i][1], rep[b]['mtf_rms']))
        print('   frozen after S%d: %s' % (si + 1, ST['FM'].sum(1).tolist()), flush=True)
    # greedy endgame: snap the closest free element of every design, re-optimise the rest
    w, tau = map(float, a.gw.split(':'))
    CFG.update(w=w, tau=tau, pd=True)
    step = 0
    while not bool(ST['FM'].all()):
        step += 1
        ds = dist_state(U)
        for b in range(B):
            free = [i for i in range(NE) if not ST['FM'][b, i]]
            if not free: continue
            i = min(free, key=lambda i: ds[b][i][1])
            freeze(U, b, i, ds[b][i][0])
            log[b]['events'].append('G%d snap %s=%s (d=%.3f)' % (step, ELEMS[i], ds[b][i][0][1], ds[b][i][1]))
        if bool(ST['FM'].all()): break
        U, pup = optimize(U, a.giters, tag='G%d' % step)
        report_stage(U, pup, 'G%d' % step)
    CFG.update(w=0.0, pd=False)
    set_rows(torch.arange(B)); pup = OB.pupil_setup(U)
    repS = OB.report(U, pup)
    print('[all fixed, before polish] mtf best %.4f median %.4f' % (min(r['mtf_rms'] for r in repS), np.median([r['mtf_rms'] for r in repS])), flush=True)
    U, pup = optimize(U, a.polish, cols=SHAPE, tag='polish')
    set_rows(torch.arange(B))
    repF = OB.report(U, pup)
    out = []
    for b in range(B):
        r = finish(repF[b], ST['KEY'][b])
        r.update(idx=b, src=S0[b]['_src'], mtf_rms_snap_nopolish=repS[b]['mtf_rms'],
                 hist=dict(cont=dict(mtf_rms=log[b]['start_mtf'], loss=log[b]['start_loss'], glass=S0[b].get('glass')), events=log[b]['events']))
        out.append(r)
    out.sort(key=lambda r: r['mtf_rms'])
    dump(out, a.out)
    print('\n==== attract: all-catalog designs ranked by mtf_rms ====')
    for r in out:
        print('#%2d %s  (continuous %.4f, before polish %.4f)' % (r['idx'], line(r), r['hist']['cont']['mtf_rms'], r['mtf_rms_snap_nopolish']), flush=True)
        print('      ', '; '.join(r['hist']['events']))
    print('total %.0fs' % (time.time() - t0))


# ---------------------------------------------------------------- polish
def cmd_polish(a):
    t0 = time.time()
    base = load_catalog_designs(a.files, a.n)
    B = len(base)
    keys = [[tuple(r['glasses'][e].split(':')) for e in ELEMS] for r in base]
    U = torch.tensor(catalog_U(base, keys), dtype=DT, device=dev)
    init_state(B, keys)
    CFG.update(w=0.0, pd=False)
    pup = OB.pupil_setup(U); rep0 = OB.report(U, pup)
    print('start mtf %s' % np.round([r['mtf_rms'] for r in rep0], 3).tolist())
    print('start cra %s' % np.round([r['cra'] for r in rep0], 2).tolist())
    print('start ri  %s' % np.round([r['ri'] for r in rep0], 3).tolist(), flush=True)
    U, pup = polish(U, a.outer, a.lm)
    set_rows(torch.arange(B)); rep = OB.report(U, pup)
    out = []
    for b in range(B):
        r = finish(rep[b], keys[b])
        r.update(idx=base[b].get('idx'), src=base[b]['_file'], mtf_rms_before=rep0[b]['mtf_rms'], cra_before=rep0[b]['cra'],
                 hist=base[b].get('hist'), swaps=base[b].get('swaps'))
        out.append(r)
    out.sort(key=lambda r: (not feasible(r), r['mtf_rms']))
    dump(out, a.out)
    print('\n==== polish (feasible first, then mtf_rms) ====')
    for r in out:
        print('%s %s  (before %.4f)' % ('OK ' if feasible(r) else 'bad', line(r), r['mtf_rms_before']), flush=True)
    print('total %.0fs' % (time.time() - t0))


# ---------------------------------------------------------------- swap
def neighbours(e, key, k):
    if k <= 0:                                   # k = 0: every allowed glass of that element
        return [q for q in CATS[e] if q != key]
    g = CAT[key]
    d = sorted((math.hypot((CAT[q]['nd'] - g['nd']) / 0.01, CAT[q]['vd'] - g['vd']), q) for q in CATS[e] if q != key)
    return [q for _, q in d[:k]]


def run_set(Un, G, outer, tag):
    Un = np.array(Un, copy=True)
    for b in range(len(G)):
        for i, k in enumerate(G[b]): set_u_glass(Un[b], i, k)
    U = torch.tensor(Un, dtype=DT, device=dev)
    init_state(len(G), G)
    U, pup = polish(U, outer, tag=tag)
    return U, real_cost(U, pup), pup


def cmd_swap(a):
    t0 = time.time()
    CFG.update(w=0.0, pd=False)
    base = load_catalog_designs(a.files, a.n, key='loss')
    G = [[tuple(r['glasses'][e].split(':')) for e in ELEMS] for r in base]
    U = np.array([P.phys_to_u(np.array(r['x'])) for r in base])
    Ut, C, _ = run_set(U, G, 0, 'base'); U = Ut.cpu().numpy(); C = C.cpu().numpy()
    hist = [[] for _ in base]
    allowed_el = a.elems.split(',') if a.elems else ELEMS
    for rd in range(a.rounds):
        cand, owner, desc = [], [], []
        for b, g in enumerate(G):
            for i, e in enumerate(ELEMS):
                if e not in allowed_el: continue
                for q in neighbours(e, g[i], a.k):
                    g2 = list(g); g2[i] = q; cand.append(g2); owner.append(b); desc.append('%s:%s->%s' % (e, g[i][1], q[1]))
        Uc, Cc = [], []
        for k in range(0, len(cand), a.chunk):
            u, c, _ = run_set(U[owner[k:k + a.chunk]], cand[k:k + a.chunk], a.outer, 'R%d c%d' % (rd, k // a.chunk))
            Uc.append(u.cpu().numpy()); Cc.append(c.cpu().numpy())
        Uc = np.concatenate(Uc); Cc = np.concatenate(Cc)
        nimp = 0
        for b in range(len(G)):
            idx = [j for j in range(len(cand)) if owner[j] == b]
            if not idx: continue
            j = min(idx, key=lambda j: Cc[j])
            if Cc[j] < 0.97 * C[b]:
                print('  round %d design %d: %s  loss %.0f -> %.0f' % (rd, b, desc[j], C[b], Cc[j]), flush=True)
                G[b] = cand[j]; U[b] = Uc[j]; C[b] = Cc[j]; hist[b].append(desc[j]); nimp += 1
        print('round %d: %d improved (%.0fs)' % (rd, nimp, time.time() - t0), flush=True)
        if nimp == 0: break
    Ut, C, pup = run_set(U, G, a.final, 'final')
    set_rows(torch.arange(len(G))); rep = OB.report(Ut, pup)
    out = []
    for b, r in enumerate(rep):
        r = finish(r, G[b])
        r.update(swaps=hist[b], from_mtf=base[b]['mtf_rms'], from_glasses=base[b]['glasses'], hist=base[b].get('hist'),
                 idx=base[b].get('idx'), src=base[b].get('src'))
        out.append(r)
    out.sort(key=lambda r: r['mtf_rms'])
    dump(out, a.out)
    for r in out:
        print('%s  (was %.4f) swaps %s' % (line(r), r['from_mtf'], r['swaps']), flush=True)
    print('total %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('attract')
    p.add_argument('--starts', required=True, help='continuous-search jsonl file(s), comma separated')
    p.add_argument('--top', type=int, default=16)
    p.add_argument('--sched', default='1:0.5,3:0.25,10:0.12,30:0.06,100:0.03', help='attraction stages w:tau')
    p.add_argument('--siters', type=int, default=8)
    p.add_argument('--dfreeze', type=float, default=0.15)
    p.add_argument('--gw', default='30:0.06', help='attraction kept on during the greedy endgame')
    p.add_argument('--giters', type=int, default=6)
    p.add_argument('--polish', type=int, default=0)
    p.add_argument('--out', required=True)
    p = sub.add_parser('polish')
    p.add_argument('files', nargs='+'); p.add_argument('--n', type=int, default=16)
    p.add_argument('--outer', type=int, default=12); p.add_argument('--lm', type=int, default=6)
    p.add_argument('--out', required=True)
    p = sub.add_parser('swap')
    p.add_argument('files', nargs='+'); p.add_argument('--n', type=int, default=4); p.add_argument('--k', type=int, default=3)
    p.add_argument('--rounds', type=int, default=2); p.add_argument('--outer', type=int, default=6)
    p.add_argument('--final', type=int, default=10); p.add_argument('--chunk', type=int, default=48)
    p.add_argument('--elems', default='', help='elements allowed to change (default all)')
    p.add_argument('--out', required=True)
    a = ap.parse_args()
    dict(attract=cmd_attract, polish=cmd_polish, swap=cmd_swap)[a.cmd](a)
