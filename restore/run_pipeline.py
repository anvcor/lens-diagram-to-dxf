"""run_pipeline.py -- the whole restoration search for one lens config, per glass mode:

  c      continuous search, model glasses (objective.py: random starts -> quick DLS -> keep the best -> full DLS,
         hard constraints)                                                       -> <mode>_c.jsonl
  a      catalog attraction + progressive freezing + greedy snap (snap.py attract) -> <mode>_a.jsonl
  p      shape polish of the all-catalog designs (snap.py polish)                   -> <mode>_p.jsonl
  s      one-element glass swaps, each polished (snap.py swap)                      -> <mode>_s.jsonl
  t      tight polish: CRA / RI limits tightened by a margin (snap.py polish)       -> <mode>_t.jsonl
  cv     optional CODE V re-ranking of the best designs (cvtools.py rank)           -> cv/<mode>_cvrank.json

Every step runs as its own process (limits are read at import); a step whose output exists is skipped (resume).
Settings: defaults below < config "pipeline" section < command line.  Glass modes: config glass.modes.
usage: python run_pipeline.py --config examples/mylens/config.json --out runs/mylens_r1 [--modes a,b] [--codev]
       python run_pipeline.py --config ... --out ... --smoke          (tiny settings: checks the setup in minutes)"""
import os, sys, json, time, argparse, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULTS = dict(
    nstart=512, quick=15, keep=96, iters=60, chunk=48, seed=1, starts=None,
    top=16, sched='1:0.5,3:0.25,10:0.12,30:0.06,100:0.03', siters=8, giters=6, dfreeze=0.15, apolish=0,
    pn=16, pouter=12, plm=6,
    sn=4, sk=3, srounds=2, souter=6, sfinal=10,
    tn=8, touter=12, tlm=4, cra_margin=0.2, ri_margin=0.007,
    cvn=6, env={})
SMOKE = dict(nstart=24, quick=2, keep=4, iters=3, chunk=4, top=2, sched='1:0.5,30:0.06', siters=1, giters=1,
             pn=2, pouter=1, plm=1, sn=1, sk=1, srounds=1, souter=1, sfinal=1, tn=2, touter=1, tlm=1, cvn=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--modes', default=None, help='comma separated glass modes (default: all of config glass.modes)')
    ap.add_argument('--steps', default='c,a,p,s,t', help='subset of c,a,p,s,t to run')
    ap.add_argument('--codev', action='store_true', help='CODE V re-ranking at the end (Windows + CODE V)')
    ap.add_argument('--smoke', action='store_true')
    for k, v in DEFAULTS.items():
        if k != 'env':
            ap.add_argument('--' + k, type=type(v) if v is not None else str, default=None)
    a = ap.parse_args()
    cfg_path = os.path.abspath(a.config)
    C = json.load(open(cfg_path, encoding='utf-8'))
    S = dict(DEFAULTS); S.update(C.get('pipeline', {}))
    if a.smoke: S.update(SMOKE)
    S.update({k: getattr(a, k) for k in DEFAULTS if k != 'env' and getattr(a, k, None) is not None})
    modes = a.modes.split(',') if a.modes else list(C.get('glass', {}).get('modes', {'default': {}}))
    steps = a.steps.split(',')
    out = os.path.abspath(a.out); os.makedirs(out, exist_ok=True)
    lim = C['limits']
    env0 = dict(os.environ, RESTORE_CONFIG=cfg_path, PYTHONUNBUFFERED='1')
    env0.update({k: str(v) for k, v in S['env'].items()})
    json.dump(dict(settings=S, modes=modes, steps=steps, config=cfg_path), open(os.path.join(out, 'pipeline.json'), 'w'), indent=1)
    tlog = open(os.path.join(out, 'times.txt'), 'a')
    t0 = time.time()

    def run(tag, args, env, outfile):
        if os.path.exists(outfile):
            print('[skip] %s (%s exists)' % (tag, os.path.basename(outfile)), flush=True); return
        print('[run ] %s ...' % tag, flush=True)
        t1 = time.time()
        with open(os.path.join(out, tag + '.log'), 'w') as lg:
            r = subprocess.run([sys.executable, '-u'] + args, cwd=HERE, env=env, stdout=lg, stderr=subprocess.STDOUT)
        tlog.write('%s %s %.0fs rc=%d\n' % (time.strftime('%H:%M:%S'), tag, time.time() - t1, r.returncode)); tlog.flush()
        if r.returncode != 0 or not os.path.exists(outfile):
            raise SystemExit('step %s failed (rc %d): see %s' % (tag, r.returncode, os.path.join(out, tag + '.log')))
        print('       done in %.0fs' % (time.time() - t1), flush=True)

    for m in modes:
        env = dict(env0, GLASS_MODE=m)
        f = {k: os.path.join(out, '%s_%s.jsonl' % (m, k)) for k in ('c', 'a', 'p', 's', 't')}
        if 'c' in steps:
            args = ['objective.py', '--keep', str(S['keep']), '--iters', str(S['iters']), '--chunk', str(S['chunk']),
                    '--seed', str(S['seed']), '--out', f['c']]
            args += ['--starts', os.path.abspath(S['starts'])] if S['starts'] else ['--nstart', str(S['nstart']), '--quick', str(S['quick'])]
            run('%s_c' % m, args, env, f['c'])
        if 'a' in steps:
            run('%s_a' % m, ['snap.py', 'attract', '--starts', f['c'], '--top', str(S['top']), '--sched', S['sched'],
                             '--siters', str(S['siters']), '--giters', str(S['giters']), '--dfreeze', str(S['dfreeze']),
                             '--polish', str(S['apolish']), '--out', f['a']], env, f['a'])
        if 'p' in steps:
            run('%s_p' % m, ['snap.py', 'polish', f['a'], '--n', str(S['pn']), '--outer', str(S['pouter']), '--lm', str(S['plm']),
                             '--out', f['p']], env, f['p'])
        if 's' in steps:
            run('%s_s' % m, ['snap.py', 'swap', f['p'], '--n', str(S['sn']), '--k', str(S['sk']), '--rounds', str(S['srounds']),
                             '--outer', str(S['souter']), '--final', str(S['sfinal']), '--out', f['s']], env, f['s'])
        if 't' in steps:
            et = dict(env, LENS_CRA_MAX=str(float(lim.get('cra_max', 90)) - S['cra_margin']),
                      LENS_RI_MIN=str(float(lim.get('ri_min', 0)) + S['ri_margin']))
            run('%s_t' % m, ['snap.py', 'polish', f['p'], f['s'], '--n', str(S['tn']), '--outer', str(S['touter']), '--lm', str(S['tlm']),
                             '--out', f['t']], et, f['t'])
        if a.codev:
            cvdir = os.path.join(out, 'cv')
            run('%s_cv' % m, ['cvtools.py', 'rank', '--tag', m, '--files', f['t'], f['s'], '--n', str(S['cvn']), '--outdir', cvdir],
                env, os.path.join(cvdir, '%s_cvrank.json' % m))
    # ---- summary: best feasible design per mode (solver numbers; CODE V numbers if ranked)
    print('\n==== summary (%.0f min) ====' % ((time.time() - t0) / 60))
    for m in modes:
        R = []
        for k in ('t', 's', 'p'):
            p = os.path.join(out, '%s_%s.jsonl' % (m, k))
            if os.path.exists(p): R += [dict(json.loads(l), _f=k) for l in open(p)]
        ok = [r for r in R if r['cra'] <= float(lim.get('cra_max', 90)) and r['ri'] >= float(lim.get('ri_min', 0))]
        if not R:
            print('%-10s no all-catalog designs yet' % m); continue
        b = min(ok or R, key=lambda r: r['mtf_rms'])
        print('%-10s best%s mtf_rms %.4f  CRA %.2f  RI %.3f  EFL %.2f  (%s step) | %s' % (
            m, '' if ok else ' (NOT feasible)', b['mtf_rms'], b['cra'], b['ri'], b['efl'], b['_f'],
            ' '.join(v.split(':')[1] for v in b['glasses'].values())))
        cvf = os.path.join(out, 'cv', '%s_cvrank.json' % m)
        if os.path.exists(cvf):
            X = json.load(open(cvf))
            if X:
                x = X[0]
                print('%-10s CODE V pick %s: GEO %.4f DIF %.4f CRA %.2f RI %.3f ok=%s -> deliver from cv/%s.seq' % (
                    '', x['stem'], x.get('GEO', float('nan')), x.get('DIF', float('nan')), x['cra'], x['ri'], x['ok'], x['stem']))


if __name__ == '__main__':
    main()
