#!/usr/bin/env python3
"""第 3 级：把虚拟玻璃逐片吸附到真实牌号，每吸附一批就在 CODE V 里重优化。

    python cv_snap.py SIGMA85_s2.json --xlsx SIGMA85.xlsx -o SIGMA85_s3 [--vendor HOYA] [--batch-dist 0.8] [--k 1]

策略（贪心）：
  - 距离度量：vd 差 1 ≈ nd 差 0.01。离实牌号最近的先吸附（改动最小、最不容易把解踢飞）。
  - 距离 < batch-dist 的一次全吸附；其余一片一片来，每片吸附后 AUT mxc-step 个循环，让剩下的虚拟玻璃和曲率补偿。
  - --k > 1：每片试最近的 k 个候选，各跑一次短 AUT，留误差函数最小的（慢 k 倍）。
  - 非球面片允许模压料（M-/MP-/MC-），球面片只用研磨料。
  - 全部吸附完再跑一次长 AUT。
每一步都打印误差函数，便于盯着看哪一片最伤。
"""
import argparse, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import start_fix as sf
import cv_opt
from cv_session import CodeV
import live

MOLD = re.compile(r'^(M-|MP-|MC-)')


def cv_name(g):
    return g['name'].replace('-', '') + '_' + g['vendor']


def run_aut(cv, m, J, mxc, poly, uid, fict, log):
    nf = int(cv.eva('(NUM F)') or len(cv_opt.FIELDS))
    ef = cv_opt.run_block(cv, cv_opt.aut_block(m, J, mxc, poly, uid, fict, nf=nf), log + f'_{uid}.seq')[1]
    cv_opt.refresh_vig(cv, m)                            # 光阑口径与渐晕跟着镜头重算，轴上渐晕保持为零
    return ef


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('json'); ap.add_argument('--xlsx', required=True); ap.add_argument('-o', required=True)
    ap.add_argument('--batch-dist', type=float, default=0.8); ap.add_argument('--k', type=int, default=1)
    ap.add_argument('--mxc-step', type=int, default=8); ap.add_argument('--mxc-final', type=int, default=40)
    ap.add_argument('--poly', nargs='+', default=['FCD100', 'TAC6', 'TAFD30L', 'TAFD65', 'FDS16W'])
    a = ap.parse_args()
    J = json.load(open(a.json, encoding='utf-8'))
    m = sf.Model(sf.read_xlsx(a.xlsx), argparse.Namespace(**J['args'])); x0 = np.array(J['x'])
    o = os.path.abspath(a.o); src = os.path.splitext(os.path.abspath(a.json))[0] + '.len'
    gs = [i + 1 for i, s in enumerate(m.S) if s['glass']]                    # CODE V 面号
    asph_el = {m.S[i]['elem'] for i in m.ai if m.S[i]['glass']} | {m.S[i - 1]['elem'] for i in m.ai if i > 0 and m.S[i - 1]['glass']}
    chosen = {}
    live.init(os.path.dirname(o), m, xlsx=os.path.abspath(a.xlsx), title=sf.read_xlsx(a.xlsx)['title'])

    def show(cv, msg, ef):
        gl = [{'name': chosen[e]['name']} if e in chosen else {} for e in range(m.ne)]
        live.publish(cv_opt.read_back(cv, m, x0), stage='3 吸附', msg=msg, errf=ef, glasses=gl, mtf=cv_opt.mtf_report(cv, m), force=True)

    with CodeV(workdir=os.path.dirname(o)) as cv:
        cv.cmd(f'RES "{src}"'); uid = 0

        def cands(e):
            k = gs[e]; nd, vd = cv.eva(f'(IND S{k} W2)'), cv.eva(f'(ABV S{k})')
            lib = [g for g in m.lib if (e in asph_el or not MOLD.match(g['name']))]
            if m.is_sld[e]: lib = [g for g in lib if g['vd'] >= 67]
            lib.sort(key=lambda g: ((g['nd'] - nd) / 0.01) ** 2 + (g['vd'] - vd) ** 2)
            return [(np.hypot((g['nd'] - nd) / 0.01, g['vd'] - vd), g) for g in lib[:max(a.k, 3)]]

        def set_glass(e, g):
            out = cv.cmd(f'GLA S{gs[e]} {cv_name(g)}')
            return 'rror' not in out

        remaining = list(range(m.ne))
        e0 = run_aut(cv, m, J, 3, a.poly, 'u0', [gs[e] for e in remaining], o); live.log(f'起点误差函数 {e0:.3f}'); show(cv, '起点', e0)
        # 批量：很近的一次吸附
        batch = [(e, cands(e)) for e in remaining]
        for e, cs in batch:
            if cs[0][0] < a.batch_dist:
                for dist, g in cs:
                    if set_glass(e, g): chosen[e] = g; remaining.remove(e); live.log(f'  片{e + 1:2d} → {g["name"]:<12s} (距离 {dist:.2f}) [批量]'); break
        if chosen:
            uid += 1; ef = run_aut(cv, m, J, a.mxc_step, a.poly, f'u{uid}', [gs[e] for e in remaining], o); live.log(f'  批量吸附 {len(chosen)} 片后 误差函数 {ef:.3f}'); show(cv, f'批量 {len(chosen)} 片', ef)
        # 逐片
        while remaining and not live.should_stop():
            order = sorted(((cands(e)[0][0], e) for e in remaining)); _, e = order[0]
            cs = cands(e)[:a.k]; best = None
            if len(cs) > 1: cv.cmd(f'SAV "{o}_tmp.len"')
            for dist, g in cs:
                if len(cs) > 1: cv.cmd(f'RES "{o}_tmp.len"')
                if not set_glass(e, g): print(f'    CODE V 不认 {cv_name(g)}，跳过'); continue
                uid += 1; ef = run_aut(cv, m, J, a.mxc_step, a.poly, f'u{uid}', [gs[r] for r in remaining if r != e], o)
                live.log(f'  片{e + 1:2d} 试 {g["name"]:<12s} (距离 {dist:.2f}) → 误差函数 {ef:.3f}')
                if best is None or ef < best[0]:
                    best = (ef, g)
                    if len(cs) > 1: cv.cmd(f'SAV "{o}_best.len"')
            if best is None:
                print(f'  片{e + 1} 没有可用候选，保留虚拟玻璃'); remaining.remove(e); continue
            if len(cs) > 1: cv.cmd(f'RES "{o}_best.len"')
            chosen[e] = best[1]; remaining.remove(e); show(cv, f'片{e + 1} → {best[1]["name"]}（剩 {len(remaining)}）', best[0])
        uid += 1; ef = run_aut(cv, m, J, a.mxc_final, a.poly, f'u{uid}', [], o); live.log(f'收尾长 AUT：误差函数 {ef:.3f}')
        cv_opt.refresh_vig(cv, m); print(f"  EFL {cv.eva('(EFL)'):.4f}")
        spots = cv_opt.spot_report(cv)
        for f, d in spots: live.log(f'  视场 {f}°  RMS 弥散斑直径 {d:.1f} µm')
        cv.cmd(f'SAV "{o}.len"'); cv.cmd(f'WRL "{o}.seq"')
        x = cv_opt.read_back(cv, m, x0)
        live.publish(x, stage='3 完成', msg=os.path.basename(o), errf=ef, spots=spots, mtf=cv_opt.mtf_report(cv, m, dense=11),
                     glasses=[{'name': chosen[e]['name']} if e in chosen else {} for e in range(m.ne)], force=True)
    for f in (o + '_tmp.len', o + '_best.len'):
        if os.path.exists(f): os.remove(f)
    glasses = [{'elem': e + 1, 'nd': x[m.sl['nd']][e], 'vd': x[m.sl['vd']][e], 'sld': m.is_sld[e],
                'name': chosen[e]['name'] if e in chosen else None, 'nearest': chosen[e]['name'] if e in chosen else '?'} for e in range(m.ne)]
    print('  牌号：' + '  '.join(f"{g['elem']}:{g['name']}" for g in glasses))
    J2 = dict(J); J2.update(x=x.tolist(), glasses=glasses, c=x[m.sl['c']].tolist(), stage=3)
    json.dump(J2, open(o + '.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('写出', o + '.len', o + '.seq', o + '.json')


if __name__ == '__main__':
    main()
