#!/usr/bin/env python3
"""第 2 级：把第 1 级的起点送进 CODE V 跑 AUT（实光线、全口径、含弧矢）。

    python cv_opt.py SIGMA85_start.json --xlsx SIGMA85.xlsx -o SIGMA85_s2 [--mxc 40] [--poly FCD100 TAC6 TAFD30L TAFD65 FDS16W] [--ap-cons]

做的事：
  1. 生成两份人能读、能改的宏：
       <o>_setup.seq —— in 起点、波长/视场/F 数、光阑口径、变量（CCY/GLC/AC..DC/THC）。只有这份设变量。
       <o>_aut.seq   —— 约束宏，写法照用户的手工宏（SKILL.md §6.4）：宏里不设变量、只有 INF
           焦距   EFL 上下限；畸变 DIY ±dist_max；后截距 ET/THI
           球面   RDY 取起点半径 ±5%（RDY_PCT）—— 传统写法，用户放开口径、设渐晕点后仍然有效。
                  旧版用「边缘矢高框 ≡ CUY 上下界」：固定高度上的矢高约束，放开口径后就失效了，弃用。
           非球面 顶点 RDY 与起点同号、|R| 30..500（500 = 加工极限），高次项不加约束。
                  旧版用 SAGF 卡总矢高，弃用；顶点 R 不限的话会变号，再靠高次项掰回来，整片弯反。
           加工   MNT / MNE / MNA 通用约束。旧版 @E 在固定高度上用 SAGF 算边厚，同样随口径失效，弃用。
           玻璃   GLA 多边形边界（用户惯用的 5 个顶点牌号）；SLD 片另卡 ABV > 67
           口径   图上口径上限的光线约束默认不加（用户习惯自己放开口径、设渐晕点），--ap-cons 打开
       在 CODE V 里手工复现：in <o>_setup.seq（或自己设变量）→ in <o>_aut.seq。
  2. COM 里跑它（setup + 分段 AUT），存 <o>.len / <o>.seq，AUT 全文存 <o>_aut.txt
  3. 把结果读回写成 <o>.json（与第 1 级同格式）→ 可直接喂 layout_png.py 出叠图
"""
import argparse, json, math, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import start_fix as sf
from cv_session import CodeV
import live


FIELDS = [round(k / 9, 4) for k in range(10)]            # 优化用的相对像高（YRI = 这些 × 像高）：10 个视场点


def set_fields(cv, m, fields):
    """视场用实像高 YRI（EFL 浮动时视场角跟着变，像高不变——与用户的 .len 一致）。"""
    cv.cmd('YRI ' + ' '.join(f'{f * m.a.ih:.4f}' for f in fields)); cv.cmd('WTF ' + ' '.join(['1'] * len(fields)))


def refresh_vig(cv, m):
    """光阑口径 = 轴上满口径边缘光线在光阑面的高度（由 F 数定，不用图上量的）→ 重算渐晕 → 轴上渐晕清零。
    轴上要是还有渐晕，说明边缘光线超了某片的口径：那是要用约束解决的设计问题，不能让 SET VIG 悄悄裁掉。"""
    z = 'VUY F1 0; VLY F1 0; VUX F1 0; VLX F1 0'; a = m.a
    cv.cmd(z); nf = int(cv.eva('(NUM F)')); ca = {}
    for i, s in enumerate(m.S):
        k = i + 1; ym = abs(cv.eva(f'(Y R2 S{k} F1)') or 0.0)
        if s['stop']: ca[k] = ym; continue
        # 通光口径由光线定（图上的口径是画大了的，只当上限）：轴上光束全过 + 最大视场保留 v 比例的光束
        yc = abs(cv.eva(f'(Y R1 S{k} F{nf})') or 0.0)
        ca[k] = min(max(a.ca_k * ym, yc + a.vig_v * ym), float(m.cmax[i]))
    cv.cmd('; '.join(f'CIR S{k} {v:.4f}' for k, v in ca.items()))
    cv.cmd('SET VIG'); cv.cmd(z)
    return ca


def rel_illum(cv):
    """最大视场相对照度（取自 MTF 选项的输出，含渐晕与 cos⁴）。"""
    r = re.findall(r'RELATIVE ILLUMINATION\s*=\s*([\d.]+)', cv.cmd('MTF; MFR 10; IFR 10; GO'))
    return float(r[-1]) / 100 if r else None


def tune_vig(cv, m, fields):
    """标定 v：让最大视场相对照度 ≈ ri_target（这类大光圈定焦一般 0.25 左右）。只留轴上和最大视场两个点算得快。"""
    set_fields(cv, m, [0.0, 1.0]); lo, hi = 0.05, 1.0; ri = None
    for _ in range(7):
        m.a.vig_v = (lo + hi) / 2; refresh_vig(cv, m); ri = rel_illum(cv)
        if ri is None: break
        if ri > m.a.ri_target: hi = m.a.vig_v
        else: lo = m.a.vig_v
    set_fields(cv, m, fields); refresh_vig(cv, m)
    return m.a.vig_v, ri


RDY_PCT = 0.05                       # 球面 RDY 上下限：起点半径 ±5%
ASPH_R = (30.0, 500.0)               # 非球面顶点 |R| 范围（500 = 加工极限）
FLAT_C = 1e-5                        # |c| 小于它当平面：不放变量、不加 RDY


def sph_surfs(m):
    """[(CODE V 面号, 起点 R)]：曲率可变的球面（不含平面和非球面）。"""
    return [(i + 1, 1.0 / c) for i, c in zip(m.ci, m.c0) if i not in m.ai and abs(c) > FLAT_C]


def asph_surfs(m):
    """[(CODE V 面号, 起点 R)]：顶点曲率可变的非球面（平面基底的非球面顶点冻结）。"""
    return [(i + 1, 1.0 / m.c0[m.ci.index(i)]) for i in m.ai if abs(m.c0[m.ci.index(i)]) > FLAT_C]


def aut_block(m, J, mxc, poly, uid='', fict=None, nf=len(FIELDS), x=None, ap_cons=False):
    """AUT ... GO 这一段：只有约束，不设变量（变量在 _setup.seq）。
    uid：同一会话里 UDC 名字只能定义一次，多次 AUT 时给不同后缀。
    fict：仍是虚拟玻璃的面号（CODE V 编号）；None = 全部。"""
    S = m.S; last = len(S)
    glass_s = [i + 1 for i, s in enumerate(S) if s['glass']]
    if fict is None: fict = glass_s
    a = m.a
    L = ['AUT', '', f'! 焦距：标称 {a.efl} 的 ±{a.efl_tol * 100:g}% 以内都算对',
         f'EFL Z1 > {a.efl * (1 - a.efl_tol):.4f}', f'EFL Z1 < {a.efl * (1 + a.efl_tol):.4f}', '',
         f'! 后截距：末片后表面边缘 → 盖板 的空气边缘厚度（带盖板时 S{last} 就是 SI-3，即用户写的 ET SI-3）',
         f'ET S{last} > {a.bfl_et}', f'THI S{last} < {a.back_max}', '',
         f'! ---- 畸变 ±{a.dist_max * 100:g}%（DIY 是比例，0.02 = 2%）']
    for k in range(2, nf + 1): L += [f'DIY F{k} Z1 < {a.dist_max}', f'DIY F{k} Z1 > {-a.dist_max}']
    A = asph_surfs(m)
    if A:
        L += ['', f'! ---- 非球面顶点半径：与起点同号，|R| {ASPH_R[0]:g}..{ASPH_R[1]:g}（{ASPH_R[1]:g} = 加工极限）；高次项不约束']
        for k, R in A:
            hi = max(ASPH_R[1], 1.2 * abs(R))          # 起点本身比 500 还平的，上限放到 1.2|R|
            if R > 0: L += [f'RDY S{k} > {ASPH_R[0]:g}', f'RDY S{k} < {hi:g}']
            else:     L += [f'RDY S{k} < {-ASPH_R[0]:g}', f'RDY S{k} > {-hi:g}']
    L += ['', f'! ---- 球面半径：起点 ±{RDY_PCT * 100:g}%（不用 SAGF 矢高框：放开口径后会失效）']
    for k, R in sph_surfs(m):
        lo, hi = sorted((R * (1 - RDY_PCT), R * (1 + RDY_PCT)))
        L += [f'RDY S{k} < {hi:.4f}', f'RDY S{k} > {lo:.4f}']
    if ap_cons:
        L += ['', '! ---- 口径（--ap-cons）：需要的通光口径 ≤ 图上该面的绘制范围（图上口径是画大了的，只当上限）',
              f'!      轴上：{a.ca_k}×边缘光线高度(R2 F1)；轴外：|主光线| + {a.vig_v:.3f}×边缘光线高度（主光线在光阑前为负、后为正）']
        for i, s in enumerate(S):
            if s['stop']: continue
            k = i + 1; lim = float(m.lim[i]); sg = '-' if i < m.ista else '+'
            L.append(f'Y R2 S{k} F1 < {lim / a.ca_k:.3f}')
            L += [f'@B{k}{uid} == {a.vig_v:.4f}*(Y R2 S{k} F1) {sg} (Y R1 S{k} F{nf})', f'@B{k}{uid} < {lim:.3f}']
    if fict:
        L += ['', '! ---- 玻璃：多边形边界 + SLD 片的阿贝数下限', f'GLA S1..{last} ' + ' '.join(poly)]
        for e_, k in enumerate(glass_s):
            if m.is_sld[e_] and k in fict: L += [f'@V{k}{uid} == (ABV S{k})', f'@V{k}{uid} > 67']
    return L + ['', 'INT NO', f'MXC {mxc}', f'MNC {min(5, mxc)}', 'EFT TA', 'DRA S1..I YES', '',
                'EFP ALL Y', f'MNT {a.et_min:g}', f'MNE {a.et_min:g}', 'MNA 0.05', '', 'GO', '']


def build_setup(m, a, start_seq, out):
    """起点 + 波长/视场/F 数 + 光阑口径 + 变量。只有这份设变量。"""
    S = m.S; last = len(S)                               # CODE V 面号 = 下标 + 1
    glass_s = [i + 1 for i, s in enumerate(S) if s['glass']]
    L = [f'! 第 2 级 setup —— 由 cv_opt.py 生成；起点 {os.path.basename(start_seq)}', f'in "{start_seq}"',
         'WL 656.3 587.6 546.1 486.1 435.8', 'REF 3', 'WTW 10 27 29 23 11']
    k = m.ista + 1; z = 'VUY F1 0; VLY F1 0; VUX F1 0; VLX F1 0'
    L += ['! 视场用实像高（EFL 在 ±5% 内浮动时像高不变）；F 数固定', f'FNO {m.a.fno}',
          'YRI ' + ' '.join(f'{f * m.a.ih:.4f}' for f in a.fields), 'WTF ' + ' '.join(['1'] * len(a.fields)),
          '! 光阑口径由 F 数定；轴上渐晕清零',
          z, f'CIR S{k} (Y R2 S{k} F1)', 'SET VIG', z, '']
    L += ['! ---- 变量：球面 / 非球面顶点曲率（平面不放）、虚拟玻璃、非球面 A..D、像距']
    L += [f'CCY S{kk} 0' for kk, _ in sph_surfs(m) + asph_surfs(m)] + [f'GLC S{kk} 0' for kk in glass_s]
    for i in m.ai: L += [f'{v} S{i + 1} 0' for v in ('AC', 'BC', 'CC', 'DC')]
    L += [f'THC S{last} 0', '']
    open(out, 'w', encoding='utf-8', newline=chr(13) + chr(10)).write(chr(10).join(L))


def build_aut(m, J, a, start_seq, out):
    """约束宏：不设变量、只有 INF，写法照 SKILL.md §6.4。"""
    L = [f'! 第 2 级约束宏 —— 由 cv_opt.py 生成；起点 {os.path.basename(start_seq)}',
         '! 先 in 对应的 _setup.seq（或自己在镜头里设好变量），再 in 本文件。本宏不设变量。', '']
    L += aut_block(m, J, a.mxc, a.poly, nf=len(a.fields), ap_cons=a.ap_cons)
    open(out, 'w', encoding='utf-8', newline=chr(13) + chr(10)).write(chr(10).join(L))


def run_block(cv, lines, path):
    """把一段 AUT 写成临时 seq 跑掉，返回 (全文, 最后的 ERR.F.)。"""
    open(path, 'w', encoding='utf-8', newline=chr(13) + chr(10)).write(chr(10).join(lines))
    txt = cv.run_seq(path); os.remove(path)
    bad = [l for l in txt.splitlines() if re.search(r'\bError\b', l)]
    if bad: live.log('    ! ' + bad[0].strip())
    e = re.findall(r'ERR\. F\.\s*=\s*([\d.Ee+-]+)', txt)
    return txt, (float(e[-1]) if e else float('nan'))


def read_back(cv, m, x0):
    x = x0.copy(); S = m.S
    x[m.sl['c']] = [cv.eva(f'(CUY S{i + 1})') for i in m.ci]
    gs = [i + 1 for i, s in enumerate(S) if s['glass']]
    x[m.sl['nd']] = [cv.eva(f'(IND S{k} W2)') for k in gs]
    x[m.sl['vd']] = [cv.eva(f'(ABV S{k})') for k in gs]
    A = []
    for i in m.ai: A += [cv.eva(f'({v} S{i + 1})') or 0.0 for v in 'ABCD']
    x[m.sl['A']] = A
    x[m.sl['bk']] = cv.eva(f'(THI S{len(S)})')
    return x


def spot_report(cv):
    out = []; fld = None
    for l in cv.cmd('SPO; GO').splitlines():
        mm = re.search(r'Field +(\d+), *\(([^)]*)\)', l)
        if mm: fld = mm.group(2).split(',')[-1].strip(); continue
        mm = re.search(r'([\d.E+-]+) MM\s*$', l)
        if mm and fld is not None: out.append((fld, float(mm.group(1)) * 1000)); fld = None
    return out


def mtf_report(cv, m, dense=0):
    """CODE V 衍射 MTF（10/30 lp/mm，RAD=S / TAN=M）+ 相对照度 + 畸变，逐视场。
    dense>0：临时把视场换成 dense 个等间隔像高点（重新 SET VIG），只在一级结束、镜头已存盘后用——会改掉视场设置。"""
    if dense:
        set_fields(cv, m, [k / (dense - 1) for k in range(dense)]); refresh_vig(cv, m)
    txt = cv.cmd('MTF; MFR 30; IFR 10; GO'); out = []; cur = None
    mm = re.search(r'F-numbers for each field angle\s+X\s+([\d.]+)', txt)
    fno_axis = float(mm.group(1)) if mm else None                         # 轴上实际 F 数：≠ 设定值就说明轴上被裁了
    for l in txt.splitlines():
        mm = re.search(r'FIELD\s+(\d+)\s*\((ANG|RIH)\)\s*=\s*\(\s*[-\d.]+,\s*([-\d.]+)\)', l)
        if mm:
            v = float(mm.group(3)); cur = {'h': round(v if mm.group(2) == 'RIH' else math.tan(math.radians(v)) * m.a.efl, 2), 'fno_axis': fno_axis}
            out.append(cur); continue
        if cur is None: continue
        mm = re.search(r'RELATIVE ILLUMINATION\s*=\s*([\d.]+)', l)
        if mm: cur['ri'] = float(mm.group(1)); continue
        mm = re.search(r'DISTORTION\s*=\s*([-\d.]+)', l)
        if mm: cur['dist'] = float(mm.group(1)); continue
        q = l.split()
        if len(q) >= 4 and q[0] in ('10', '20', '30'):
            try:                                                          # [RAD(S), TAN(M)]；轴上只印一列
                cur['f' + q[0]] = [float(q[-2]), float(q[-1])] if len(q) >= 6 else [float(q[-1])] * 2
            except ValueError: pass
    for d in out: d.setdefault('ri', 100.0); d.setdefault('dist', 0.0)     # 轴上那一块不印 RI / 畸变
    return [d for d in out if 'f30' in d]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('json'); ap.add_argument('--xlsx', required=True); ap.add_argument('-o', required=True)
    ap.add_argument('--mxc', type=int, default=40); ap.add_argument('--chunk', type=int, default=3, help='每段 AUT 的循环数（段间刷新网页）')
    ap.add_argument('--mtf-fields', type=int, default=11, help='结束时 MTF vs 像高 曲线的视场点数')
    ap.add_argument('--fields', type=float, nargs='+', default=FIELDS)
    ap.add_argument('--ap-cons', action='store_true', help='加上图上口径上限的光线约束（默认不加）')
    ap.add_argument('--poly', nargs='+', default=['FCD100', 'TAC6', 'TAFD30L', 'TAFD65', 'FDS16W'],
                    help='GLA 玻璃图多边形的顶点牌号（3~5 个）')
    a = ap.parse_args()
    J = json.load(open(a.json, encoding='utf-8'))
    m = sf.Model(sf.read_xlsx(a.xlsx), argparse.Namespace(**J['args'])); x0 = np.array(J['x'])
    wd = os.path.dirname(os.path.abspath(a.o)); o = os.path.abspath(a.o)
    start_seq = os.path.splitext(os.path.abspath(a.json))[0] + '.seq'
    build_setup(m, a, start_seq, o + '_setup.seq'); build_aut(m, J, a, start_seq, o + '_aut.seq')
    print('写出', o + '_setup.seq', '（起点 + 变量）和', o + '_aut.seq', '（约束宏，不设变量）')
    live.init(wd, m, xlsx=os.path.abspath(a.xlsx), title=sf.read_xlsx(a.xlsx)['title'])
    with CodeV(workdir=wd) as cv:
        # 为了能实时看：AUT 切成每段 chunk 个循环，段间回读并发布；网页点「停止」则在段间收尾
        cv.run_seq(o + '_setup.seq')
        v, ri = tune_vig(cv, m, a.fields); x = read_back(cv, m, x0)
        live.log(f'  渐晕标定：v = {v:.3f} → 最大视场相对照度 {ri}（目标 {m.a.ri_target}）')
        live.publish(x, stage='2 CODE V AUT', msg='渐晕标定完成', force=True)
        done = 0; full = ''; k = 0; ef = None
        while done < a.mxc and not live.should_stop():
            n = min(a.chunk, a.mxc - done); k += 1
            txt, ef = run_block(cv, aut_block(m, J, n, a.poly, uid=f'C{k}', nf=len(a.fields), x=x, ap_cons=a.ap_cons), o + f'_c{k}.seq'); full += txt; done += n
            refresh_vig(cv, m)                           # 镜头变了，光阑口径与渐晕跟着重算
            x = read_back(cv, m, x0)
            live.log(f"  AUT 循环 {done}/{a.mxc}: ERR.F. {ef:.4f}   EFL {cv.eva('(EFL)'):.3f}")
            live.publish(x, stage='2 CODE V AUT', msg=f'循环 {done}/{a.mxc}', errf=ef, mtf=mtf_report(cv, m), force=True)
        open(o + '_aut.txt', 'w', encoding='utf-8').write(full)
        refresh_vig(cv, m)
        print(f"  EFL {cv.eva('(EFL)'):.4f}")
        spots = spot_report(cv)
        for f, d in spots: live.log(f'  视场 {f}°  RMS 弥散斑直径 {d:.1f} µm')
        cv.cmd(f'SAV "{o}.len"'); cv.cmd(f'WRL "{o}.seq"')
        x = read_back(cv, m, x0)
        live.publish(x, stage='2 完成', msg=os.path.basename(o), errf=ef, spots=spots, mtf=mtf_report(cv, m, dense=a.mtf_fields), force=True)
    glasses = []
    for e in range(m.ne):
        nd, vd = x[m.sl['nd']][e], x[m.sl['vd']][e]
        g = min(m.lib, key=lambda g: ((g['nd'] - nd) / 0.01) ** 2 + (g['vd'] - vd) ** 2)
        glasses.append({'elem': e + 1, 'nd': nd, 'vd': vd, 'nearest': g['name'], 'sld': m.is_sld[e]})
        print(f"  {e + 1:2d}  {nd:.5f}  {vd:6.2f}  {g['name']:<20s} Δnd {g['nd'] - nd:+.4f}  Δvd {g['vd'] - vd:+6.2f}" + ('  SLD' if m.is_sld[e] else ''))
    J2 = dict(J); J2.update(args=vars(m.a), x=x.tolist(), glasses=glasses, c=x[m.sl['c']].tolist(), stage=2)
    json.dump(J2, open(o + '.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('写出', o + '.len', o + '.seq', o + '.json')


if __name__ == '__main__':
    main()
