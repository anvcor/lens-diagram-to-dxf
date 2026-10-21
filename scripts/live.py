#!/usr/bin/env python3
"""优化过程的实时状态发布：各级脚本调 publish()，viewer.py 的网页轮询 <workdir>/_live.json。

    import live
    live.init(workdir, m, xlsx=..., dxf=..., image=..., title=...)
    live.publish(x, stage='1b', msg='孔径 ×0.8', errf=12.3)        # 自带节流，可放在残差函数里随便调
    if live.should_stop(): break                                   # 网页上点了「停止」

约定：全部是「尽力而为」——发布失败绝不能拖垮优化，所以一律吞异常。
"""
import json, os, time
import numpy as np

_S = {'dir': None, 'm': None, 'meta': {}, 'hist': [], 'log': [], 't_last': 0.0, 't0': time.time(), 'extra': {}}


def _dxf_image(path):
    """从 diag_build.py 写的 DXF 里恢复 像素↔mm 映射 和 贴图文件名。"""
    t = open(path, encoding='utf-8', errors='ignore').read().split('\n'); out = {}
    for i in range(0, len(t) - 1, 2):
        k, v = t[i].strip(), t[i + 1].strip()
        if k == '0' and v == 'IMAGE':
            rec = {}; j = i + 2
            while t[j].strip() != '0': rec.setdefault(t[j].strip(), t[j + 1].strip()); j += 2
            out.update(xi=float(rec['10']), yi=float(rec['20']), mmpx=float(rec['11']), W=float(rec['13']), H=float(rec['23']))
        if k == '0' and v == 'IMAGEDEF':
            j = i + 2
            while t[j].strip() != '0':
                if t[j].strip() == '1': out['file'] = t[j + 1].strip().replace('.\\', '')
                j += 2
    return out


def init(workdir, m, xlsx=None, dxf=None, image=None, title=''):
    _S.update(dir=os.path.abspath(workdir), m=m, hist=[], log=[], t0=time.time(), extra={})
    meta = {'title': title, 'efl': m.a.efl, 'fno': m.a.fno, 'ih': m.a.ih}
    try:
        if dxf is None and xlsx: dxf = os.path.splitext(xlsx)[0] + '.dxf'
        if dxf and os.path.exists(dxf):
            im = _dxf_image(dxf)
            f = image or os.path.join(os.path.dirname(os.path.abspath(dxf)), im.get('file', ''))
            if os.path.exists(f): meta['image'] = dict(im, path=os.path.abspath(f))
    except Exception: pass
    _S['meta'] = meta
    try:                                                 # 接着上一级的历史画曲线
        old = json.load(open(os.path.join(_S['dir'], '_live.json'), encoding='utf-8'))
        _S['hist'] = old.get('hist', [])[-2000:]
    except Exception: pass
    clear_stop()


def log(msg):
    print(msg, flush=True)
    _S['log'].append(time.strftime('%H:%M:%S ') + str(msg)); _S['log'] = _S['log'][-300:]


def should_stop():
    return bool(_S['dir']) and os.path.exists(os.path.join(_S['dir'], '_stop'))


def clear_stop():
    try: os.remove(os.path.join(_S['dir'], '_stop'))
    except Exception: pass


def geometry(m, x, nray=201):
    import start_fix as sf
    c, t, n, dn, Aco = m.table(x); zv = np.r_[0, np.cumsum(t)]; ns = len(m.S)
    cx = x[m.sl['c']]; surf = []; ends = []; ca = m.clear_ap(m.seidel(x))
    for i in range(len(c)):
        if i < ns and m.S[i]['stop']:
            semi = float(m.S[i]['semi'] or 20); surf.append({'k': i + 1, 'stop': True, 'z': zv[i], 'semi': semi}); ends.append(None); continue
        semi = float(m.S[i]['draw'] or m.S[i]['semi']) if i < ns else 24.0
        ys = np.linspace(-semi, semi, 41); zs = zv[i] + sf.sag(c[i], ys, Aco[i])
        d = {'k': i + 1, 'pts': [[round(float(a), 3), round(float(b), 3)] for a, b in zip(zs, ys)], 'asph': i in m.ai,
             'R': None if abs(c[i]) < 1e-12 else round(1 / c[i], 3), 'semi': semi, 'ca': round(float(min(ca[i], semi)), 2)}
        if i < ns:
            j = m.ci.index(i); dz = float(sf.sag(cx[j], m.h[j], Aco[i]) - m.z0[j])
            d.update(dz=round(dz, 4), eps=round(float(m.eps[j]) * (m.a.asph_loose if i in m.ai else 1.0), 4), R0=None if abs(m.c0[j]) < 1e-12 else round(1 / m.c0[j], 3))
        surf.append(d); ends.append((float(zs[-1]), semi))
    # 镜片：前面轮廓 + 外缘台阶 + 后面轮廓 围成的填充多边形
    edges = []
    for i in range(len(c) - 1):
        if n[i] > 1.0 and ends[i] and ends[i + 1]:
            ha, hb = ends[i][1], ends[i + 1][1]; hh = max(ha, hb)
            fa, fb = surf[i]['pts'], surf[i + 1]['pts']; za, zb = fa[-1][0], fb[-1][0]
            poly = fa + [[za, hh], [zb, hh]] + fb[::-1] + [[zb, -hh], [za, -hh]]
            sld = i < ns and m.S[i].get('glass') and m.is_sld[m.S[i]['elem']]
            edges.append({'poly': [[round(float(a), 3), round(float(b), 3)] for a, b in poly], 'sld': bool(sld), 'cover': i >= ns})
    # 光线：每视场 3 条 —— 主光线 + 渐晕后光瞳的上下边缘光线（≈ CODE V 的 R1/R2/R3）；轴上画满口径 ±1
    fields = [0.0, 0.7, 1.0]; rhos = np.linspace(-1, 1, nray); path = []; ic = nray // 2
    hm, ubar, _ = m.pupil(x)
    yimg, ok, clip = m.rays(x, fields, rhos, m.seidel(x)['ep_z'], path=path)
    good = ok.all(axis=-1) & (clip <= 1.0)
    rays = []
    for f in range(len(fields)):
        if f == 0: pick = [0, ic, nray - 1]
        else:
            lo = hi = ic
            while lo > 0 and good[f, lo - 1]: lo -= 1
            while hi < nray - 1 and good[f, hi + 1]: hi += 1
            pick = [lo, ic, hi]
        ang = np.arctan(fields[f] * ubar)
        for r in pick:
            if not ok[f, r].all(): continue
            z0, y0 = path[0][0][f, r], path[0][1][f, r]
            pts = [[z0 - 15, y0 - 15 * np.tan(ang)]] + [[pz[f, r], py[f, r]] for pz, py in path]
            rays.append({'f': f, 'p': [[round(float(a), 2), round(float(b), 2)] for a, b in pts]})
    return {'surf': surf, 'edges': edges, 'rays': rays, 'img_z': float(zv[-1])}


def publish(x, stage='', msg='', errf=None, glasses=None, spots=None, force=False, **extra):
    """节流发布。errf：这一级自己的误差量（1a/1b 是 cost，CODE V 级是 ERR.F.）；spots：[(视场, µm)]。"""
    now = time.time()
    if not _S['dir'] or (not force and now - _S['t_last'] < 0.8): return
    _S['t_last'] = now
    try:
        m = _S['m']; x = np.asarray(x, float); o = m.seidel(x)
        if errf is not None and np.isfinite(errf): _S['hist'].append({'t': round(now - _S['t0'], 1), 'stage': stage, 'e': float(errf)})
        if extra: _S['extra'].update(extra)
        if spots is not None: _S['extra']['spots'] = [[str(f), float(d)] for f, d in spots]
        gl = []
        for e in range(m.ne):
            nd, vd = float(x[m.sl['nd']][e]), float(x[m.sl['vd']][e])
            g = min(m.lib, key=lambda g: ((g['nd'] - nd) / 0.01) ** 2 + (g['vd'] - vd) ** 2)
            dist = float(np.hypot((g['nd'] - nd) / 0.01, g['vd'] - vd))
            name = (glasses[e].get('name') if glasses and e < len(glasses) else None)
            gl.append({'e': e + 1, 'nd': round(nd, 5), 'vd': round(vd, 2), 'near': g['name'], 'dist': round(dist, 2),
                       'name': name, 'sld': bool(m.is_sld[e])})
        st = {'meta': _S['meta'], 'stage': stage, 'msg': msg, 'time': time.strftime('%H:%M:%S'), 'errf': errf,
              'geo': geometry(m, x), 'aper': m.aper_report(x), 'glasses': gl, 'hist': _S['hist'][-2000:], 'log': _S['log'][-120:], 'extra': _S['extra'],
              'metrics': {'efl': round(o['efl'], 4), 'y_stop': round(o['y_stop'], 3), 'back_et': round(m.back_et(x), 3), 'back': round(float(x[m.sl['bk']][0]), 3),
                          'seidel': {k: round(float(o[k]), 5) for k in ('TSA', 'TCO', 'TAS', 'SAS', 'PTB', 'AX', 'LAT')},
                          'tan_rms': [[f, None if p is None else round(p * 1000, 2), round(v * 100)] for f, _, p, v in m.ray_report(x)]}}
        p = os.path.join(_S['dir'], '_live.json'); tmp = p + '.tmp'
        json.dump(st, open(tmp, 'w', encoding='utf-8'), ensure_ascii=False)
        os.replace(tmp, p)
    except Exception as ex:                              # 发布失败不能影响优化
        _S['log'].append(f'[live] {type(ex).__name__}: {ex}')


if __name__ == '__main__':
    # 把某一级存下来的 json 发布一次（不跑优化也能在网页里看）：python live.py SIGMA85_s3.json --xlsx SIGMA85.xlsx
    import argparse, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import start_fix as sf
    ap = argparse.ArgumentParser(); ap.add_argument('json'); ap.add_argument('--xlsx', required=True)
    ap.add_argument('--codev', action='store_true', help='用同名 .len 让 CODE V 现算 MTF / 弥散斑'); a = ap.parse_args()
    J = json.load(open(a.json, encoding='utf-8')); d = sf.read_xlsx(a.xlsx)
    m = sf.Model(d, argparse.Namespace(**J['args']))
    init(os.path.dirname(os.path.abspath(a.json)), m, xlsx=os.path.abspath(a.xlsx), title=d['title'])
    kw = {}
    if a.codev:
        import cv_opt; from cv_session import CodeV
        with CodeV(workdir=os.path.dirname(os.path.abspath(a.json))) as cv:
            cv.cmd(f'RES "{os.path.splitext(os.path.abspath(a.json))[0]}.len"'); cv.cmd('SET VIG')
            kw = dict(spots=cv_opt.spot_report(cv), mtf=cv_opt.mtf_report(cv, m, dense=11))
    publish(J['x'], stage=f"第{J.get('stage', 1)}级(回放)", msg=os.path.basename(a.json), glasses=J.get('glasses'), force=True, **kw)
    print('已发布 →', os.path.join(_S['dir'], '_live.json'))
