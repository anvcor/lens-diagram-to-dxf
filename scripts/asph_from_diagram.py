#!/usr/bin/env python3
"""asph_from_diagram.py -- 专利缺非球面系数时，从结构图量非球面初值。

前提：专利印了 R/D（面表），只是没印非球面系数；结构图是按比例画的 CAD 图。
做法：
  1. 用印刷 R/D 算各面顶点，在光轴附近（h=7..14 px）量墨线中心，最小二乘标定
     px/mm 和 x 偏移（只用印刷球面，非球面在轴附近也近似球面）。
  2. 沿每个面逐行描墨线中心，求它相对「印刷顶点球面」的偏离 Δz(h)（mm，+ 指向像方）。
     球面的偏离 rms 就是这张图的读数噪声（实测 0.01~0.03 mm）。
  3. 非球面的 Δz(h) 用 K=0 偶次多项式 A4 h^4 + A6 h^6 (+ A8 h^8) 拟合，作为优化初值。

用法：
  python3 asph_from_diagram.py lens.json presc.json -o asph.json [--seq asph_snippet.seq]

  lens.json  : diag_extract.py 的输出（取 image/crop/flip/pxmm/axis_px 和第一面顶点 px 当初值）
  presc.json : 专利面表，例如
      {"surfaces": [{"id": "2", "R": 63.4831, "D": 2.0}, ..., {"id": "STO", "R": null, "D": 1.13}, ...],
       "asph": ["17", "18", "28", "29"],
       "hmax": {"17": 17.3, "28": 14.1}}          # 可选：拟合截止半高(mm)，默认用描到的最大高度
     R = null 或 0 表示平面；D 为该面之后的间隔（INF 态）。第一个面就是图上的第一面。

输出：asph.json（标定、逐面描线、偏离摘要、拟合系数），stdout 打印每面偏离表；
     --seq 给出 CODE V 的 ASP 片段（面号按 presc 顺序 1..N，自己再对一下 CODE V 面号）。
"""
import argparse, json, math, os, sys
import numpy as np
from PIL import Image


def load_gray(lens, base):
    path = lens['image']
    if not os.path.isabs(path):
        path = os.path.join(base, path)
    im = Image.open(path)
    if im.mode in ('RGBA', 'LA', 'P'):
        im = im.convert('RGBA')
        bg = Image.new('RGBA', im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    im = im.convert('L')
    if lens.get('crop'):
        wh, x, y = lens['crop'].split('+')[0], *map(int, lens['crop'].split('+')[1:])
        w, h = map(int, wh.split('x'))
        im = im.crop((x, y, x + w, y + h))
    if lens.get('flip'):
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    return np.array(im).astype(float)


def sag(r, h):
    if r is None or r == 0 or math.isinf(r):
        return 0.0
    c = 1.0 / r
    q = 1 - c * c * h * h
    if q < 0:
        return float('nan')
    return c * h * h / (1 + math.sqrt(q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('lens')
    ap.add_argument('presc')
    ap.add_argument('-o', default='asph.json')
    ap.add_argument('--seq', help='写 CODE V ASP 片段')
    ap.add_argument('--dark', type=float, default=170, help='墨线灰度阈值（<阈值算墨）')
    ap.add_argument('--axis', type=float, help='覆盖 lens.json 的光轴行')
    ap.add_argument('--nterm', type=int, default=3, help='偶次项个数：2=A4,A6  3=A4..A8')
    a = ap.parse_args()

    lens = json.load(open(a.lens, encoding='utf-8'))
    P = json.load(open(a.presc, encoding='utf-8'))
    g = load_gray(lens, os.path.dirname(os.path.abspath(a.lens)))
    H, W = g.shape
    dark = g < a.dark
    AX = a.axis if a.axis is not None else lens['axis_px']

    S = P['surfaces']
    ids = [str(s['id']) for s in S]
    R = {str(s['id']): (s['R'] if s['R'] not in (None, 0) else None) for s in S}
    V, z = {}, 0.0
    for s in S:
        V[str(s['id'])] = z
        z += s['D'] or 0.0
    asph = [str(k) for k in P.get('asph', [])]
    sph = [k for k in ids if k not in asph and k.upper() not in ('STO', 'STOP')]

    def ink_center(row, xc, win):
        lo = int(max(0, math.floor(xc - win)))
        hi = int(min(W - 1, math.ceil(xc + win)))
        seg = dark[row, lo:hi + 1]
        if not seg.any():
            return None
        idx = np.where(seg)[0] + lo
        runs, st, pv = [], idx[0], idx[0]
        for i in idx[1:]:
            if i != pv + 1:
                runs.append((st, pv)); st = i
            pv = i
        runs.append((st, pv))
        best = None
        for p, q in runs:
            wt = 255 - g[row, p:q + 1]
            c = (np.arange(p, q + 1) * wt).sum() / wt.sum()
            if best is None or abs(c - xc) < best[0]:
                best = (abs(c - xc), c, q - p + 1)
        return best[1], best[2]

    # ---- 1. 标定 px/mm 与 x 偏移 ----
    s = lens['pxmm']
    x0 = lens['surfaces'][0]['vertex']
    res = np.array([0.0])
    for _ in range(5):
        A, b = [], []
        for k in sph:
            for hp in range(7, 15):
                for sg in (1, -1):
                    row = int(AX + sg * hp)
                    if not 0 <= row < H:
                        continue
                    zz = V[k] + sag(R[k], hp / s)
                    if math.isnan(zz):
                        continue
                    r = ink_center(row, x0 + s * zz, 3.0)
                    if r is None or r[1] > 6:
                        continue
                    A.append([1.0, zz]); b.append(r[0])
        if len(b) < 10:
            sys.exit('[错误] 轴附近匹配到的墨线点太少：检查 lens.json 的 pxmm / 第一面顶点 / 光轴行，或 --dark')
        A, b = np.array(A), np.array(b)
        (x0, s), *_ = np.linalg.lstsq(A, b, rcond=None)
        res = b - A @ np.array([x0, s])
    print(f'[标定] {s:.4f} px/mm  x0={x0:.2f}px  残差 rms {np.sqrt((res**2).mean()):.2f}px  ({len(b)} 点)')

    # ---- 2. 逐面描墨线 ----
    def trace(k, sg):
        out, dev, miss, hp = [], 0.0, 0, 7
        while True:
            row = int(round(AX + sg * hp))
            if not 0 <= row < H:
                break
            hmm = hp / s
            zz = V[k] + sag(R[k], hmm)
            if math.isnan(zz):
                break
            xp = x0 + s * zz
            r = ink_center(row, xp + dev, 2.5)
            if r is None or r[1] > 7:
                miss += 1
                if miss > 2:
                    break
            else:
                miss = 0
                d = r[0] - xp
                if out and abs(d - dev) > 1.6:   # 跳到别的线上了
                    break
                dev = d
                out.append((hmm, d / s))
            hp += 1
        return out

    out = dict(pxmm=s, x0=x0, axis=AX, calib_rms_px=float(np.sqrt((res**2).mean())), surf={}, fit={})
    print(f'{"面":>5} {"类型":4} {"R":>10} {"h_max":>6}  Δz@0.5/0.7/0.85/1.0 h_max (mm)   rms')
    sph_rms = []
    for k in ids:
        if k.upper() in ('STO', 'STOP'):
            continue
        pts = trace(k, 1) + trace(k, -1)
        out['surf'][k] = pts
        if not pts:
            print(f'{k:>5} 描不到')
            continue
        P2 = np.array(pts)
        hm = P2[:, 0].max()
        cols = []
        for f in (0.5, 0.7, 0.85, 1.0):
            sel = np.abs(P2[:, 0] - f * hm) < 0.25
            cols.append(f'{P2[sel, 1].mean():+.3f}' if sel.any() else '   -  ')
        rms = float(np.sqrt((P2[:, 1] ** 2).mean()))
        if k not in asph:
            sph_rms.append(rms)
        rr = 'INF' if R[k] is None else f'{R[k]:.3f}'
        print(f'{k:>5} {"ASP" if k in asph else "   ":4} {rr:>10} {hm:6.2f}  {" ".join(cols)}   {rms:.3f}')
    if sph_rms:
        print(f'[读数噪声] 球面偏离 rms 中位数 {np.median(sph_rms):.3f} mm —— 非球面偏离小于这个量级就别信')
        print('[提示] 个别球面 rms 很大（>0.3 mm）多半是描线在边缘跳到了相邻面的墨线上（两面靠得近），'
              '看 0.5/0.7 两列是否接近 0 即可；若 0.5 处就大，说明印刷 R/D 与图对不上，先查印错')

    # ---- 3. 非球面偶次拟合 ----
    hcap = {str(k): v for k, v in P.get('hmax', {}).items()}
    for k in asph:
        P2 = np.array(out['surf'].get(k) or [])
        if len(P2) < 8:
            print(f'[警告] 非球面 {k} 描到的点太少，跳过拟合')
            continue
        hx = hcap.get(k, P2[:, 0].max())
        P2 = P2[(P2[:, 0] >= 2.5) & (P2[:, 0] <= hx)]
        h, zz = P2[:, 0], P2[:, 1]
        n = a.nterm
        M = np.stack([h ** (2 * i) for i in range(2, 2 + n)], 1)
        sc = np.array([hx ** (2 * i) for i in range(2, 2 + n)])
        c, *_ = np.linalg.lstsq(M / sc, zz, rcond=None)
        c = c / sc
        r = zz - M @ c
        edge = float(sum(c[i] * hx ** (2 * i + 4) for i in range(n)))
        out['fit'][k] = dict(coef=[float(x) for x in c], hmax=float(hx), rms=float(np.sqrt((r**2).mean())), edge_dz=edge)
        print(f'[拟合] 面 {k}: ' + ' '.join(f'A{4+2*i}={c[i]:.4e}' for i in range(n)) +
              f'   rms {np.sqrt((r**2).mean()):.4f} mm   边缘 Δz {edge:+.3f} mm @ h={hx:.1f}')

    json.dump(out, open(a.o, 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
    print(f'[输出] {a.o}')

    if a.seq:
        L = ['! aspheric start values measured from the diagram (K=0, even terms)',
             '! surface numbers = position in presc.json (1..N); check against your CODE V numbering']
        for k, f in out['fit'].items():
            sn = ids.index(k) + 1
            terms = ['A', 'B', 'C', 'D', 'E'][:len(f['coef'])]
            L.append(f'ASP S{sn}; K S{sn} 0; ' + '; '.join(f'{t} S{sn} {v:.6e}' for t, v in zip(terms, f['coef']))
                     + f'   ! presc id {k}, edge dz {f["edge_dz"]:+.3f} mm @ h={f["hmax"]:.1f}')
        open(a.seq, 'w', newline='\r\n').write('\n'.join(L) + '\n')
        print(f'[输出] {a.seq}')


if __name__ == '__main__':
    main()
