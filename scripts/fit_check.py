#!/usr/bin/env python3
"""交付前的硬指标：把拟合弧压回原图，逐行量它到描边中心线的偏差。

    python3 fit_check.py lens.json diagram_crop.png [-v]

比看叠加图可靠得多 —— 肉眼分辨不了 1~2 px 的系统偏差，但那对应 R 偏 10% 以上。
输出每个面的平均/最大偏差（px）和全局平均。经验值：**全局平均 < 1 px 算合格**；
某个面平均 > 1.5 px 基本就是拟合出了问题（历史案例：trim_rim 把真实弧砍掉，R 偏 17%）。

注意：光轴虚线那几行会把描边打断，脚本已跳过 |y-ay| < 8 px 的行。
"""
import argparse, json, math
import numpy as np
from PIL import Image


def load(image, js):
    d = json.load(open(js))
    im = Image.open(image)
    if im.mode in ('RGBA', 'LA', 'P'):
        im = Image.alpha_composite(Image.new('RGBA', im.size, (255,) * 4), im.convert('RGBA'))
    arr = np.asarray(im.convert('RGB'))
    if d.get('crop'):
        wh, x, y = d['crop'].split('+'); w, h = wh.split('x')
        arr = arr[int(y):int(y) + int(h), int(x):int(x) + int(w)]
    if d.get('flip'):
        arr = arr[:, ::-1]
    return d, arr


def outline_mask(d, arr):
    """描边 = 既不是背景、也没被任何镜片用作填充色的颜色类。"""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from diag_extract import quantize
    cols = d.get('colors') or {}
    lab, cent = quantize(arr, len(cols) or 6, 28)
    fill = {e['color'] for e in d['elements']} | {d.get('bg', 0)}
    return np.isin(lab, [c for c in range(len(cent)) if c not in fill])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('json'); ap.add_argument('image')
    ap.add_argument('-v', '--verbose', action='store_true')
    a = ap.parse_args()
    d, arr = load(a.image, a.json)
    H, W = arr.shape[:2]
    om = outline_mask(d, arr)
    cen = []
    for y in range(H):
        xs = np.where(om[y])[0]; segs = []
        for x in xs:
            if segs and x - segs[-1][1] <= 2: segs[-1][1] = x
            else: segs.append([x, x])
        cen.append([(s + e) / 2 for s, e in segs if e - s >= 1])
    px = d['pxmm']; ay = d['axis_px']; x0 = d['surfaces'][0]['vertex']
    print(f"{'面':>3} {'R(mm)':>10} {'半径mm':>7} {'平均|Δ|':>7} {'最大|Δ|':>7}")
    tot = []
    for k, s in enumerate(d['surfaces'], 1):
        if s.get('stop'): continue
        R = s['R_mm']; semi = s.get('semi_fit_mm') or s['semi_mm']
        if semi is None: continue
        zv = (s['vertex'] - x0) / px; ds = []
        for t in np.linspace(-semi, semi, 80):
            yy = int(round(ay + t * px))
            if not (0 <= yy < H) or abs(yy - ay) < 8: continue
            if R is None: xp = x0 + zv * px
            else:
                r = abs(R); tt = min(abs(t), r * 0.9999)
                sag = r - math.sqrt(r * r - tt * tt)
                xp = x0 + (zv + (sag if R > 0 else -sag)) * px
            cc = cen[yy]
            if not cc: continue
            m = min(cc, key=lambda q: abs(q - xp))
            if abs(m - xp) < 25: ds.append(m - xp)
        ds = np.array(ds) if ds else np.array([0.0])
        tot.append(np.abs(ds).mean())
        flag = '  ←偏大' if np.abs(ds).mean() > 1.5 else ''
        print(f'{k:>3} {("INF" if R is None else f"{R:.2f}"):>10} {semi:>7.2f} '
              f'{np.abs(ds).mean():>7.2f} {np.abs(ds).max():>7.2f}{flag}')
    print(f'\n全部面平均 {np.mean(tot):.2f} px = {np.mean(tot)/px*1000:.0f} µm'
          f'   （<1 px 合格）')


if __name__ == '__main__':
    main()
