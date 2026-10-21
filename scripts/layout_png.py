#!/usr/bin/env python3
"""把某一级的结果画成 layout PNG：优化后的面轮廓 + 子午光线，**叠在原结构图上**，一眼看出片型差。

    python layout_png.py SIGMA85_start.json --xlsx SIGMA85.xlsx --dxf SIGMA85.dxf --image s85_base.jpg -o SIGMA85_start_layout.png

像素↔mm 的映射从 diag_build.py 写的 DXF 里的 IMAGE 实体恢复（插入点 + 像素尺寸），不用另存参数。
光线：各视场 d 线子午光扇，被结构图半口径挡掉的整条不画（=真实渐晕后的光束）。
"""
import argparse, json, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import start_fix as sf

COL = [(220, 30, 30), (20, 140, 20), (30, 60, 220), (150, 20, 120)]


def dxf_image_map(path):
    t = open(path, encoding='utf-8', errors='ignore').read().split('\n')
    for i in range(0, len(t) - 1, 2):
        if t[i].strip() == '0' and t[i + 1].strip() == 'IMAGE':
            rec = {}; j = i + 2
            while t[j].strip() != '0':
                rec.setdefault(t[j].strip(), t[j + 1].strip()); j += 2
            return float(rec['10']), float(rec['20']), float(rec['11']), float(rec['23'])   # x_ins, y_ins, mm/px, H
    raise SystemExit('DXF 里没有 IMAGE 实体')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('json'); ap.add_argument('--xlsx', required=True); ap.add_argument('--dxf', required=True)
    ap.add_argument('--image', required=True); ap.add_argument('-o', required=True)
    ap.add_argument('--nray', type=int, default=9); ap.add_argument('--title', default='')
    a = ap.parse_args()
    J = json.load(open(a.json, encoding='utf-8'))
    m = sf.Model(sf.read_xlsx(a.xlsx), argparse.Namespace(**J['args'])); x = np.array(J['x'])
    xi, yi, mmpx, H = dxf_image_map(a.dxf); k = 1 / mmpx
    c, t, n, dn, Aco = m.table(x); zv = np.r_[0, np.cumsum(t)]
    img = Image.open(a.image).convert('RGB')
    W = int((zv[-1] + 6 - xi) * k)
    can = Image.new('RGB', (max(W, img.width), img.height), 'white'); can.paste(img, (0, 0))
    can = Image.blend(can, Image.new('RGB', can.size, 'white'), 0.45)       # 底图淡显
    d = ImageDraw.Draw(can)
    P = lambda z, y: ((z - xi) * k, H - (y - yi) * k)

    # 光线
    fields = [0.0, 0.4, 0.7, 1.0]; rhos = np.linspace(-1, 1, a.nray); path = []
    ep_z = m.seidel(x)['ep_z']
    yimg, ok, clip = m.rays(x, fields, rhos, ep_z, path=path)
    good = ok.all(axis=-1) & (clip <= 1.0); good[0, :] = ok[0].all(axis=-1)
    for f in range(len(fields)):
        for r in range(a.nray):
            if not good[f, r]: continue
            z0, y0 = path[0][0][f, r], path[0][1][f, r]
            ang = np.arctan(fields[f] * m.pupil(x)[1])
            pts = [P(z0 - 12, y0 - 12 * np.tan(ang))] + [P(pz[f, r], py[f, r]) for pz, py in path]
            d.line(pts, fill=COL[f], width=2)
    # 面轮廓（优化后）
    ends = []
    for i in range(len(c)):
        if i < len(m.S) and m.S[i]['stop']:
            # 光阑行常常没有口径（图上没画、或由 F 数定）：用轴上边缘光线在光阑处的高度
            semi = float(m.S[i]['draw'] or m.S[i]['semi'] or m.seidel(x)['y_stop'])
            for sgn in (1, -1): d.line([P(zv[i], sgn * semi), P(zv[i], sgn * (semi + 4))], fill=(0, 0, 0), width=5)
            ends.append(None); continue
        semi = float(m.S[i]['draw'] or m.S[i]['semi']) if i < len(m.S) else 24.0
        ys = np.linspace(-semi, semi, 81); zs = zv[i] + sf.sag(c[i], ys, Aco[i])
        asph = i in m.ai
        d.line([P(z, y) for z, y in zip(zs, ys)], fill=(255, 120, 0) if asph else (0, 0, 0), width=3)
        ends.append((zs[-1], semi))
    for i in range(len(c) - 1):                          # 镜片边缘
        if n[i] > 1.0 and ends[i] and ends[i + 1]:
            (za, ha), (zb, hb) = ends[i], ends[i + 1]; hh = max(ha, hb)
            for sgn in (1, -1):
                d.line([P(za, sgn * ha), P(za, sgn * hh), P(zb, sgn * hh), P(zb, sgn * hb)], fill=(0, 0, 0), width=3)
    d.line([P(zv[-1], -22), P(zv[-1], 22)], fill=(0, 0, 0), width=4)      # 像面
    d.line([P(-5, 0), P(zv[-1] + 3, 0)], fill=(120, 120, 120), width=1)
    try: font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 38); small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 28)
    except Exception: font = small = ImageFont.load_default()
    o = m.seidel(x); rep = m.ray_report(x)
    d.text((30, 20), a.title or os.path.basename(a.json), fill=(0, 0, 0), font=font)
    d.text((30, 75), f"EFL {o['efl']:.3f}  F/{m.a.fno}  像高 {m.a.ih}   黑=优化后轮廓  橙=非球面候选面  淡底=原结构图", fill=(60, 60, 60), font=small)
    d.text((30, 115), '子午 RMS 横向像差 µm (CdF)：' + '  '.join(f'{f:.1f}H {("—" if p is None else f"{p * 1000:.1f}")} 通光{v * 100:.0f}%' for f, _, p, v in rep), fill=(60, 60, 60), font=small)
    # 牌号标注
    for g, (i0) in zip(J['glasses'], [i for i, s in enumerate(m.S) if s['glass']]):
        zc = (zv[i0] + zv[i0 + 1]) / 2; px, py = P(zc, -float(m.S[i0]['semi']) - 2 - (g['elem'] % 3) * 1.6)
        d.text((px - 40, py), f"{g['elem']}:{g.get('name') or g['nearest']}", fill=(0, 0, 160) if g['sld'] else (90, 90, 90), font=small)
    can.save(a.o); print('写出', a.o, can.size)


if __name__ == '__main__':
    main()
