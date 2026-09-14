#!/usr/bin/env python3
"""画一张「已知处方」的合成结构图，用来做往返精度测试：
   真值处方 → 渲染成厂商风格的 SVG → 跑 diag_extract → 比对反推出来的 R/D。

    python3 make_demo.py            # 生成 demo.svg 和 truth.json
"""
import json, math, os

# 一个古典库克三片式（Cooke triplet），单位 mm。semi 是机械半径。
PRESC = [   # R,        D,     glass?, semi
    ( 22.014,  3.259, True,  9.0),
    (-435.76,  6.008, False, 9.0),
    (-22.213,  1.000, True,  6.5),
    ( 20.291,  4.750, False, 6.5),
    ('STO',    0.750, False, 5.2),
    ( 79.684,  2.952, True,  8.0),
    (-18.395,  0.0,   False, 8.0),
]
PXMM, PAD = 40.0, 40.0
FILL = ['#cfd8e6', '#8fb3e0', '#cfd8e6']   # 三片的填充色


def sag(R, y):
    if R is None: return 0.0
    r = abs(R); y = min(abs(y), r * 0.999999)
    s = r - math.sqrt(r * r - y * y)
    return s if R > 0 else -s


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    z, surf = 0.0, []
    for R, D, glass, semi in PRESC:
        surf.append(dict(R=None if R == 'STO' else R, z=z, glass=glass, semi=semi,
                         stop=(R == 'STO')))
        z += D
    total = surf[-1]['z'] - surf[0]['z']
    semi_max = max(s['semi'] for s in surf)
    W = int((total + 2 * PAD / PXMM) * PXMM)
    H = int((2 * semi_max + 2 * PAD / PXMM) * PXMM)
    ox, ay = PAD, H / 2

    def P(s, y):                      # 面在半径 y 处的画布坐标
        return (ox + (s['z'] + sag(s['R'], y)) * PXMM, ay + y * PXMM)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}"><rect width="{W}" height="{H}" fill="white"/>']
    gi = 0
    for i, s in enumerate(surf[:-1]):
        if s['stop'] or not s['glass']: continue
        b = surf[i + 1]
        semi = min(s['semi'], b['semi'])
        ys = [-semi + 2 * semi * k / 120 for k in range(121)]
        pts = [P(s, y) for y in ys] + [P(b, y) for y in reversed(ys)]
        d = 'M ' + ' L '.join(f'{x:.2f} {y:.2f}' for x, y in pts) + ' Z'
        out.append(f'<path d="{d}" fill="{FILL[gi % len(FILL)]}" stroke="#111" stroke-width="2.4"/>')
        gi += 1
    st = next(s for s in surf if s['stop'])
    x = ox + st['z'] * PXMM
    out.append(f'<line x1="{x:.2f}" y1="{ay - st["semi"]*PXMM:.2f}" x2="{x:.2f}" '
               f'y2="{ay + st["semi"]*PXMM:.2f}" stroke="#111" stroke-width="3"/>')
    out.append(f'<line x1="0" y1="{ay}" x2="{W}" y2="{ay}" stroke="#111" '
               f'stroke-width="1.5" stroke-dasharray="14 10"/>')
    out.append('</svg>')
    open(os.path.join(here, 'demo.svg'), 'w').write('\n'.join(out))

    truth = dict(pxmm=PXMM, total_length_mm=round(total, 3),
                 surfaces=[dict(R=s['R'], semi=s['semi'], stop=s['stop'],
                                D=round(surf[i + 1]['z'] - s['z'], 3) if i + 1 < len(surf) else 0.0)
                           for i, s in enumerate(surf)])
    json.dump(truth, open(os.path.join(here, 'truth.json'), 'w'), ensure_ascii=False, indent=1)
    print(f'demo.svg  {W}x{H}px   总长(首面顶点→末面顶点) {total:.3f} mm')


if __name__ == '__main__':
    main()
