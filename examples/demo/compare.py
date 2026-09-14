#!/usr/bin/env python3
"""比对合成示例的「真值处方」与反推结果 —— 仓库自带的往返精度自测。

    python3 make_demo.py
    python3 ../../scripts/diag_extract.py demo.svg -o demo.json --length 18.719 --svg-width 1800
    python3 compare.py
"""
import json, os

here = os.path.dirname(os.path.abspath(__file__))
T = json.load(open(os.path.join(here, 'truth.json')))['surfaces']
G = json.load(open(os.path.join(here, 'demo.json')))['surfaces']
assert len(T) == len(G), f'面数对不上：真值 {len(T)}，反推 {len(G)}'

print(f"{'面':>3} {'R 真值':>10} {'R 反推':>10} {'误差':>8} | "
      f"{'D 真值':>8} {'D 反推':>8} {'误差':>8}")
dr, dd = [], []
for i, (t, g) in enumerate(zip(T, G), 1):
    name = 'STO' if t['stop'] else str(i)
    if t['R'] is None or g['R_mm'] is None:
        rt, rg, er = 'INF', 'INF', ''
    else:
        rt, rg = f"{t['R']:.3f}", f"{g['R_mm']:.3f}"
        e = (g['R_mm'] - t['R']) / abs(t['R']) * 100
        er = f'{e:+.2f}%'; dr.append(abs(e))
    e2 = g['D'] - t['D']; dd.append(abs(e2))
    print(f'{name:>3} {rt:>10} {rg:>10} {er:>8} | '
          f"{t['D']:>8.3f} {g['D']:>8.3f} {e2:>+8.3f}")
print(f'\n曲率半径：最大误差 {max(dr):.2f}%，平均 {sum(dr)/len(dr):.2f}%')
print(f'厚度/间隔：最大误差 {max(dd):.3f} mm，平均 {sum(dd)/len(dd):.3f} mm')
