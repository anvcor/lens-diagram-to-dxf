#!/usr/bin/env python3
"""一步到位：结构图反推的 xlsx → 第 1 级起点修复 → 第 2 级 CODE V AUT → 第 3 级吸附实牌号。

    python run_all.py E:\\Download\\SIGMA85\\SIGMA85.xlsx --efl 85 --fno 1.25 --ih 21.6 [--from 2] [--mxc 45]

产出都写在 xlsx 所在目录：<前缀>_start / _s2 / _s3 的 .seq .len .json。任何一级失败立即停（绝不拿上一轮的旧文件接着跑）。
想边跑边看：另开一个终端 python viewer.py <目录>，浏览器开 http://localhost:8765。
"""
import argparse, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(script, *args):
    cmd = [sys.executable, os.path.join(HERE, script), *map(str, args)]
    print('\n>>>', ' '.join(cmd), flush=True)
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    r = subprocess.run(cmd, env=env)
    if r.returncode: sys.exit(f'✗ {script} 失败（退出码 {r.returncode}），流水线停止')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('xlsx'); ap.add_argument('--efl', type=float, required=True); ap.add_argument('--fno', type=float, required=True)
    ap.add_argument('--ih', type=float, default=21.6); ap.add_argument('--from', dest='start', type=int, default=1, choices=(1, 2, 3))
    ap.add_argument('--starts', type=int, default=10); ap.add_argument('--keep', type=int, default=3)
    ap.add_argument('--mxc', type=int, default=45); ap.add_argument('--k', type=int, default=1, help='第 3 级每片试几个候选牌号')
    a, extra = ap.parse_known_args()                     # 其余参数原样传给第 1 级（--efl-tol --dist-max --sld-tag …）
    x = os.path.abspath(a.xlsx); p = os.path.splitext(x)[0]
    if a.start <= 1:
        for f in (p + '_start.json', p + '_s2.json', p + '_s3.json'):
            if os.path.exists(f): os.remove(f)
        run('start_fix.py', x, '-o', p + '_start', '--efl', a.efl, '--fno', a.fno, '--ih', a.ih, '--starts', a.starts, '--keep', a.keep, *extra)
    if a.start <= 2:
        run('cv_opt.py', p + '_start.json', '--xlsx', x, '-o', p + '_s2', '--mxc', a.mxc)
    run('cv_snap.py', p + '_s2.json', '--xlsx', x, '-o', p + '_s3', '--k', a.k)
    print('\n✓ 完成：', p + '_s3.len')


if __name__ == '__main__':
    main()
