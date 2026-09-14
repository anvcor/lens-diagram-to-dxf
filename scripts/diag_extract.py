#!/usr/bin/env python3
"""镜头结构图（PNG/JPG/SVG）→ 逐面数据 JSON + 核对叠加图。

    python3 diag_extract.py diagram.png -o lens.json [--crop WxH+X+Y] \
        [--length 75.21 | --diameter 30.11 | --pxmm 12.3] [--axis 210] \
        [--min-height 0.25] [--flip] [--overlay ov.png]

流程（全部自动，任何一步都能用参数手动覆盖）：
 1. 颜色聚类：背景 = 最大色块；其余颜色各自做连通域 → 每个连通域 = 一片镜片（或光阑、文字）。
    粘合双胶合两片颜色不同就天然分开；同色粘合靠中间的描边线分开。
 2. 光轴：每片镜片上下对称，取各片 (ymin+ymax)/2 的中位数；--axis 可强制。
 3. 每片镜片逐行取最左/最右像素 → 左右两条轮廓，先剔掉上下的平边（磨边），
    再按圆心落在光轴上的约束做最小二乘圆拟合：x²+(y-ay)² = 2·cx·x + (R²-cx²)。
    近乎竖直的轮廓判为平面（R=∞）。拟合残差大（>1px）的面在 note 里标「可能非球面」。
 4. 比例尺：--length（首面顶点到末面顶点 mm）/ --diameter（最大镜片直径 mm）/ --pxmm 三选一；
    都不给就 1 px = 1 单位并告警。
 5. 按顶点 x 排序生成面表：厚度 = 后顶点 - 前顶点；空气间隔 = 下一片前顶点 - 本片后顶点；
    间隔 < 0.6 px 判为胶合，两条拟合弧合并成一个面。细长竖线（宽≤4px、高≥30%）判为光阑 STO。
 6. 输出 JSON（供 diag_build.py 生成 xlsx/dxf/spec.json）+ 叠加图（拟合弧画回原图，肉眼核对）。

SVG 输入会先用 cairosvg 栅格化到 --svg-width（默认 4000 px）再走同一流程。
"""
import argparse, json, math, os, sys
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


def load_image(path, svg_width):
    if path.lower().endswith('.svg'):
        import cairosvg, io
        png = cairosvg.svg2png(url=path, output_width=svg_width, background_color='white')
        im = Image.open(io.BytesIO(png)).convert('RGB')
    else:
        im = Image.open(path).convert('RGB')
    return im


def parse_crop(s):
    # WxH+X+Y
    wh, x, y = s.split('+')
    w, h = wh.split('x')
    return int(x), int(y), int(w), int(h)


def quantize(arr, k, merge=28):
    """简易 k-means 颜色聚类（灰度+色度），返回 label 图和各类中心。"""
    from scipy.cluster.vq import kmeans2
    px = arr.reshape(-1, 3).astype(np.float32)
    sub = px[np.random.RandomState(0).choice(len(px), min(len(px), 60000), replace=False)]
    cent, _ = kmeans2(sub, k, minit='++', seed=0)
    # 合并相距 < merge 的中心（线稿只有 2~3 种颜色，硬分 6 类会把抗锯齿灰阶拆成假类）
    keep = []
    for c in cent:
        if all(np.linalg.norm(c - k2) >= merge for k2 in keep):
            keep.append(c)
    cent = np.array(keep)
    d = ((px[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
    lab = d.argmin(1).reshape(arr.shape[:2])
    return lab, cent


def fit_circle(xs, ys, ay):
    """圆心约束在 y=ay 的最小二乘圆。返回 (cx, R, rms)。"""
    yy = ys - ay
    A = np.c_[2 * xs, np.ones_like(xs)]
    b = xs ** 2 + yy ** 2
    (cx, k), *_ = np.linalg.lstsq(A, b, rcond=None)
    R2 = k + cx ** 2
    if R2 <= 0:
        return None
    R = math.sqrt(R2)
    res = np.sqrt((xs - cx) ** 2 + yy ** 2) - R
    return cx, R, float(np.sqrt(np.mean(res ** 2)))


def _branch_x(cx, R, ys, ay, left):
    """圆心在 (cx, ay)、半径 R 的圆，在各行 y 处的 x（left=True 取左支）。"""
    dy = np.clip(np.abs(ys - ay), 0, R * 0.999999)
    off = np.sqrt(R * R - dy * dy)
    return cx - off if left else cx + off


def fit_surface(xs, ys, ay, side, flat_span=2.5, seed_frac=0.5, tol_min=1.5):
    """稳健拟合一条面轮廓。返回 dict(R 带符号, vertex, rms, flat, semi_fit, n)。
    符号约定：光从左到右，圆心在面右侧为正。

    **不要再用「x 连续几行不变就当磨边砍掉」那套**（旧的 trim_rim）：弱曲率面的 dx/dy 本来
    就小于 0.5 px/行（R=840px、半高 280px 时边缘处只有 0.33），结果把 40% 的真实弧当磨边砍了，
    剩下的矢高只有 15 px，拟合出来的 R 偏 17%（实测 E7 后表面：真值 ~840px，砍完拟成 996px）。

    改成种子 + 生长：靠近光轴的一半一定是真实面，拿它起拟；再把符合这个圆的点逐轮吸收进来。
    机械磨边是一段横平边，离圆很远，自然留在外面当离群点。
    `semi_fit` = 最终入选点的最大半径，也就是这个面在图上真正画到哪儿为止。
    """
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    if len(xs) < 8:
        return None
    dy = np.abs(ys - ay); h = dy.max()
    if h <= 0:
        return None
    seed = dy <= seed_frac * h
    if seed.sum() < 6:
        seed = np.ones(len(xs), bool)
    inl = seed.copy()
    left = None
    for _ in range(6):
        f = fit_circle(xs[inl], ys[inl], ay)
        if f is None:
            break
        cx, R, rms = f
        left = xs[inl].mean() < cx
        pred = _branch_x(cx, R, ys, ay, left)
        tol = max(tol_min, 3.0 * max(rms, 0.3))
        new = (np.abs(xs - pred) <= tol) | seed
        if new.sum() == inl.sum() and bool((new == inl).all()):
            break
        inl = new
    f = fit_circle(xs[inl], ys[inl], ay)
    if f is None:
        return None
    cx, R, rms = f
    left = xs[inl].mean() < cx
    span = float(xs[inl].max() - xs[inl].min())
    # 取 99 分位而不是最大值：机械磨边那一两行偶尔会落进容差里，用最大值会让弧多画出去一截
    semi_fit = float(np.percentile(np.abs(ys[inl] - ay), 99))
    if span < flat_span:
        return dict(R=None, vertex=float(np.median(xs[inl])), rms=0.0, flat=True,
                    n=int(inl.sum()), span=span, semi_fit=semi_fit)
    vertex = cx - R if left else cx + R
    signed = R if cx > vertex else -R
    return dict(R=signed, vertex=float(vertex), rms=float(rms), flat=False,
                n=int(inl.sum()), cx=float(cx), span=span, semi_fit=semi_fit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('image'); ap.add_argument('-o', required=True)
    ap.add_argument('--crop', help='WxH+X+Y，先裁到只剩镜头图')
    ap.add_argument('--svg-width', type=int, default=4000)
    ap.add_argument('--k', type=int, default=6, help='颜色聚类数（背景+描边+若干填充色）')
    ap.add_argument('--min-height', type=float, default=0.25, help='镜片最小高度（相对最高镜片）')
    ap.add_argument('--axis', type=float, help='强制光轴像素 y（裁剪后坐标）')
    ap.add_argument('--length', type=float, help='首面顶点→末面顶点 距离 (mm)')
    ap.add_argument('--diameter', type=float, help='最大镜片直径 (mm)')
    ap.add_argument('--pxmm', type=float, help='每 mm 像素数')
    ap.add_argument('--flip', action='store_true', help='物方在右侧时先水平翻转')
    ap.add_argument('--cement-gap', type=float, default=None, help='判为胶合的最大顶点间隙 (px)，默认 2*line_dilate+2')
    ap.add_argument('--overlay', help='叠加核对图输出路径')
    ap.add_argument('--bg', help='强制背景色 R,G,B')
    ap.add_argument('--min-frac', type=float, default=0.05, help='分片最小面积占其所在镜片整体的比例')
    ap.add_argument('--bridge', type=int, default=9, help='竖向闭运算核高度(px)，用来桥接穿过镜片的光轴线')
    ap.add_argument('--debug', help='调试图前缀：写 <前缀>_parts.png / _body.png')
    ap.add_argument('--line-dilate', type=int, default=0, help='描边膨胀次数（接上抗锯齿断点；0=不膨胀）')
    ap.add_argument('--merge', type=float, default=28, help='颜色中心合并距离(RGB 欧氏)')
    ap.add_argument('--axis-span', type=float, default=0.3, help='判定光轴线：线条像素占该行宽度的比例阈值')
    ap.add_argument('--min-thick', type=float, default=0.4, help='中心厚度小于此值(mm)且两侧都贴着邻片的背景色区域视为围起来的空气间隔')
    ap.add_argument('--drop', help='丢掉指定分片（按 x 排序的 1 起序号，逗号分隔）——用于被描边围起来的空气间隔、图例色块等误识别')
    ap.add_argument('--tags', help='颜色类→标记，如 "1=ED,5=HR,3=ASPH"（类号看 [颜色] 输出）；写进 JSON 供建表用')
    ap.add_argument('--stop-aspect', type=float, default=0.025, help='宽/最高镜片高 小于此值的竖条视为光阑线而非平板')
    ap.add_argument('--keep-bg', action='store_true', help='背景色分片一律当镜片，不做空气楔判定')
    ap.add_argument('--rim-min', type=float, default=4, help='镜片边缘的最小像素宽度；顶/底都窄于它的背景色分片判为空气楔')
    ap.add_argument('--rim-frac', type=float, default=0.03, help='同上，相对中心厚度的比例，取两者较大值')
    ap.add_argument('--flat-span', type=float, default=2.5, help='轮廓 x 跨度小于此值(px)判为平面——描边越粗噪声越大，可适当调高')
    ap.add_argument('--outline', help='强制线条颜色类号，逗号分隔（默认自动）')
    a = ap.parse_args()
    if a.cement_gap is None: a.cement_gap = 2 * a.line_dilate + 2

    im = load_image(a.image, a.svg_width)
    ox = oy = 0
    if a.crop:
        ox, oy, w, h = parse_crop(a.crop)
        im = im.crop((ox, oy, ox + w, oy + h))
    if a.flip:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    arr = np.asarray(im)
    H, W = arr.shape[:2]

    lab, cent = quantize(arr, a.k, a.merge)
    a.k = len(cent)
    counts = np.bincount(lab.ravel(), minlength=a.k)
    bg = int(counts.argmax())
    if a.bg:
        bgc = np.array([float(v) for v in a.bg.split(',')])
        bg = int(((cent - bgc) ** 2).sum(1).argmin())
    print(f'[颜色] 中心={[tuple(int(v) for v in c) for c in cent]} 背景=类{bg} {tuple(int(v) for v in cent[bg])}',
          file=sys.stderr)

    # 线条类：3x3 开运算能抹掉大半像素的颜色类（描边、光阑线、光轴虚线、文字）
    line_cls = set()
    if a.outline:
        line_cls = {int(v) for v in a.outline.split(',')}
    else:
        for c in range(a.k):
            if c == bg: continue
            m = lab == c
            if m.sum() == 0: continue
            kept = ndimage.binary_opening(m, structure=np.ones((3, 3))).sum()
            if kept < 0.4 * m.sum():
                line_cls.add(c)
    print(f'[线条类] {sorted(line_cls)}', file=sys.stderr)

    nonbg = lab != bg
    line_m = np.isin(lab, list(line_cls))
    # 镜片整体 = 描边围起来的区域（含空心轮廓镜片）
    # 描边先膨胀 --line-dilate 次把抗锯齿断点接上，再填洞；分片时用膨胀后的描边当分隔
    # 先 3x3 闭运算接上 1px 的抗锯齿断点（不增加线宽），再按需膨胀
    line_d = ndimage.binary_closing(line_m, structure=np.ones((3, 3)))
    if a.line_dilate:
        line_d = ndimage.binary_dilation(line_d, structure=np.ones((3, 3)), iterations=a.line_dilate)
    body = ndimage.binary_fill_holes(nonbg | line_d)
    if body.mean() > 0.75:
        print(f'[警告] 填洞后 {body.mean():.0%} 的画面都被当成镜片——多半是图有外框/边界线把背景围住了，'
              '请把 --crop 收到边框以内（或去掉边框）再跑', file=sys.stderr)
    inner = body & ~line_d
    # 光轴线：横向贯穿大半幅面的细线所在的行（虚线也算）
    rows = np.where(line_m.sum(1) > a.axis_span * W)[0]
    axis_band = None; axis_line_y = None
    if len(rows):
        axis_line_y = float(np.median(rows))
        rows = rows[np.abs(rows - axis_line_y) < 6]
        axis_band = np.zeros((H, W), bool); axis_band[rows.min() - 2: rows.max() + 3] = True
        print(f'[光轴线] 行 {rows.min()}..{rows.max()} → y={axis_line_y:.1f}', file=sys.stderr)
    # 在 body 内按填充色分片（空心镜片的内部就是背景色）
    # 镜片整体的连通域要去掉细线（光轴线会把所有镜片串成一个）
    kk = 2 * a.line_dilate + 5  # 开运算核要比膨胀后的线宽大，否则光轴线把所有镜片串成一个
    body_core = ndimage.binary_opening(body, structure=np.ones((kk, kk)))
    body_lab, nb = ndimage.label(body_core)
    body_area = ndimage.sum(body_core, body_lab, range(1, nb + 1))
    comps = []
    for c in range(a.k):
        if c in line_cls: continue
        m = inner & (lab == c)
        if axis_band is not None:
            # 只在光轴线所在的几行上做竖向闭运算，把被光轴线切成上下两半的镜片接回去
            m = m | (ndimage.binary_closing(m, structure=np.ones((a.bridge, 1))) & body & axis_band)
        cl, n = ndimage.label(m)
        if n == 0: continue
        sizes = ndimage.sum(m, cl, range(1, n + 1))
        objs = ndimage.find_objects(cl)
        for i, sz in enumerate(sizes, 1):
            sl = objs[i - 1]
            bl = body_lab[sl][cl[sl] == i]; bl = bl[bl > 0]
            bid = int(np.bincount(bl).argmax()) if len(bl) else 0
            # 抗锯齿过渡带、小杂色块：占所在 body 面积不足 --min-frac 的一律并入邻片
            if bid == 0 or sz < a.min_frac * body_area[bid - 1] or sz < 30: continue
            comps.append(dict(color=c, lab_img=cl, id=i, area=int(sz)))
    print(f"[分片] {len(comps)} 个: {sorted(c['area'] for c in comps)}", file=sys.stderr)
    # 把描边像素归还给最近的分片
    seed = np.zeros((H, W), np.int32)
    for j, c in enumerate(comps, 1):
        seed[c['lab_img'] == c['id']] = j
    dist, idx = ndimage.distance_transform_edt(seed == 0, return_indices=True)
    # 只归还紧贴分片的描边像素（距离 ≤ 线宽），否则光轴线会顺着空气间隔把镜片轮廓拖出去
    full = np.where(dist <= 2 * a.line_dilate + 2.5, seed[idx[0], idx[1]], 0) * body
    if a.debug:
        rng = np.random.RandomState(1); pal = np.r_[[[0, 0, 0]], rng.randint(60, 255, (len(comps) + 1, 3))].astype(np.uint8)
        Image.fromarray(pal[full]).save(a.debug + '_parts.png')
        Image.fromarray((body * 255).astype(np.uint8)).save(a.debug + '_body.png')
    for j, c in enumerate(comps, 1):
        m = full == j
        ys, xs = np.where(m)
        c.update(x0=int(xs.min()), x1=int(xs.max()) + 1, y0=int(ys.min()), y1=int(ys.max()) + 1)
        c['h'], c['w'] = c['y1'] - c['y0'], c['x1'] - c['x0']
        c['mask'] = m[c['y0']:c['y1'], c['x0']:c['x1']]
        c.pop('lab_img')
    hmax = max(c['h'] for c in comps)
    thin = max(4, a.stop_aspect * hmax)
    elems = [c for c in comps if c['h'] >= a.min_height * hmax and c['w'] > thin
             and c['area'] / (c['h'] * c['w']) > 0.2]
    # 背景色分片：可能是"未填充的镜片"，也可能是"两片镜片之间被描边围起来的空气"。
    # 判据是边缘（rim）：镜片一定画出有限厚度的边缘，最上/最下一行有实宽；
    # 空气楔两端是两条面弧的交点，是真正的尖点（1~3 px）。颜色不是背景的一律是玻璃。
    def tips(c):
        m = c['mask']; mid = int(m[m.shape[0] // 2].sum())
        return int(m[0].sum()), int(m[-1].sum()), mid

    air = set()
    if not a.keep_bg:
        for k_, c in enumerate(elems):
            t, b, mid = tips(c)
            lim = max(a.rim_min, a.rim_frac * mid)
            if c['color'] == bg and mid > 8 and t <= lim and b <= lim:
                print(f"[丢弃] x={c['x0']}..{c['x1']} 上下两端是尖点（顶宽 {t}/{b} px，中宽 {mid} px）"
                      f"→ 判为被围起来的空气，不是镜片（要保留用 --keep-bg）", file=sys.stderr)
                air.add(k_)
    # 空气楔要留到第二遍归属结束才剔除 —— 它得当种子挡住两侧镜片，否则镜片会长进空气里
    # ——— 描边是透镜边缘本身，只是画得粗：真实面在描边的中心线上，不是填充区的内边 ———
    # 描边宽 = 那些「既不是背景、也没被任何镜片用作填充色」的颜色类构成的笔画宽度
    fill_cls = {c['color'] for c in elems} | {bg}
    _om = np.isin(lab, [c for c in range(a.k) if c not in fill_cls]) & body
    linew = 2 * float(np.percentile(ndimage.distance_transform_edt(_om)[_om], 95)) if _om.any() else 2.0
    print(f'[描边宽] {linew:.1f} px —— 各片按它的一半外扩到中心线', file=sys.stderr)
    # 第二遍归属：只拿镜片分片当种子（第一遍把白描边自己也当了种子，各片才停在填充边上），
    # 让每片长到描边中心。两片共用一条描边时，最近邻正好把它从中间劈开 → 两片恰好相接，不留假缝、不重叠。
    grow = linew / 2
    seed2 = np.zeros((H, W), np.int32)
    for j, c in enumerate(elems, 1):
        seed2[c['y0']:c['y1'], c['x0']:c['x1']][c['mask']] = j
    d2, i2 = ndimage.distance_transform_edt(seed2 == 0, return_indices=True)
    # 只许长进描边像素里，不许跨过黑色空气 —— 否则相邻两片会在空气隙正中间分界，
    # 两条独立的描边各自的位置就丢了（实测会把一个真平面拉成 R=-115）
    full2 = np.where((d2 <= grow) & (_om | (seed2 > 0)), seed2[i2[0], i2[1]], 0) * body
    for j, c in enumerate(elems, 1):
        m = full2 == j
        ys, xs = np.where(m)
        c.update(x0=int(xs.min()), x1=int(xs.max()) + 1, y0=int(ys.min()), y1=int(ys.max()) + 1)
        c['h'], c['w'] = c['y1'] - c['y0'], c['x1'] - c['x0']
        c['mask'] = m[c['y0']:c['y1'], c['x0']:c['x1']]
        c['area'] = int(m.sum())
    if a.debug:
        rng = np.random.RandomState(1); pal = np.r_[[[0, 0, 0]], rng.randint(60, 255, (len(elems) + 1, 3))].astype(np.uint8)
        Image.fromarray(pal[full2]).save(a.debug + '_parts.png')

    elems = [c for k_, c in enumerate(elems) if k_ not in air]
    stops = [dict(x0=c['x0'], x1=c['x1'], y0=c['y0'], y1=c['y1']) for c in comps
             if c['h'] >= 0.3 * hmax and 4 < c['w'] <= thin]
    # 光阑：线条类里细而高的竖直连通域（不在任何镜片 body 内）
    vert = ndimage.binary_opening(line_m & ~body_core, structure=np.ones((15, 1)))  # 只留竖直线段
    cl, n = ndimage.label(vert)
    for i, sl in enumerate(ndimage.find_objects(cl), 1):
        if sl is None: continue
        hh, ww = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if ww <= 5 and hh >= 0.3 * hmax:
            stops.append(dict(x0=sl[1].start, x1=sl[1].stop, y0=sl[0].start, y1=sl[0].stop))
    # 光轴
    ay_sym = float(np.median([(c['y0'] + c['y1']) / 2 for c in elems]))
    ay = a.axis if a.axis is not None else (axis_line_y if axis_line_y is not None and abs(axis_line_y - ay_sym) < 8 else ay_sym)
    print(f'[光轴] 对称中心 y={ay_sym:.1f}  采用 y={ay:.1f}', file=sys.stderr)
    print(f'[光轴] y={ay:.1f}  镜片数={len(elems)}  光阑候选={len(stops)}', file=sys.stderr)

    # 逐片拟合
    parts = []
    for c in elems:
        m = c['mask']
        rows = np.where(m.any(1))[0]
        L = np.array([[m[r].argmax() + c['x0'], r + c['y0']] for r in rows], float)
        Rr = np.array([[c['x1'] - 1 - m[r][::-1].argmax(), r + c['y0']] for r in rows], float)
        fl = fit_surface(L[:, 0], L[:, 1], ay, 'L', a.flat_span)
        fr = fit_surface(Rr[:, 0], Rr[:, 1], ay, 'R', a.flat_span)
        if fl is None or fr is None:
            continue
        # 边缘平边端点（DXF 用）：轮廓最上/最下行的 x
        top, bot = rows[0] + c['y0'], rows[-1] + c['y0']
        parts.append(dict(color=int(c['color']), front=fl, back=fr,
                          semi=(c['y1'] - c['y0']) / 2.0, x0=c['x0'], x1=c['x1'],
                          top=int(top), bot=int(bot),
                          edge_L=[float(L[0, 0]), float(L[-1, 0])], edge_R=[float(Rr[0, 0]), float(Rr[-1, 0])]))
    parts.sort(key=lambda p: p['front']['vertex'])
    for k, p in enumerate(parts, 1):
        print(f"[分片{k:2d}] 色类{p['color']} x={p['front']['vertex']:7.1f}..{p['back']['vertex']:7.1f}  半高 {p['semi']:.0f}px", file=sys.stderr)
    if a.drop:
        drop = {int(v) for v in a.drop.split(',')}
        parts = [p for k, p in enumerate(parts, 1) if k not in drop]
        print(f'[丢弃] 按 --drop 去掉分片 {sorted(drop)}', file=sys.stderr)

    pxmm_hint = 1.0

    def cemented(pa, pb):
        """pa 的后面与 pb 的前面是否胶合：顶点距离小，且两顶点之间靠近光轴的几行里没有背景像素。"""
        v1, v2 = pa['back']['vertex'], pb['front']['vertex']
        if not (-3 < v2 - v1 < a.cement_gap): return False
        # 胶合面两侧拟合出的半径必须一致；符号相反或相差 >30% 的是极薄空气间隔（如 0.15 mm）
        R1, R2 = pa['back']['R'], pb['front']['R']
        if (R1 is None) != (R2 is None): return False
        if R1 is not None and (R1 * R2 < 0 or abs(R1 - R2) > 0.3 * min(abs(R1), abs(R2))):
            print(f'[提示] x≈{v1:.0f} 两侧半径 {R1 / pxmm_hint:.1f} / {R2 / pxmm_hint:.1f} 不一致 → 按极薄空气间隔处理', file=sys.stderr)
            return False
        c0, c1 = int(round(min(v1, v2))) + 1, int(round(max(v1, v2)))
        if c1 <= c0: return True
        r0 = int(ay) + (a.bridge // 2 + 3)
        rows = np.r_[max(0, int(ay) - (a.bridge // 2 + 6)): max(0, int(ay) - (a.bridge // 2 + 2)),
                     min(H, r0): min(H, r0 + 4)]
        return not (lab[rows][:, c0:c1] == bg).any()

    # 光阑
    stops.sort(key=lambda s: s['x0'])

    # 比例尺
    v_first, v_last = parts[0]['front']['vertex'], parts[-1]['back']['vertex']
    if a.pxmm:
        pxmm = a.pxmm
    elif a.length:
        pxmm = (v_last - v_first) / a.length
    elif a.diameter:
        pxmm = 2 * max(p['semi'] for p in parts) / a.diameter
    else:
        pxmm = 1.0
        print('[警告] 未给比例尺，1 px = 1 单位', file=sys.stderr)
    print(f'[比例] {pxmm:.4f} px/mm  总长 {(v_last - v_first) / pxmm:.2f} mm', file=sys.stderr)
    pxmm_hint = pxmm
    # 被描边围起来的薄空气间隔（两侧都紧贴邻片、内部是背景色、中心厚度 < --min-thick mm）不是镜片，丢掉
    keep = []
    for i, p in enumerate(parts):
        thick = p['back']['vertex'] - p['front']['vertex']
        if p['color'] == bg and 0 < i < len(parts) - 1 and thick < a.min_thick * pxmm \
                and cemented(parts[i - 1], p) and cemented(p, parts[i + 1]):
            print(f"[丢弃] x={p['front']['vertex']:.0f}..{p['back']['vertex']:.0f} 是被围起来的空气间隔", file=sys.stderr)
            continue
        keep.append(p)
    parts = keep

    # 生成面序列
    def mm(px): return px / pxmm
    surfaces = []
    stop_x = [(s['x0'] + s['x1']) / 2 for s in stops]
    i = 0
    n = len(parts)
    while i < n:
        p = parts[i]
        # 胶合链：后顶点与下一片前顶点几乎重合
        chain = [p]
        while i + 1 < n and cemented(chain[-1], parts[i + 1]):
            chain.append(parts[i + 1]); i += 1
        for j, q in enumerate(chain):
            f = q['front'] if j == 0 else None
            if j == 0:
                surfaces.append(dict(R=f['R'], vertex=f['vertex'], rms=f['rms'], semi=q['semi'],
                                     semi_fit=f.get('semi_fit'),
                                     glass=True, color=q['color'], asph=(not f['flat'] and f['rms'] > 1.0)))
            else:
                # 胶合面：两条弧取平均
                pf, cb = q['front'], chain[j - 1]['back']
                Rs = [r for r in (pf['R'], cb['R']) if r is not None]
                Rm = float(np.mean(Rs)) if Rs else None
                surfaces.append(dict(R=Rm, vertex=(pf['vertex'] + cb['vertex']) / 2, rms=max(pf['rms'], cb['rms']),
                                     semi=q['semi'], semi_fit=max(pf.get('semi_fit') or 0, cb.get('semi_fit') or 0),
                                     glass=True, color=q['color'], cemented=True,
                                     asph=max(pf['rms'], cb['rms']) > 1.0))
        b = chain[-1]['back']
        surfaces.append(dict(R=b['R'], vertex=b['vertex'], rms=b['rms'], semi=chain[-1]['semi'],
                             semi_fit=b.get('semi_fit'),
                             glass=False, color=None, asph=(not b['flat'] and b['rms'] > 1.0)))
        i += 1
    # 插入光阑
    for sx in stop_x:
        k = next((k for k, s in enumerate(surfaces) if s['vertex'] > sx), len(surfaces))
        if k > 0 and surfaces[k - 1]['glass']:
            print(f'[警告] 光阑 x={sx:.0f} 落在玻璃内部，忽略', file=sys.stderr); continue
        surfaces.insert(k, dict(R=None, vertex=sx, rms=0, semi=None, glass=False, stop=True, color=None))
    # 厚度
    for k, s in enumerate(surfaces):
        s['D'] = mm(surfaces[k + 1]['vertex'] - s['vertex']) if k + 1 < len(surfaces) else 0.0
        if not s['glass'] and not s.get('stop') and k + 1 < len(surfaces) \
                and s['D'] * pxmm <= linew + 1.5:
            s['thin_gap'] = True
        s['R_mm'] = None if s['R'] is None else round(mm(s['R']), 3)
        s['semi_mm'] = None if s['semi'] is None else round(mm(s['semi']), 2)
        s['semi_fit_mm'] = None if s.get('semi_fit') is None else round(mm(s['semi_fit']), 2)
        s['D'] = round(s['D'], 3)

    tags = {}
    if a.tags:
        for kv in a.tags.split(','):
            k_, v_ = kv.split('='); tags[int(k_)] = v_.strip()
    for k_, s_ in enumerate(surfaces):
        if s_.get('color') is None or s_['color'] not in tags: continue
        s_['tag'] = tags[s_['color']]
        # 图例说是非球面就直接定性，别只靠拟合残差；标记落在元件前面，后面那个面是同一元件的后表面
        if any(w in s_['tag'].upper() for w in ('非球面', 'ASPH')):
            s_['asph'] = True
            if k_ + 1 < len(surfaces) and not surfaces[k_ + 1].get('stop'):
                surfaces[k_ + 1]['asph'] = True
                surfaces[k_ + 1]['asph_by_tag'] = True
            s_['asph_by_tag'] = True
    out = dict(image=a.image, crop=a.crop, flip=a.flip, pxmm=pxmm, axis_px=ay, W=W, H=H, bg=bg, tags=tags,
               stop_semi_px=[(st['y1'] - st['y0']) / 2 for st in stops],
               total_length_mm=round(mm(v_last - v_first), 3),
               colors={int(c): [int(v) for v in cent[c]] for c in range(a.k)},
               elements=[dict(color=p['color'], semi_mm=round(mm(p['semi']), 2),
                              front_vertex=p['front']['vertex'], back_vertex=p['back']['vertex'],
                              top=p['top'], bot=p['bot'], edge_L=p['edge_L'], edge_R=p['edge_R']) for p in parts],
               surfaces=surfaces)
    json.dump(out, open(a.o, 'w'), ensure_ascii=False, indent=1)

    # 打印表
    print(f"{'面':>4} {'R(mm)':>10} {'D(mm)':>8} {'半径mm':>7} {'rms':>5}  备注", file=sys.stderr)
    for k, s in enumerate(surfaces, 1):
        name = 'STO' if s.get('stop') else k
        note = ('胶合 ' if s.get('cemented') else '') + ('可能非球面' if s.get('asph') else '')
        print(f"{str(name):>4} {('INF' if s['R_mm'] is None else s['R_mm']):>10} {s['D']:>8} "
              f"{(s['semi_mm'] if s['semi_mm'] is not None else ''):>7} {s['rms']:>5.2f}  {note}", file=sys.stderr)

    if a.overlay:
        ov = im.copy().convert('RGB'); dr = ImageDraw.Draw(ov)
        dr.line([(0, ay), (W, ay)], fill=(255, 0, 0), width=1)
        for s in surfaces:
            if s.get('stop'):
                dr.line([(s['vertex'], ay - 0.35 * H), (s['vertex'], ay + 0.35 * H)], fill=(0, 200, 0), width=2)
                continue
            semi = s['semi']
            if s['R'] is None:
                dr.line([(s['vertex'], ay - semi), (s['vertex'], ay + semi)], fill=(255, 0, 255), width=2)
            else:
                R = abs(s['R']); cx = s['vertex'] + s['R']
                pts = []
                for t in np.linspace(-semi, semi, 60):
                    if abs(t) >= R: continue
                    dx = math.sqrt(R * R - t * t)
                    x = cx - dx if s['R'] > 0 else cx + dx
                    pts.append((x, ay + t))
                dr.line(pts, fill=(255, 0, 255), width=2)
            dr.ellipse([s['vertex'] - 3, ay - 3, s['vertex'] + 3, ay + 3], outline=(255, 255, 0), width=2)
        ov.save(a.overlay)
        print(f'[叠加图] {a.overlay}', file=sys.stderr)


if __name__ == '__main__':
    main()
