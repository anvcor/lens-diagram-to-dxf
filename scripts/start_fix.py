#!/usr/bin/env python3
"""第 1 级「起点修复」：结构图反推的面数据（玻璃全占位）→ 一个光线追得通、像差已初步收敛的起点。

    python start_fix.py SIGMA85.xlsx -o SIGMA85_start --efl 85 --fno 1.25 --ih 21.6 \
        --sld-tag 蓝色填充 --asph-tag 红色描边 [--codev]

两个阶段：
  1a 近轴 + Seidel（永远不会「光线追不过去」）
     变量  每面曲率 c、每片连续 nd/vd、非球面片的 A4、后截距空气
     目标  EFL、像面在近轴焦点、轴向/倍率色差、Seidel 横向量、单面贡献（应变）、近轴入射角上限
     多起点：玻璃初值按片的正负随机撒，留最好的几个进 1b。
  1b 子午面实光线（2D 精确追迹，球面 + 偶次非球面 A4..A10，C/d/F 三色）
     F1.2 级镜头五阶以上球差是主角，三阶和为零时轴上边缘光线仍可偏 1 mm，必须用实光线收一遍。
     孔径分步开大（0.6→0.8→1.0），每步按结构图半口径冻结一份渐晕掩膜；弧矢方向仍靠 Seidel SAS 兜着。

片型约束（两阶段通用）：不卡 R±x%，卡**边缘矢高差** |sag(h) − sag_图(h)| ≤ ε，
  ε = max(eps_px/pxmm, rel·|sag|)。弱面自动放宽、强面自动收紧——结构图量到的本来就是矢高。
  球面：直接折算成 c 的上下界；非球面：c 的框放宽 3 倍，含高次项的总矢高在 0.5/0.7/0.85/1.0 h 处另卡。
玻璃约束：(vd, nd) 落在厂家目录的凸包内；SLD 标记片限 vd≥67、nd≤1.60。

输出  <o>.seq（CODE V，虚拟玻璃 nnnnnn.vvvvvv 写法）+ <o>.json（全部数值，供后续级使用）
--codev  用 CODE V COM 复核：EFL、THO 三阶和、各视场参考光线、SET VIG 后的 RMS 弥散斑。
"""
import argparse, json, math, os, re, sys
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import live

HERE = os.path.dirname(os.path.abspath(__file__))
PFD = 0.7046          # (nF−nd)/(nF−nC) 的正常玻璃典型值；这一级只用来把 nF−nC 拆到 F/C 两侧
NA = 4                # 每个非球面的系数个数：A4 A6 A8 A10
A_SCALE = np.array([1e-7, 1e-10, 1e-13, 1e-16])
A_BOUND = np.array([5e-5, 5e-8, 5e-11, 5e-14])


# ---------------------------------------------------------------- 输入
def read_xlsx(path):
    import openpyxl
    ws = openpyxl.load_workbook(path, data_only=True)['面数据']
    rows = list(ws.iter_rows(values_only=True))
    m = re.search(r'比例\s*([\d.]+)\s*px/mm', str(rows[1][0]))
    pxmm = float(m.group(1)) if m else None
    surfs = []
    for r in rows[3:]:
        if r[0] is None or r[0] == 'IMA': continue
        stop = (r[0] == 'STO')
        R = None if (r[2] in (None, 'Infinity') or stop) else float(r[2])
        surfs.append({'name': r[0], 'stop': stop, 'R': R, 'D': float(r[3] or 0), 'glass': bool(r[4]),
                      'tag': r[5], 'semi': r[6], 'draw': r[7] or r[6]})
    return {'title': rows[0][0], 'pxmm': pxmm, 'surfaces': surfs}


def glass_hull(vendor='HOYA'):
    """厂家目录在 (vd, nd/0.01) 平面上的凸包 → A·p ≤ b。"""
    sys.path.insert(0, os.path.join(HERE, '..', '..', 'patent-lens-to-excel', 'scripts'))
    import lensmath
    lib = [g for g in lensmath.load(vendor) if not lensmath.legacy(g) and 1.4 < g['nd'] < 2.2 and 15 < g['vd'] < 100]
    P = np.array([[g['vd'], g['nd'] / 0.01] for g in lib])
    h = ConvexHull(P)
    return h.equations[:, :2], -h.equations[:, 2], lib, P


# ---------------------------------------------------------------- 面形
def sag(c, h, A=None):
    h = np.asarray(h, float); s = np.maximum(1 - (c * h) ** 2, 1e-9)
    z = c * h * h / (1 + np.sqrt(s))
    if A is not None:
        h2 = h * h; z = z + h2 * h2 * (A[0] + h2 * (A[1] + h2 * (A[2] + h2 * A[3])))
    return z


def c_from_sag(z, h):
    return 2 * z / (h * h + z * z)


class Model:
    def __init__(self, d, a):
        for k, v in (('efl_tol', 0.05), ('dist_max', 0.02), ('ca_margin', 0.1), ('ca_k', 1.0), ('vig_v', 0.35),
                     ('asph_loose', 3.0), ('bfl_et', 12.5), ('ri_target', 0.25), ('et_ratio', 5.0), ('et_min', 1.0), ('edge_margin', 1.0)):     # 旧版 json 里没有这些参数
            if not hasattr(a, k): setattr(a, k, v)
        self.a = a
        S = [dict(s) for s in d['surfaces']]
        for s in S[:-1]:                               # 图上贴在一起的气隙给个最小值
            if not s['glass'] and not s['stop'] and s['D'] < a.min_gap: s['D'] = a.min_gap
        self.S = S; ns = len(S)
        e = -1
        for s in S:                                    # 元件编号：每个 glass=True 的面开一片
            if s['glass']: e += 1; s['elem'] = e
        self.ne = e + 1
        self.ista = next(i for i, s in enumerate(S) if s['stop'])
        self.ci = [i for i, s in enumerate(S) if not s['stop']]
        self.is_sld = [False] * self.ne
        for s in S:
            if s['glass'] and a.sld_tag and a.sld_tag in str(s['tag'] or ''): self.is_sld[s['elem']] = True
        # 非球面：带标记那片的前后两面都放开（哪一面真用，留给下一级裁）
        self.ai = [i for i, s in enumerate(S) if not s['stop'] and a.asph_tag and a.asph_tag in str(s['tag'] or '')]
        eps0 = a.eps_px / d['pxmm']
        self.c0 = np.array([0.0 if S[i]['R'] is None else 1.0 / S[i]['R'] for i in self.ci])
        self.h = np.array([float(S[i]['draw'] or S[i]['semi']) for i in self.ci])
        self.z0 = sag(self.c0, self.h)
        self.eps = np.maximum(eps0, a.eps_rel * np.abs(self.z0))
        # 非球面：表里的 R 只是最佳拟合球面的近似值，基准曲率不值得卡——放得很开（只防跑飞），
        # 片型靠「含高次项的总矢高」另卡，而且只要求大致像（容差 × asph_loose）。
        k = np.array([12.0 if i in self.ai else 1.0 for i in self.ci])
        self.clo = c_from_sag(self.z0 - self.eps * k, self.h); self.chi = c_from_sag(self.z0 + self.eps * k, self.h)
        self.semi = np.array([float(s['semi'] or 1e9) for s in S] + ([1e9, 1e9] if a.cover > 0 else []))
        self.HA, self.Hb, self.lib, self.GP = glass_hull(a.vendor)
        # 变量布局与尺度（内部全用无量纲 z = x/scale）
        nc, ne, na = len(self.ci), self.ne, len(self.ai)
        o = 0; self.sl = {}
        for key, n in (('c', nc), ('nd', ne), ('vd', ne), ('A', na * NA), ('bk', 1)):
            self.sl[key] = slice(o, o + n); o += n
        self.n = o
        sc = np.ones(o); sc[self.sl['c']] = 1e-3; sc[self.sl['nd']] = 0.02; sc[self.sl['vd']] = 2.0
        sc[self.sl['A']] = np.tile(A_SCALE, na); sc[self.sl['bk']] = 0.5
        self.sc = sc
        lo = np.empty(o); hi = np.empty(o)
        lo[self.sl['c']], hi[self.sl['c']] = self.clo, self.chi
        ndlo, ndhi = np.full(ne, 1.43), np.full(ne, 2.11)       # 上限要容得下 TAFD65 (2.0509)
        vdlo, vdhi = np.full(ne, 16.0), np.full(ne, 96.0)       # 下限要容得下 FDS16-W (16.48)
        for e_ in range(ne):
            if self.is_sld[e_]: ndhi[e_] = 1.60; vdlo[e_] = 67.0
        lo[self.sl['nd']], hi[self.sl['nd']] = ndlo, ndhi
        lo[self.sl['vd']], hi[self.sl['vd']] = vdlo, vdhi
        lo[self.sl['A']], hi[self.sl['A']] = -np.tile(A_BOUND, na), np.tile(A_BOUND, na)
        lo[self.sl['bk']], hi[self.sl['bk']] = a.back_min, a.back_max
        self.lo, self.hi = lo / sc, hi / sc
        # 1a 阶段只放开 A4，高次项固定为 0
        fr = np.ones(o, bool); idx = np.arange(o)[self.sl['A']]; fr[idx[np.arange(len(idx)) % NA != 0]] = False
        self.free_seidel = fr
        self.ubar = a.ih / a.efl; self.hm = a.efl / a.fno / 2      # 标称值；实际值随 EFL 浮动，用 pupil(x)
        # —— 口径模型 ——
        # 官方结构图的口径是**画大了的**（所以前组看起来边缘极薄），不能当通光口径用。图上「面画到哪儿」(draw) 只是上限 cmax。
        # 真实通光口径由光线定：ca = max(k·|y|, |ȳ| + v·|y|)，y/ȳ = 轴上边缘光线 / 最大视场主光线在该面的高度。
        #   k≈1.03：轴上光束全过再留一点余量（轴上不许有渐晕）；v：最大视场保留的光束半宽比例，调它让角上照度 ≈ ri_target。
        # 约束是「需要的 ca ≤ cmax − 余量」。光阑与盖板不参与。
        cm = [1e9 if s['stop'] else float(s['draw'] or s['semi'] or 1e9) for s in S]
        for i in self.ai:                                # 非球面的 draw 是「球面拟合残差变大处」截断的，不是真实的面边界 → 用元件外框
            if S[i]['semi']: cm[i] = max(cm[i], float(S[i]['semi']) - 0.5)
            # 同一片的另一面画到多大，这一面至少也有多大（用户 2026-09-18：S7 的口径和 S8 一样）
            j = i + 1 if S[i]['glass'] else i - 1
            if 0 <= j < ns and not S[j]['stop'] and (S[i]['glass'] or S[j]['glass']): cm[i] = max(cm[i], float(S[j]['draw'] or 0))
        for kv in (getattr(a, 'cmax', None) or []):      # 人工覆盖：--cmax 7=24.0 11=21.5（面号按 CODE V 编号）
            k, v = kv.split('='); cm[int(k) - 1] = float(v)
        self.cmax = np.array(cm + ([1e9, 1e9] if a.cover > 0 else []))
        self.lim = self.cmax - a.ca_margin
        self.elems = [i for i, s in enumerate(S) if s['glass']]          # 每片：前面 i、后面 i+1
        self.hE = None                                                   # 算边缘厚度用的半径（按阶段冻结）

    def clear_ap(self, o):
        """由近轴光线高度算各面通光口径（见上）。o = seidel(x) 的返回值。"""
        a = self.a; y, yb = np.abs(o['y']), np.abs(o['yb'])
        ca = np.minimum(np.maximum(a.ca_k * y, yb + a.vig_v * y), self.cmax)
        ca[self.ista] = o['y_stop'] * 1.002; ca[len(self.S):] = 1e9
        return ca

    def edge_h(self, x):
        """每片算边缘厚度的半径 = 两面通光口径的大者 + 装夹余量，不超过元件机械半径。"""
        ca = self.clear_ap(self.seidel(x))
        return np.array([min(max(ca[i], ca[i + 1]) + self.a.edge_margin, self.semi[i]) for i in self.elems])

    def edge_thick(self, x, hE=None):
        """[(CT, ET)]：ET 在 hE 处算。加工极限：max(CT/ET, ET/CT) ≤ et_ratio，且 ET ≥ et_min。"""
        c, t, n, dn, Aco = self.table(x); hE = self.edge_h(x) if hE is None else hE
        return [(t[i], t[i] + float(sag(c[i + 1], h, Aco[i + 1]) - sag(c[i], h, Aco[i]))) for i, h in zip(self.elems, hE)]

    def pupil(self, x):
        """(入瞳半径, 最大视场 tanω, 光阑半径)。F 数与像高固定，EFL 在 ±tol 内浮动，所以三者都跟着 x 走。"""
        o = self.seidel(x)
        return o['hm'], o['ubar'], o['y_stop']

    # -- 把变量摊成逐面表：c, t(到下一面), n_after(d), dn_after(nF−nC), A(ns×4)
    def table(self, x):
        a = self.a; S = self.S; ns = len(S)
        c = np.zeros(ns); A = np.zeros((ns, NA))
        c[self.ci] = x[self.sl['c']]
        if self.ai: A[self.ai] = x[self.sl['A']].reshape(-1, NA)
        nd, vd = x[self.sl['nd']], x[self.sl['vd']]
        n = np.ones(ns); dn = np.zeros(ns); t = np.array([s['D'] for s in S])
        for i, s in enumerate(S):
            if s['glass']: n[i] = nd[s['elem']]; dn[i] = (nd[s['elem']] - 1) / vd[s['elem']]
        t[-1] = x[self.sl['bk']][0]
        if a.cover > 0:                                  # 盖板：两个平面
            c = np.r_[c, 0, 0]; A = np.vstack([A, np.zeros((2, NA))]); t = np.r_[t, a.cover, a.cover_gap]
            n = np.r_[n, a.cover_nd, 1.0]; dn = np.r_[dn, (a.cover_nd - 1) / a.cover_vd, 0.0]
        return c, t, n, dn, A

    # ------------------------------------------------------------ 近轴 + Seidel
    @staticmethod
    def ptrace(c, t, n, y, u):
        Y = np.empty(len(c)); U0 = np.empty(len(c)); U1 = np.empty(len(c)); nb = 1.0
        for i in range(len(c)):
            Y[i] = y; U0[i] = u
            u = (nb * u - y * c[i] * (n[i] - nb)) / n[i]
            U1[i] = u; y = y + u * t[i]; nb = n[i]
        return Y, U0, U1, y

    def seidel(self, x):
        c, t, n, dn, Aco = self.table(x)
        nb = np.r_[1.0, n[:-1]]; dnb = np.r_[0.0, dn[:-1]]
        Ya, Ua0, Ua1, ya_img = self.ptrace(c, t, n, 1.0, 0.0)
        Yb, Ub0, Ub1, _ = self.ptrace(c, t, n, 0.0, 1.0)
        efl = -1.0 / Ua1[-1]
        # 光瞳只在标称 EFL 附近跟随（连续夹紧）。不能写成「范围内跟随、范围外取标称」——那是一堵不连续的墙：
        # 起点 EFL=337 往 85 走，跨进范围的一瞬间入瞳半径从 34 跳到 100，口径罚项暴涨，优化器被挡在外面（实测过）。
        e_ = float(np.clip(efl, 0.9 * self.a.efl, 1.1 * self.a.efl)) if np.isfinite(efl) else self.a.efl
        hm, ubar = e_ / self.a.fno / 2, self.a.ih / e_   # F 数、像高固定（与 CODE V 的 FNO + YRI 一致）
        y1bar = -ubar * Yb[self.ista] / Ya[self.ista]
        y = hm * Ya; u0 = hm * Ua0; u1 = hm * Ua1
        yb = ubar * Yb + y1bar * Ya; ub0 = ubar * Ub0 + y1bar * Ua0
        A = nb * (u0 + y * c); Ab = nb * (ub0 + yb * c)
        H = ub0[0] * y[0] - u0[0] * yb[0]
        dun = u1 / n - u0 / nb
        S1 = -A * A * y * dun; S2 = -A * Ab * y * dun; S3 = -Ab * Ab * y * dun
        S4 = -H * H * c * (1 / n - 1 / nb)
        asp = 8 * Aco[:, 0] * (n - nb) * y ** 4
        r = np.divide(yb, y, out=np.zeros_like(y), where=np.abs(y) > 1e-9)
        S1 = S1 + asp; S2 = S2 + asp * r; S3 = S3 + asp * r * r
        dd = dn / n - dnb / nb
        C1 = A * y * dd; C2 = Ab * y * dd
        k = 1.0 / (2 * u1[-1])                           # 像方 n'=1
        return {'efl': efl, 'defocus_ta': ya_img * hm, 'ep_z': y1bar / ubar * -1.0, 'hm': hm, 'ubar': ubar,
                'y_stop': abs(float(y[self.ista])), 'y': y, 'yb': yb,
                'TSA': S1.sum() * k, 'TCO': 3 * S2.sum() * k, 'TAS': (3 * S3.sum() + S4.sum()) * k,
                'SAS': (S3.sum() + S4.sum()) * k, 'PTB': S4.sum() * k, 'AX': C1.sum() * 2 * k, 'LAT': C2.sum() * 2 * k,
                'imax': float(np.max(np.abs(np.r_[A / nb, A / n]))),
                'ibmax': float(np.max(np.abs(np.r_[(A + Ab) / nb, (A + Ab) / n]))),
                'per_surf': {'S1': S1 * k, 'S2': 3 * S2 * k, 'S3': 3 * S3 * k, 'C1': C1 * 2 * k}}

    def res_common(self, x):
        """玻璃图凸包 + 非球面总矢高框。"""
        a = self.a
        P = np.c_[x[self.sl['vd']], x[self.sl['nd']] / 0.01]
        viol = np.maximum(0.0, P @ self.HA.T - self.Hb + a.hull_margin)
        r = list(viol.max(axis=1) / 0.3)
        # 凸包挡不住玻璃图内部的空白区（FCD 与 LAC/TAC 之间那一大片），再卡「离最近真实牌号的距离」。
        # 度量：vd 差 1 ≈ nd 差 0.01。距离 ≤ glass_dist 不罚，保证下一级吸附到实牌号时改动很小。
        dmin = np.sqrt(((P[:, None, :] - self.GP[None, :, :]) ** 2).sum(-1)).min(axis=1)
        r += list(np.maximum(0.0, dmin - a.glass_dist) / 0.3)
        if self.ai:
            Aco = x[self.sl['A']].reshape(-1, NA); cx = x[self.sl['c']]
            for j, i in enumerate(self.ai):
                jj = self.ci.index(i)
                for f in (1.0, 0.85, 0.7, 0.5):
                    h = self.h[jj] * f
                    dz = float(sag(cx[jj], h, Aco[j]) - sag(self.c0[jj], h))
                    r.append(max(0.0, abs(dz) - a.asph_loose * self.eps[jj] * max(f * f, 0.4)) / (0.2 * self.eps[jj]))
        return r

    def res_efl(self, o):
        """焦距：商品标称值 ±efl_tol 以内都算对（标称 85 的镜头实际 82~85 很常见），出了这个带才罚。"""
        a = self.a
        return max(0.0, abs(o['efl'] - a.efl) - a.efl_tol * a.efl) / 0.02

    def res_aper(self, o, ym_real=None):
        """需要的通光口径不许超过图上的上限：近轴版 max(k|y|, |ȳ|+v|y|) ≤ lim；有实光线时再卡轴上满口径边缘光线。"""
        a = self.a; y, yb = np.abs(o['y']), np.abs(o['yb']); n = len(self.lim)
        r = list(np.maximum(0.0, np.maximum(a.ca_k * y, yb + a.vig_v * y) - self.lim) / 0.05)
        if ym_real is not None: r += list(np.maximum(0.0, a.ca_k * np.abs(ym_real[:n]) - self.lim) / 0.05)
        return r

    def res_et(self, x):
        """加工极限：中心厚/边缘厚 之比 ≤ et_ratio（正片别磨成刀口，负片边缘别太厚），边缘厚 ≥ et_min。"""
        a = self.a; r = []
        if self.hE is None: self.hE = self.edge_h(x)
        for ct, et in self.edge_thick(x, self.hE):       # 只卡「边缘太薄」。负片 ET/CT 大是结构图自己的几何（CT 固定 ~1 mm），不去较劲
            r.append(max(0.0, max(a.et_min, ct / a.et_ratio) - et) / 0.05)
        r.append(max(0.0, a.bfl_et - self.back_et(x)) / 0.02)      # 后截距：CODE V 里的 ET SI-3 > bfl_et
        return r

    def back_et(self, x):
        """末片后表面 → 盖板前表面 这段空气的**边缘厚度**（在末面通光口径处算）= CODE V 的 ET SI-3。
        镜头最靠近传感器的是末片边缘而不是顶点，所以后截距卡 ET 不卡 CT。"""
        c, t, n, dn, Aco = self.table(x); i = len(self.S) - 1
        h = float(self.clear_ap(self.seidel(x))[i])
        return float(t[i] - sag(c[i], h, Aco[i]))

    def res_seidel(self, x):
        a = self.a; o = self.seidel(x)
        r = [self.res_efl(o), o['defocus_ta'] / 0.01, o['AX'] / a.w_color, o['LAT'] / a.w_color,
             o['TSA'] / a.w_seidel, o['TCO'] / a.w_seidel, o['TAS'] / a.w_seidel, o['SAS'] / a.w_seidel,
             max(0.0, o['imax'] - a.imax) / 0.02, max(0.0, o['ibmax'] - a.imax - 0.1) / 0.02]
        # 应变：三阶和为零可以靠各面巨大的正负贡献对消得到，那样高阶像差极大。让每个面自己的贡献也小。
        if a.w_strain > 0:
            ps = o['per_surf']
            r += list(ps['S1'] / a.w_strain) + list(ps['S2'] / a.w_strain) + list(ps['S3'] / a.w_strain)
            r += list(ps['C1'] / (a.w_strain * 0.1))
        r += self.res_aper(o) + self.res_et(x)            # 近轴版的口径约束（实光线版在 1b）+ 边缘厚度
        return np.array(r + self.res_common(x))

    # ------------------------------------------------------------ 子午面实光线
    def rays(self, x, fields, rhos, ep_z, path=None):
        """返回 (yimg[nf, nr, 3], ok[nf, nr, 3], clip[nf, nr])；波长顺序 C, d, F。"""
        c, t, n, dn, Aco = self.table(x)
        N = np.stack([n - (1 - PFD) * dn, n, n + PFD * dn], axis=-1)          # (ns, 3)
        nf, nr = len(fields), len(rhos)
        o = self.seidel(x); hm, ubar = o['hm'], o['ubar']
        semi = self.clear_ap(o)                          # 裁光线用「光线决定的通光口径」，不是图上画大了的外框；光阑半径由 F 数定
        ang = np.arctan(np.asarray(fields) * ubar)[:, None, None] * np.ones((1, nr, 3))
        # 光线从 z=-10（首面前）出发，瞄准近轴入瞳面上的 ρ·hm
        L = np.cos(ang); M = np.sin(ang)
        yp = (np.asarray(rhos) * hm)[None, :, None] * np.ones((nf, 1, 3))
        z = np.full_like(yp, -10.0); y = yp + (z - ep_z) * (M / L)
        ok = np.ones_like(yp, bool); clip = np.zeros((nf, nr))
        zv = 0.0; nbef = np.ones(3)
        for i in range(len(c)):
            ci, Ai = c[i], Aco[i]
            tt = (zv - z) / L
            for _ in range(8):                           # Newton：光线与面的交点
                yy = y + M * tt; s = 1 - (ci * yy) ** 2
                bad = s <= 1e-6; s = np.where(bad, 1.0, s); rt = np.sqrt(s); y2 = yy * yy
                f = ci * y2 / (1 + rt) + y2 * y2 * (Ai[0] + y2 * (Ai[1] + y2 * (Ai[2] + y2 * Ai[3])))
                fp = ci * yy / rt + yy * y2 * (4 * Ai[0] + y2 * (6 * Ai[1] + y2 * (8 * Ai[2] + y2 * 10 * Ai[3])))
                g = (z + L * tt - zv) - f
                tt = tt - g / (L - fp * M)
            ok &= ~bad
            z = z + L * tt; y = yy
            if path is not None: path.append((z[..., 1].copy(), y[..., 1].copy()))
            clip = np.maximum(clip, np.abs(y[..., 1]) / semi[i])
            nrm = np.sqrt(1 + fp * fp); Nz, Ny = 1 / nrm, -fp / nrm
            mu = nbef / N[i]
            cosi = L * Nz + M * Ny
            sin2t = mu * mu * (1 - cosi * cosi)
            tir = sin2t >= 1.0; ok &= ~tir
            cost = np.sqrt(np.where(tir, 0.0, 1 - sin2t))
            L = mu * L + (cost - mu * cosi) * Nz; M = mu * M + (cost - mu * cosi) * Ny
            nbef = N[i]; zv += t[i]
        yimg = y + (zv - z) * M / L
        if path is not None: path.append((np.full_like(yimg[..., 1], zv), yimg[..., 1].copy()))
        return yimg, ok, clip

    def vig_mask(self, x, fields, rhos, ep_z):
        _, ok, clip = self.rays(x, fields, rhos, ep_z)
        return ok.all(axis=-1) & (clip <= 1.0)

    def res_real(self, x, fields, rhos, mask, ep_z):
        a = self.a; o = self.seidel(x); path = []
        yimg, ok, _ = self.rays(x, fields, rhos, ep_z, path=path)
        ic = len(rhos) // 2
        ref = yimg[:, ic:ic + 1, 1:2]                     # 各视场 d 线主光线
        ta = np.where(ok, yimg - ref, a.fail_ta) / a.w_ta
        w = mask[:, :, None] * np.array([0.7, 1.0, 0.7])[None, None, :]
        r = list((ta * w).ravel())
        f = np.asarray(fields); nz = f > 0
        dist = ref.ravel()[nz] / (f[nz] * a.ih) - 1.0     # 主光线畸变；|畸变| ≤ dist_max 不罚
        r += list(np.maximum(0.0, np.abs(dist) - 0.8 * a.dist_max) / 0.002)
        r += [self.res_efl(o), o['SAS'] / (2 * a.w_seidel), o['LAT'] / a.w_color]
        Y = np.array([p[1] for p in path[:-1]])           # (面, 视场, 光线)；rhos 两端是 ±满口径
        full = abs(rhos[-1]) > 0.999                      # 孔径还没开满的阶段，两端光线不是满口径边缘光线
        r += self.res_aper(o, np.maximum(np.abs(Y[:, 0, 0]), np.abs(Y[:, 0, -1])) if full else None) + self.res_et(x)
        if a.w_strain > 0:
            ps = o['per_surf']; ws = a.w_strain * 2
            r += list(ps['S1'] / ws) + list(ps['S2'] / ws) + list(ps['S3'] / ws)
        return np.array(r + self.res_common(x))

    # ------------------------------------------------------------ 求解
    def _ls(self, fun, x, free, nfev):
        z = x / self.sc; best = [np.inf, z.copy()]

        class _Stop(Exception): pass

        def f(zf):
            zz = z.copy(); zz[free] = zf; r = fun(zz * self.sc); c = float(r @ r)
            if c < best[0]: best[0], best[1] = c, zz.copy()
            live.publish(best[1] * self.sc, stage=getattr(self, 'stage', ''), msg=getattr(self, 'msg', ''), errf=best[0])
            if live.should_stop(): raise _Stop
            return r
        z0 = np.clip(z[free], self.lo[free] + 1e-9, self.hi[free] - 1e-9)
        try:
            sol = least_squares(f, z0, bounds=(self.lo[free], self.hi[free]), method='trf', max_nfev=nfev, xtol=1e-10, ftol=1e-10)
            zz = z.copy(); zz[free] = sol.x; cost = 2 * sol.cost
            self.last_sol = sol                          # 调试用：nfev / status / message
        except _Stop:
            zz, cost = best[1], best[0]
        live.publish(zz * self.sc, stage=getattr(self, 'stage', ''), msg=getattr(self, 'msg', ''), errf=cost, force=True)
        return zz * self.sc, cost

    def x_init(self, rng):
        x = np.zeros(self.n); x[self.sl['c']] = self.c0; x[self.sl['bk']] = self.a.back
        nd = np.empty(self.ne); vd = np.empty(self.ne); first = {}
        for i, s in enumerate(self.S):
            if s['glass']: first[s['elem']] = i
        cf = np.zeros(len(self.S) + 1); cf[self.ci] = self.c0
        for e in range(self.ne):
            pos = (cf[first[e]] - cf[first[e] + 1]) > 0   # 每片正负：两面曲率差的符号（薄透镜近似）
            if self.is_sld[e]: nd[e], vd[e] = 1.497, 81.6
            elif pos: nd[e], vd[e] = rng.uniform(1.62, 1.90), rng.uniform(38, 60)
            else: nd[e], vd[e] = rng.uniform(1.60, 1.85), rng.uniform(22, 36)
        x[self.sl['nd']], x[self.sl['vd']] = nd, vd
        return x

    def solve_seidel(self, starts, seed, keep):
        rng = np.random.default_rng(seed); sols = []
        for k in range(starts):
            if live.should_stop(): break
            self.stage = '1a Seidel'; self.msg = f'起点 {k + 1}/{starts}'
            xi = self.x_init(rng); self.hE = self.edge_h(xi) if np.isfinite(self.seidel(xi)['efl']) else None
            x, cost = self._ls(self.res_seidel, xi, self.free_seidel, 300)
            live.log(f'  1a 起点 {k + 1:2d}/{starts}: cost {cost:12.4f}')
            sols.append((cost, x))
        sols.sort(key=lambda s: s[0])
        return sols[:keep]

    def solve_real(self, x, verbose=True):
        a = self.a; free = np.ones(self.n, bool)
        fields = list(np.linspace(0, 1, a.nfields)); base = np.linspace(-1, 1, a.nrho)
        for scale in a.ramp:
            if live.should_stop(): break
            self.stage = '1b 实光线'; self.msg = getattr(self, 'cand', '') + f' 孔径 ×{scale:.2f}'
            rhos = base * scale
            ep_z = self.seidel(x)['ep_z']; self.hE = self.edge_h(x)      # 渐晕掩膜与边缘厚度的计算半径按阶段冻结
            mask = self.vig_mask(x, fields, rhos, ep_z)
            mask[:, a.nrho // 2] = True
            mask[0, :] = True                            # 轴上没有渐晕：口径由 F 数定，不由图上量的光阑半径定
            x, cost = self._ls(lambda xx: self.res_real(xx, fields, rhos, mask, ep_z), x, free, a.nfev_real)
            if verbose: live.log(f'     1b 孔径 ×{scale:.2f}: 参与光线 {int(mask.sum())}/{mask.size}  cost {cost:10.3f}')
        return x, cost

    def aper_report(self, x):
        """口径体检：轴上边缘光线 / 各视场主光线 在每面的高度 ÷ 通光口径，以及各视场主光线畸变。
        返回 {'marg': [(面号, 比值)…超 1 的], 'chief': […], 'worst_marg': (面号, 比值), 'dist': [(视场, %)]}。"""
        a = self.a; fields = [0.0, 0.4, 0.7, 1.0]; path = []; o = self.seidel(x)
        yimg, ok, clip = self.rays(x, fields, np.linspace(-1, 1, 201), o['ep_z'], path=path)
        Y = np.abs(np.array([p[1] for p in path[:-1]])); n = len(self.lim)
        rm = a.ca_k * np.maximum(Y[:n, 0, 0], Y[:n, 0, -1]) / self.lim          # 轴上满口径边缘光线（含余量 k）÷ 上限
        rc = (np.abs(o['yb']) + a.vig_v * np.abs(o['y']))[:n] / self.lim        # 最大视场保留光束 ÷ 上限
        f = np.array(fields[1:]); dist = (yimg[1:, 100, 1] / (f * a.ih) - 1) * 100
        good = ok.all(axis=-1) & (clip <= 1.0); T = float(good[-1].mean())     # 最大视场子午方向通光比例
        et = [(e + 1, round(float(ct), 2), round(float(t), 2), round(float(ct / max(t, 1e-6)), 2))
              for e, (ct, t) in enumerate(self.edge_thick(x))]
        return {'marg': [(int(k + 1), round(float(v), 3)) for k, v in enumerate(rm) if v > 1.0],
                'chief': [(int(k + 1), round(float(v), 3)) for k, v in enumerate(rc) if v > 1.0],
                'worst_marg': (int(rm.argmax() + 1), round(float(rm.max()), 3)),
                'dist': [(float(a_), round(float(b), 2)) for a_, b in zip(f, dist)], 'tan_pass': round(T, 3),
                'et_bad': [q for q in et if q[3] > a.et_ratio + 0.05 or q[2] < a.et_min - 0.02], 'et_worst': max(et, key=lambda q: q[3]),
                'ca': [round(float(v), 2) for v in self.clear_ap(o)[:len(self.S)]]}

    def ray_report(self, x):
        fields = [0.0, 0.4, 0.7, 1.0]; rhos = np.linspace(-1, 1, 21); ep_z = self.seidel(x)['ep_z']
        yimg, ok, clip = self.rays(x, fields, rhos, ep_z)
        m = ok.all(axis=-1) & (clip <= 1.0); ref = yimg[:, 10:11, 1:2]; ta = yimg - ref
        out = []
        for k, f in enumerate(fields):
            if m[k].sum() < 2: out.append((f, None, None, 0)); continue
            t = ta[k][m[k]]
            out.append((f, float(np.sqrt(np.mean(t[:, 1] ** 2))), float(np.sqrt(np.mean(t ** 2))), float(m[k].mean())))
        return out


# ---------------------------------------------------------------- 输出
def glass_code(nd, vd):
    return f'{round(nd * 1e6) - 1000000:06d}.{round(vd * 1e4):06d}'


def write_seq(m, x, path, title):
    a = m.a; c, t, n, dn, Aco = m.table(x)
    hm, ubar, y_stop = m.pupil(x); ca = m.clear_ap(m.seidel(x))
    yan =[round(math.degrees(math.atan(f * ubar)), 4) for f in (0, 0.4, 0.7, 1.0)]
    L = ['RDM;LEN', f"TITLE '{title}'", f'FNO   {a.fno}', 'DIM   M',
         'WL    656.3 587.6 486.1', 'REF   2', 'WTW   1 1 1', 'YAN   ' + ' '.join(map(str, yan)), 'SO    0.0 0.1e19']
    nd, vd = x[m.sl['nd']], x[m.sl['vd']]
    for i, s in enumerate(m.S):
        if s['stop']:
            # 光阑口径 = 轴上边缘光线高度（由 F 数定）。图上量的光阑半径只是个示意符号，写进来会把轴上光束裁掉。
            L += [f'S     0.0 {t[i]:.6f}', '  STO', f'  CIR {y_stop:.4f}']
            continue
        R = 0.0 if abs(c[i]) < 1e-12 else 1.0 / c[i]
        g = f' {glass_code(nd[s["elem"]], vd[s["elem"]])}' if s['glass'] else ''
        L.append(f'S     {R:.6f} {t[i]:.6f}{g}')
        if i in m.ai and np.any(Aco[i] != 0):
            L += ['  ASP', '  K 0.0'] + [f'  {k} {v:.8e}' for k, v in zip('ABCD', Aco[i])]
        L.append(f'  CIR {ca[i]:.3f}')                    # 光线决定的通光口径（不是图上画大了的外框）
    if a.cover > 0:
        L += [f'S     0.0 {a.cover} BSC7_HOYA', f'S     0.0 {a.cover_gap}']
    L += ['SI    0.0 0.0', 'GO', 'SET VIG']
    open(path, 'w', encoding='utf-8').write('\n'.join(L) + '\n')


def report(m, x):
    o = m.seidel(x); ps = o['per_surf']; ar = m.aper_report(x)
    print(f"  口径：轴上边缘光线最紧在面 {ar['worst_marg'][0]}（需要/上限 = {ar['worst_marg'][1]}）"
          + (f"  ★超上限的面 {ar['marg']}" if ar['marg'] else '  轴上无渐晕 ✓') + (f"  ★轴外光束超上限 {ar['chief']}" if ar['chief'] else '')
          + f"   最大视场子午通光 {ar['tan_pass'] * 100:.0f}%")
    w = ar['et_worst']; print(f"  边缘厚：最极端是片 {w[0]}（CT {w[1]} / ET {w[2]}，比 {w[3]}）" + (f"  ★超加工极限 {ar['et_bad']}" if ar['et_bad'] else '  全部在加工极限内 ✓'))
    print('  畸变 %：' + '  '.join(f'{f:.1f}H {d:+.2f}' for f, d in ar['dist']))
    print(f"  EFL {o['efl']:.4f}   后截距空气 CT {x[m.sl['bk']][0]:.3f} / ET {m.back_et(x):.3f}（要求 ET ≥ {m.a.bfl_et}）近轴焦点处横向离焦 {o['defocus_ta']:+.4f}")
    print(f"  三阶横向量(mm): TSA {o['TSA']:+.4f}  TCO {o['TCO']:+.4f}  TAS {o['TAS']:+.4f}  SAS {o['SAS']:+.4f}  PTB {o['PTB']:+.4f}   色差: AX {o['AX']:+.5f}  LAT {o['LAT']:+.5f}")
    print(f"  应变 Σ|单面| (mm): SA {np.abs(ps['S1']).sum():.2f}  TCO {np.abs(ps['S2']).sum():.2f}  AST {np.abs(ps['S3']).sum():.2f}   最大单面 SA {np.abs(ps['S1']).max():.2f}")
    print('  子午实光线 RMS 横向像差 (mm)：' + '   '.join(
        f"{f:.1f}视场 " + ('全部被挡' if d is None else f"d {d:.4f} / CdF {p:.4f} (通光 {v * 100:.0f}%)") for f, d, p, v in m.ray_report(x)))
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('xlsx'); ap.add_argument('-o', required=True)
    ap.add_argument('--efl', type=float, required=True); ap.add_argument('--fno', type=float, required=True)
    ap.add_argument('--ih', type=float, default=21.6, help='像高 mm')
    ap.add_argument('--efl-tol', type=float, default=0.05, help='焦距允许偏离标称值的比例（商品值 ±5% 都算对）')
    ap.add_argument('--dist-max', type=float, default=0.02, help='|畸变| 上限（比例）')
    ap.add_argument('--ca-margin', type=float, default=0.1, help='需要的通光口径 ≤ 图上面的绘制范围 − 这么多 mm')
    ap.add_argument('--asph-loose', type=float, default=3.0, help='非球面总矢高的容差 = 球面容差 × 这个倍数（非球面只要求片型大致像，基准 R 不卡）')
    ap.add_argument('--cmax', nargs='*', default=[], help='人工覆盖某些面的口径上限，如 7=24.0 11=21.5（面号按 CODE V 编号，光阑算一面）')
    ap.add_argument('--ca-k', type=float, default=1.0, help='通光口径 ≥ 轴上边缘光线高度 × k（轴上无渐晕 + 余量）')
    ap.add_argument('--vig-v', type=float, default=0.35, help='最大视场保留的光束半宽比例（第 2 级会按 --ri-target 用 CODE V 重新标定）')
    ap.add_argument('--ri-target', type=float, default=0.25, help='最大视场相对照度目标')
    ap.add_argument('--et-ratio', type=float, default=5.0, help='加工极限：中心厚/边缘厚 之比上限（大口径镜片取低一些）')
    ap.add_argument('--et-min', type=float, default=1.0, help='边缘厚下限 mm')
    ap.add_argument('--edge-margin', type=float, default=1.0, help='算边缘厚度的半径 = 通光口径 + 这么多 mm（装夹余量）')
    ap.add_argument('--nfields', type=int, default=10, help='1b 实光线阶段的视场点数')
    ap.add_argument('--sld-tag', default='蓝色填充'); ap.add_argument('--asph-tag', default='红色描边')
    ap.add_argument('--vendor', default='HOYA')
    ap.add_argument('--eps-px', type=float, default=3.0, help='边缘矢高容差（像素）')
    ap.add_argument('--eps-rel', type=float, default=0.05, help='边缘矢高相对容差')
    ap.add_argument('--bfl-et', type=float, default=12.5, help='后截距：末片后表面到盖板的空气边缘厚度下限（CODE V: ET SI-3 > 值）')
    ap.add_argument('--min-gap', type=float, default=0.1)
    ap.add_argument('--back', type=float, default=13.68, help='末面→盖板前的空气（初值）')
    ap.add_argument('--back-min', type=float, default=11.0); ap.add_argument('--back-max', type=float, default=16.0)
    ap.add_argument('--cover', type=float, default=2.5); ap.add_argument('--cover-gap', type=float, default=1.0)
    ap.add_argument('--cover-nd', type=float, default=1.5168); ap.add_argument('--cover-vd', type=float, default=64.2)
    ap.add_argument('--w-seidel', type=float, default=0.05, help='三阶横向量的「可接受」尺度 mm')
    ap.add_argument('--w-color', type=float, default=0.005)
    ap.add_argument('--w-strain', type=float, default=2.0, help='单面三阶贡献的尺度 mm；0=关闭应变项')
    ap.add_argument('--w-ta', type=float, default=0.02, help='实光线横向像差的尺度 mm')
    ap.add_argument('--fail-ta', type=float, default=2.0, help='追迹失败的光线按这么大的横向像差计')
    ap.add_argument('--imax', type=float, default=0.70, help='近轴边缘光线入射角上限')
    ap.add_argument('--hull-margin', type=float, default=-0.5,
                    help='离玻璃图凸包边界的余量（vd 单位）；负数=允许略出界。FDS18-W / FF5 / TAFD65 / FCD100 这些极限牌号就是凸包的顶点，内缩会把它们罚掉')
    ap.add_argument('--glass-dist', type=float, default=1.5, help='连续玻璃离最近实牌号的允许距离（vd 1 ≈ nd 0.01）')
    ap.add_argument('--starts', type=int, default=12); ap.add_argument('--keep', type=int, default=3)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--ramp', type=float, nargs='+', default=[0.6, 0.8, 1.0]); ap.add_argument('--nrho', type=int, default=11)
    ap.add_argument('--nfev-real', type=int, default=60)
    ap.add_argument('--codev', action='store_true')
    a = ap.parse_args()

    d = read_xlsx(a.xlsx); m = Model(d, a)
    live.init(os.path.dirname(os.path.abspath(a.o)), m, xlsx=os.path.abspath(a.xlsx), title=d['title'])
    print(f"{d['title']}：{len(m.ci)} 个折射面 / {m.ne} 片；SLD 片 {sum(m.is_sld)}；非球面候选面 {len(m.ai)}；变量 {m.n}")
    rel = (m.chi - m.clo) / 2 / np.maximum(np.abs(m.c0), 1e-9) * 100
    print(f"  矢高框折算成 R 的相对宽度：最紧 ±{rel.min():.1f}%  中位 ±{np.median(rel):.1f}%  最松 ±{rel.max():.0f}%")
    x0 = m.x_init(np.random.default_rng(0)); x0[m.sl['nd']] = 1.5168; x0[m.sl['vd']] = 64.2
    print('原始（全占位玻璃）:'); report(m, x0)

    cands = m.solve_seidel(a.starts, a.seed, a.keep)
    best = None
    for k, (cost, xs) in enumerate(cands):
        live.log(f'  候选 {k + 1}（1a cost {cost:.3f}）→ 1b'); m.cand = f'候选 {k + 1}/{len(cands)}'
        x, c1 = m.solve_real(xs)
        if best is None or c1 < best[0]: best = (c1, x, xs)
    cost, x, xs = best
    print('1a 结束时:'); report(m, xs)
    print(f'1b 结束（cost {cost:.3f}）:'); o = report(m, x)

    print('  片  nd       vd      最近牌号                  Δnd       Δvd')
    glasses = []
    for e in range(m.ne):
        nd, vd = x[m.sl['nd']][e], x[m.sl['vd']][e]
        g = min(m.lib, key=lambda g: ((g['nd'] - nd) / 0.01) ** 2 + (g['vd'] - vd) ** 2)
        glasses.append({'elem': e + 1, 'nd': nd, 'vd': vd, 'nearest': g['name'], 'sld': m.is_sld[e]})
        print(f"  {e + 1:2d}  {nd:.5f}  {vd:6.2f}  {g['name']:<22s} {g['nd'] - nd:+.4f}  {g['vd'] - vd:+6.2f}" + ('  SLD' if m.is_sld[e] else ''))
    _, _, _, _, Aco = m.table(x)
    dz = np.array([float(sag(ci, h, Aco[i])) for ci, h, i in zip(x[m.sl['c']], m.h, m.ci)]) - m.z0
    tol = m.eps * np.array([a.asph_loose if i in m.ai else 1.0 for i in m.ci])       # 非球面的容差放宽
    print(f"  边缘矢高偏移/容差：max {np.max(np.abs(dz) / tol):.2f}   贴到框边的面 {int(np.sum(np.abs(dz) / tol > 0.98))}/{len(dz)}")

    live.publish(x, stage='1 完成', msg=os.path.basename(a.o), force=True)
    write_seq(m, x, a.o + '.seq', f"{d['title']} stage1")
    json.dump({'args': vars(a), 'cost': cost, 'x': x.tolist(), 'glasses': glasses,
               'seidel': {k: float(v) for k, v in o.items() if np.ndim(v) == 0 and k != 'per_surf'},
               'c': x[m.sl['c']].tolist(), 'c0': m.c0.tolist(), 'c_lo': m.clo.tolist(), 'c_hi': m.chi.tolist(),
               'h': m.h.tolist(), 'sag0': m.z0.tolist(), 'eps': m.eps.tolist(), 'asph_surfaces': m.ai},
              open(a.o + '.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('写出', a.o + '.seq', a.o + '.json')
    if a.codev: codev_check(a.o + '.seq')


def codev_check(seq):
    from cv_session import CodeV
    with CodeV(workdir=os.path.dirname(os.path.abspath(seq))) as cv:
        out = cv.run_seq(os.path.abspath(seq))
        if 'rror' in out: print(out)
        print(f"CODE V 复核：EFL {cv.eva('(EFL)'):.4f}")
        tho = cv.cmd('THO').splitlines()
        for l in [l for l in tho if 'TCO' in l][:1] + [l for l in tho if l.strip().startswith('SUM')]: print('  ' + l.strip())
        nf = int(cv.eva('(NUM F)'))
        for f in range(1, nf + 1):
            ys = []
            for r in (1, 2, 3):
                o = cv.cmd(f'EVA (Y SI F{f} R{r} W2)'); mm = re.search(r'=\s*([-\d.eE+]+)', o)
                ys.append(float(mm.group(1)) if mm and 'rror' not in o else None)
            print(f"  F{f} 未渐晕参考光线 主/上/下 像面 y: " + ' '.join('   fail  ' if v is None else f'{v:9.4f}' for v in ys))
        cv.cmd('SET VIG')
        spo = cv.cmd('SPO; GO')
        open(os.path.splitext(seq)[0] + '_spo.txt', 'w', encoding='utf-8').write(spo)
        fld = None
        for l in spo.splitlines():
            mm = re.search(r'Field +(\d+), *\(([^)]*)\)', l)
            if mm: fld = mm.group(1), mm.group(2); continue
            mm = re.search(r'([\d.E+-]+) MM\s*$', l)
            if mm and fld: print(f'  F{fld[0]} ({fld[1].strip()})°  RMS 弥散斑直径 {float(mm.group(1)) * 1000:.1f} µm'); fld = None

if __name__ == '__main__':
    main()
