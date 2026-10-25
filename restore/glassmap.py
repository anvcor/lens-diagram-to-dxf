"""Glass variables and glass maps of the model glasses.  env LENS_GMAP = mixed (default) | hull | poly.

hull  -- CODE V style: the fictitious glass varies in (nd, dN = nF - nC) and is bounded by a convex polygon in that
         plane, one linear inequality per edge (CODE V GLA / 'GL A..E' rows, Lagrange-controlled: a glass slides along
         an edge and sits on two edges at a corner).  Every element gets the convex hull of the glasses it may use;
         its variables are linear in (nd, dN), no box, so the edge rows are exactly linear in the variables.
poly  -- variables (nd, vd) in a sigmoid box, one row per element = signed distance to a global non-convex (vd, nd)
         polygon around the real glasses (core.GMAP).
mixed -- hull for the elements restricted to moulded glass (a sparse set: restricting them early pays off), poly for
         the others (the convex hull of the polished glasses bridges empty regions; the polygon hugs the real glasses).
         Sigma 50 CL (2026-10): final CODE V-checked mtf dev  poly 0.042 / 0.097, hull 0.068 / 0.089, mixed 0.036 / 0.054
         (hybrid / all-moulded).
Allowed glasses per element: config glass.rules / glass.modes ('moulded' | 'polished' | 'any'), vendors, vd <= vd_max.
Hull edge rows a*X + b*Y + c (<= 0 inside), X = nd/0.01, Y = dN/4.4e-4, divided by 0.5 like the polygon rows."""
import os, numpy as np, torch
from scipy.spatial import ConvexHull
import config as CF
from glass import CAT, is_moulded
import core as T

KIND = os.environ.get('LENS_GMAP', 'mixed')
assert KIND in ('hull', 'poly', 'mixed'), KIND
DN_UNIT = 4.4e-4
ELEMS = CF.ELEMS
dev, DT = T.dev, T.DT


def allowed(e, mode=None):
    r = CF.rule(e, mode)
    out = []
    for (v, name), g in CAT.items():
        if v not in CF.VENDORS or name.startswith(('MP-', 'MC-')) or g['vd'] > CF.VD_MAX:
            continue
        mo = is_moulded(v, name)
        if r == 'moulded':
            ok = mo
        elif r == 'polished':
            ok = not mo
        else:
            ok = True
        if ok:
            out.append((v, name))
    return out


def _xy(nd, vd):
    return nd / 0.01, (nd - 1) / vd / DN_UNIT


def hull(e, mode=None):
    """(K,3) edge equations a, b, c (unit outward normal), corner glass names in hull order"""
    keys = allowed(e, mode)
    P = np.array([_xy(CAT[k]['nd'], CAT[k]['vd']) for k in keys])
    h = ConvexHull(P)
    return h.equations.copy(), [keys[i][1] for i in h.vertices]


USE_HULL = [KIND == 'hull' or (KIND == 'mixed' and CF.rule(e) == 'moulded') for e in ELEMS]
HMASK = torch.tensor(USE_HULL, device=dev)
PMASK = ~HMASK
EDGES, CORNERS = [], {}
for _e, _u in zip(ELEMS, USE_HULL):
    if _u:
        eq, cn = hull(_e); EDGES.append(eq); CORNERS[_e] = cn
    else:
        EDGES.append(np.zeros((0, 3)))
KMAX = max([len(q) for q in EDGES] + [1])
_EQ = np.zeros((len(ELEMS), KMAX, 3)); _EQ[..., 2] = -100.0             # padding rows: constant, never active
for _i, _q in enumerate(EDGES):
    _EQ[_i, :len(_q)] = _q
EQ = torch.tensor(_EQ, dtype=DT, device=dev)
NAMES = ([] if KIND == 'poly' else ['gl_%s_%s' % (e, chr(65 + k)) for e in ELEMS for k in range(KMAX)]) \
    + ([] if KIND == 'hull' else ['gmap_%s' % e for e in ELEMS])


def edge_rows(nd, vd):
    X, Y = _xy(nd, vd)
    g = EQ[None, :, :, 0] * X[..., None] + EQ[None, :, :, 1] * Y[..., None] + EQ[None, :, :, 2]
    return (g / 0.5).reshape(nd.shape[0], -1)


def poly_signed(vd, nd):
    """signed distance to the global polygon (negative inside), normalised (vd, nd/0.01); relu() == core.gmap_dist"""
    P = torch.stack([vd, nd / 0.01], -1)
    A = T.GM; Bv = torch.roll(T.GM, -1, 0); E = Bv - A
    AP = P[..., None, :] - A
    t = ((AP * E).sum(-1) / (E * E).sum(-1)).clamp(0, 1)
    dmin = torch.sqrt(((AP - t[..., None] * E) ** 2).sum(-1) + 1e-9).min(-1).values
    with torch.no_grad():
        y = P[..., 1:2].detach(); x = P[..., 0:1].detach()
        y1, y2 = A[:, 1], Bv[:, 1]; x1, x2 = A[:, 0], Bv[:, 0]
        cond = ((y1 > y) != (y2 > y))
        xint = x1 + (y - y1) * (x2 - x1) / (y2 - y1 + 1e-12)
        inside = ((cond & (x < xint)).sum(-1) % 2) == 1
    return torch.where(inside, -dmin, dmin)


def constraint_rows(nd, vd):
    """glass-map constraint rows (B, len(NAMES)), c <= 0 wanted"""
    rows = []
    if KIND != 'poly':
        rows.append(edge_rows(nd, vd))
    if KIND != 'hull':
        rows.append(torch.where(PMASK[None], poly_signed(vd, nd) / 0.5, torch.full_like(nd, -200.0)))
    return torch.cat(rows, 1)


def penalty_rows(nd, vd):
    """objective penalty rows = relu(constraint_rows) (the loss identity with dls holds)"""
    if KIND == 'poly':
        return T.gmap_dist(vd, nd) / 0.5
    return torch.relu(constraint_rows(nd, vd))


# ---- glass variables: hull elements linear in (nd, dN) (CODE V GLN / GLV), poly elements sigmoid box (nd, vd)
ND0, ND_S, DN0, DN_S, DN_MIN = 1.70, 0.10, 0.017, 0.004, 2.5e-3


def u_to_glass(un, ug):
    ndh = ND0 + ND_S * un
    vdh = (ndh - 1) / (DN0 + DN_S * ug).clamp(min=DN_MIN)
    if KIND == 'hull':
        return ndh, vdh
    ndp, vdp = 1.43 + 0.63 * torch.sigmoid(un), 15 + 81 * torch.sigmoid(ug)
    if KIND == 'poly':
        return ndp, vdp
    return torch.where(HMASK, ndh, ndp), torch.where(HMASK, vdh, vdp)


def glass_to_u(nd, vd, idx=None):
    """numpy inverse of u_to_glass.  nd, vd: arrays over the elements, or scalars for element index idx"""
    nd = np.asarray(nd, float); vd = np.asarray(vd, float)
    h = np.array(USE_HULL) if idx is None else USE_HULL[idx]

    def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
    uh = ((nd - ND0) / ND_S, ((nd - 1) / vd - DN0) / DN_S)
    up = (logit((nd - 1.43) / 0.63), logit((np.minimum(vd, 95.9) - 15) / 81))
    return np.where(h, uh[0], up[0]), np.where(h, uh[1], up[1])


if __name__ == '__main__':
    print('LENS_GMAP', KIND, 'mode', CF.MODE, 'hull elements', [e for e, h in zip(ELEMS, USE_HULL) if h], 'rows', len(NAMES))
    for e in ELEMS:
        print('%s rule %-8s %3d glasses' % (e, CF.rule(e), len(allowed(e))))
