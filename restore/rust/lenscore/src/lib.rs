//! lenscore: CPU ray-trace core of the restoration solver (C ABI, loaded from Python via ctypes; see restore/rcore.py).
//!
//! Exact port of restore/core.trace + objective.image_points: same intersection formulas, clamps, 5 value-only Newton
//! steps + one differentiable Newton step on aspheres, same cover-glass propagation to the image.
//! Derivatives: forward-mode dual numbers over the per-ray local parameters
//!   [0, nq)       lens parameters chosen by the caller (asphere c / A4..A8, back gap, index of each variable element at
//!                 the ray's wavelength), mapped through the surface slots cslot / aslot / nslot / t16slot
//!   nq, nq + 1    start x, start y of the ray
//! contracted inside the kernel with the per-design local -> variable Jacobian M (nb, nw, nq, nu) and the optional
//! per-ray start seeds (nb, nr, 2, nu): outputs are derivatives with respect to the optimiser variables.
//! The dual width N is a compile-time constant; lc_trace picks the smallest instantiation >= nq + 2.
use rayon::prelude::*;
use std::ops::{Add, Div, Mul, Neg, Sub};

const NSMAX: usize = 64;

// ------------------------------------------------------------------------------------------------ numbers
pub trait Num: Copy + Send + Sync + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Neg<Output = Self> {
    fn cst(v: f64) -> Self;
    fn val(&self) -> f64;
    fn sqrt(self) -> Self;
    fn var(v: f64, slot: i32) -> Self;
}

impl Num for f64 {
    #[inline] fn cst(v: f64) -> Self { v }
    #[inline] fn val(&self) -> f64 { *self }
    #[inline] fn sqrt(self) -> Self { f64::sqrt(self) }
    #[inline] fn var(v: f64, _slot: i32) -> Self { v }
}

#[derive(Clone, Copy)]
pub struct Dual<const N: usize> { pub v: f64, pub d: [f64; N] }

impl<const N: usize> Num for Dual<N> {
    #[inline] fn cst(v: f64) -> Self { Dual { v, d: [0.0; N] } }
    #[inline] fn val(&self) -> f64 { self.v }
    #[inline] fn sqrt(self) -> Self {
        let s = self.v.sqrt(); let k = 0.5 / s;
        let mut d = [0.0; N]; for i in 0..N { d[i] = self.d[i] * k; }
        Dual { v: s, d }
    }
    #[inline] fn var(v: f64, slot: i32) -> Self {
        let mut d = [0.0; N]; if slot >= 0 { d[slot as usize] = 1.0; }
        Dual { v, d }
    }
}
impl<const N: usize> Add for Dual<N> { type Output = Self; #[inline] fn add(self, o: Self) -> Self { let mut d = self.d; for i in 0..N { d[i] += o.d[i]; } Dual { v: self.v + o.v, d } } }
impl<const N: usize> Sub for Dual<N> { type Output = Self; #[inline] fn sub(self, o: Self) -> Self { let mut d = self.d; for i in 0..N { d[i] -= o.d[i]; } Dual { v: self.v - o.v, d } } }
impl<const N: usize> Neg for Dual<N> { type Output = Self; #[inline] fn neg(self) -> Self { let mut d = self.d; for i in 0..N { d[i] = -d[i]; } Dual { v: -self.v, d } } }
impl<const N: usize> Mul for Dual<N> {
    type Output = Self;
    #[inline] fn mul(self, o: Self) -> Self { let mut d = [0.0; N]; for i in 0..N { d[i] = self.d[i] * o.v + self.v * o.d[i]; } Dual { v: self.v * o.v, d } }
}
impl<const N: usize> Div for Dual<N> {
    type Output = Self;
    #[inline] fn div(self, o: Self) -> Self {
        let inv = 1.0 / o.v; let q = self.v * inv;
        let mut d = [0.0; N]; for i in 0..N { d[i] = (self.d[i] - q * o.d[i]) * inv; }
        Dual { v: q, d }
    }
}

#[inline] fn cmin<T: Num>(x: T, lo: f64) -> T { if x.val() < lo { T::cst(lo) } else { x } }
#[inline] fn cmax<T: Num>(x: T, hi: f64) -> T { if x.val() > hi { T::cst(hi) } else { x } }
#[inline] fn clmp<T: Num>(x: T, lo: f64, hi: f64) -> T { cmax(cmin(x, lo), hi) }

// ------------------------------------------------------------------------------------------------ system
struct Sys<'a> {
    ns: usize, nw: usize,
    zs: &'a [f64], asph: &'a [i32], c: &'a [f64], a: &'a [f64], n: &'a [f64],
    cslot: &'a [i32], aslot: &'a [i32], nslot: &'a [i32],
}

#[inline]
fn sag_g<T: Num>(c: T, a: &[T; 4], r2: T) -> (T, T) {
    let r2 = cmax(r2, 900.0);
    let q = cmin(T::cst(1.0) - c * c * r2, 1e-8).sqrt();
    let r4 = r2 * r2; let r6 = r4 * r2; let r8 = r6 * r2; let r10 = r8 * r2;
    let sag = c * r2 / (T::cst(1.0) + q) + a[0] * r4 + a[1] * r6 + a[2] * r8 + a[3] * r10;
    let g = c / q + T::cst(4.0) * a[0] * r2 + T::cst(6.0) * a[1] * r4 + T::cst(8.0) * a[2] * r6 + T::cst(10.0) * a[3] * r8;
    (sag, g)
}

fn trace_ray<T: Num>(s: &Sys, b: usize, w: usize, p0: [T; 3], d0: [T; 3], k0: usize, k1: usize, rs: &mut [T; NSMAX]) -> ([T; 3], [T; 3]) {
    let (ns, nw) = (s.ns, s.nw);
    let mut pp = p0; let mut dd = d0;
    let mut npv = if k0 == 0 { T::cst(1.0) } else { T::var(s.n[(b * ns + k0 - 1) * nw + w], s.nslot[k0 - 1]) };
    for k in k0..k1 {
        let zk = s.zs[k];
        let c = T::var(s.c[b * ns + k], s.cslot[k]);
        let a = [T::var(s.a[(b * ns + k) * 4], s.aslot[k * 3]), T::var(s.a[(b * ns + k) * 4 + 1], s.aslot[k * 3 + 1]),
                 T::var(s.a[(b * ns + k) * 4 + 2], s.aslot[k * 3 + 2]), T::cst(s.a[(b * ns + k) * 4 + 3])];
        let n2 = T::var(s.n[(b * ns + k) * nw + w], s.nslot[k]);
        let p = [pp[0], pp[1], pp[2] - T::cst(zk)];
        let pd = p[0] * dd[0] + p[1] * dd[1] + p[2] * dd[2];
        let p2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
        let bq = dd[2] - c * pd; let cq = c * p2 - T::cst(2.0) * p[2];
        let disc = cmin(bq * bq - c * cq, 1e-9);
        let mut den = bq + disc.sqrt();
        if den.val().abs() < 1e-4 { den = T::cst(1e-4); }
        let mut t = cq / den;
        if s.asph[k] != 0 {
            let (cv, av) = (c.val(), [a[0].val(), a[1].val(), a[2].val(), a[3].val()]);
            let (pv, dv) = ([p[0].val(), p[1].val(), p[2].val()], [dd[0].val(), dd[1].val(), dd[2].val()]);
            let mut tt = t.val();
            for _ in 0..5 {
                let q = [pv[0] + tt * dv[0], pv[1] + tt * dv[1], pv[2] + tt * dv[2]];
                let r2 = q[0] * q[0] + q[1] * q[1];
                let (sg, g) = sag_g::<f64>(cv, &av, r2);
                let f = q[2] - sg;
                let mut fp = dv[2] - g * (q[0] * dv[0] + q[1] * dv[1]);
                if fp.abs() < 1e-3 { fp = 1e-3; }
                tt = tt - f / fp;
                if !tt.is_finite() { tt = 0.0; }
                tt = tt.clamp(-200.0, 200.0);
            }
            let ttc = T::cst(tt);
            let q = [p[0] + ttc * dd[0], p[1] + ttc * dd[1], p[2] + ttc * dd[2]];
            let r2 = q[0] * q[0] + q[1] * q[1];
            let (sg, g) = sag_g(c, &a, r2);
            let f = q[2] - sg;
            let mut fp = dd[2] - g * (q[0] * dd[0] + q[1] * dd[1]);
            if fp.val().abs() < 1e-3 { fp = T::cst(1e-3); }
            t = ttc - f / fp;
        }
        let q = [p[0] + t * dd[0], p[1] + t * dd[1], p[2] + t * dd[2]];
        let r2 = q[0] * q[0] + q[1] * q[1];
        let (_, g) = sag_g(c, &a, r2);
        let nx = -g * q[0]; let ny = -g * q[1]; let nz = T::cst(1.0);
        let nn = (nx * nx + ny * ny + nz * nz).sqrt();
        let nv = [nx / nn, ny / nn, nz / nn];
        let mu = npv / n2;
        let ci = dd[0] * nv[0] + dd[1] * nv[1] + dd[2] * nv[2];
        let arg = cmin(T::cst(1.0) - mu * mu * (T::cst(1.0) - ci * ci), 1e-6);
        let kk = arg.sqrt() - mu * ci;
        let mut d2 = [mu * dd[0] + kk * nv[0], mu * dd[1] + kk * nv[1], mu * dd[2] + kk * nv[2]];
        pp = [clmp(q[0], -1e3, 1e3), clmp(q[1], -1e3, 1e3), clmp(q[2] + T::cst(zk), -1e3, 1e3)];
        let dn = cmin((d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2]).sqrt(), 1e-6);
        d2 = [d2[0] / dn, d2[1] / dn, d2[2] / dn];
        dd = d2;
        let mut r = (cmax(r2, 1e6) + T::cst(1e-12)).sqrt();
        if !r.val().is_finite() { r = T::cst(1e3); }
        rs[k] = r;
        npv = n2;
    }
    (pp, dd)
}

#[inline]
fn to_image<T: Num>(pl: [T; 3], dl: [T; 3], zl: T, ncg: f64, cg_t: f64, cg_gap: f64) -> [T; 2] {
    let dz = cmin(dl[2], 0.05);
    let t = (zl - pl[2]) / dz;
    let pc = [pl[0] + t * dl[0], pl[1] + t * dl[1]];
    let dc = [dl[0] / T::cst(ncg), dl[1] / T::cst(ncg)];
    let dcz = cmin(T::cst(1.0) - (dc[0] * dc[0] + dc[1] * dc[1]), 1e-6).sqrt();
    let mut out = [T::cst(0.0); 2];
    for i in 0..2 {
        let v = pc[i] + dc[i] / dcz * T::cst(cg_t) + dl[i] / dz * T::cst(cg_gap);
        out[i] = if v.val().is_finite() { clmp(v, -50.0, 50.0) } else { T::cst(0.0) };
    }
    out
}

#[inline]
fn contract<const N: usize>(x: &Dual<N>, m: &[f64], nq: usize, seed: Option<(&[f64], &[f64])>, nu: usize, out: &mut [f64]) {
    for j in 0..nu { out[j] = 0.0; }
    for q in 0..nq {
        let dq = x.d[q];
        if dq != 0.0 { let row = &m[q * nu..(q + 1) * nu]; for j in 0..nu { out[j] += dq * row[j]; } }
    }
    if let Some((sx, sy)) = seed {
        let (a, b) = (x.d[nq], x.d[nq + 1]);
        if a != 0.0 || b != 0.0 { for j in 0..nu { out[j] += a * sx[j] + b * sy[j]; } }
    }
}

struct SendPtr(*mut f64);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr { #[inline] fn at(&self, i: usize) -> *mut f64 { unsafe { self.0.add(i) } } }

#[inline] unsafe fn sl<'a, T>(p: *const T, n: usize) -> &'a [T] { unsafe { std::slice::from_raw_parts(p, n) } }

#[repr(C)]
pub struct Args {
    pub nb: usize, pub nr: usize, pub ns: usize, pub nw: usize, pub nu: usize, pub nq: usize,
    pub zs: *const f64, pub asph: *const i32, pub c: *const f64, pub a: *const f64, pub n: *const f64,
    pub cslot: *const i32, pub aslot: *const i32, pub nslot: *const i32,
    pub m: *const f64,
    pub p0: *const f64, pub d0: *const f64, pub iw: *const i32,
    pub seed: *const f64,
    pub k0: usize, pub k1: usize, pub mode: i32,
    pub t16: *const f64, pub t16slot: i32, pub zlast: f64, pub cg_t: f64, pub cg_gap: f64, pub ncg: *const f64,
    pub group: *const i32, pub ngroups: usize,
    pub ndd: usize, pub dd_idx: *const i32,
    pub p_out: *mut f64, pub d_out: *mut f64, pub dp_out: *mut f64, pub dd_out: *mut f64,
    pub rs_out: *mut f64, pub rsmax: *mut f64, pub drsmax: *mut f64,
}

fn run<const N: usize>(g: &Args) -> i32 {
    let (nb, nr, ns, nw, nu, nq) = (g.nb, g.nr, g.ns, g.nw, g.nu, g.nq);
    let s = unsafe { Sys {
        ns, nw, zs: sl(g.zs, ns), asph: sl(g.asph, ns), c: sl(g.c, nb * ns), a: sl(g.a, nb * ns * 4), n: sl(g.n, nb * ns * nw),
        cslot: sl(g.cslot, ns), aslot: sl(g.aslot, ns * 3), nslot: sl(g.nslot, ns),
    } };
    let p0 = unsafe { sl(g.p0, nb * nr * 3) }; let d0 = unsafe { sl(g.d0, nb * nr * 3) }; let iw = unsafe { sl(g.iw, nb * nr) };
    let mode = g.mode; let (k0, k1) = (g.k0, g.k1);
    let ncg = if mode == 1 { unsafe { sl(g.ncg, nw) } } else { &[][..] };
    let t16 = if mode == 1 { unsafe { sl(g.t16, nb) } } else { &[][..] };
    let mm = if nu > 0 { unsafe { sl(g.m, nb * nw * nq * nu) } } else { &[][..] };
    let seed = if nu > 0 && !g.seed.is_null() { Some(unsafe { sl(g.seed, nb * nr * 2 * nu) }) } else { None };
    let grp = if g.ngroups > 0 && !g.group.is_null() { Some(unsafe { sl(g.group, nb * nr) }) } else { None };
    let (sx, sy) = (nq as i32, nq as i32 + 1);
    let mut ddrank = vec![-1i32; nr];
    if mode == 1 && nu > 0 && g.ndd > 0 { let ix = unsafe { sl(g.dd_idx, g.ndd) }; for (j, &r) in ix.iter().enumerate() { ddrank[r as usize] = j as i32; } }
    let (po, do_, dpo, ddo) = (SendPtr(g.p_out), SendPtr(g.d_out), SendPtr(g.dp_out), SendPtr(g.dd_out));
    let mut rs_all = if grp.is_some() && g.rs_out.is_null() { vec![0.0f64; nb * nr * ns] } else { Vec::new() };
    let rsa = if g.rs_out.is_null() { SendPtr(rs_all.as_mut_ptr()) } else { SendPtr(g.rs_out) };
    let want_rs = !g.rs_out.is_null() || grp.is_some();
    let (has_d, has_dp, has_dd) = (!g.d_out.is_null(), !g.dp_out.is_null(), !g.dd_out.is_null());
    let (ndd, zlast, cg_t, cg_gap, t16slot) = (g.ndd, g.zlast, g.cg_t, g.cg_gap, g.t16slot);
    (0..nb * nr).into_par_iter().for_each(|idx| {
        let b = idx / nr; let r = idx % nr; let w = iw[idx] as usize;
        let i3 = idx * 3;
        if nu == 0 {
            let mut rs = [0.0f64; NSMAX];
            let (pl, dl) = trace_ray::<f64>(&s, b, w, [p0[i3], p0[i3 + 1], p0[i3 + 2]], [d0[i3], d0[i3 + 1], d0[i3 + 2]], k0, k1, &mut rs);
            unsafe {
                if mode == 1 {
                    let pi = to_image::<f64>(pl, dl, zlast + t16[b], ncg[w], cg_t, cg_gap);
                    *po.at(i3) = pi[0]; *po.at(i3 + 1) = pi[1]; *po.at(i3 + 2) = 0.0;
                } else { for j in 0..3 { *po.at(i3 + j) = pl[j]; } }
                if has_d { for j in 0..3 { *do_.at(i3 + j) = dl[j]; } }
                if want_rs { for k in k0..k1 { *rsa.at(idx * ns + k) = rs[k]; } }
            }
        } else {
            let mut rs = [Dual::<N>::cst(0.0); NSMAX];
            let px = Dual::<N>::var(p0[i3], sx); let py = Dual::<N>::var(p0[i3 + 1], sy);
            let (pl, dl) = trace_ray::<Dual<N>>(&s, b, w, [px, py, Dual::cst(p0[i3 + 2])], [Dual::cst(d0[i3]), Dual::cst(d0[i3 + 1]), Dual::cst(d0[i3 + 2])], k0, k1, &mut rs);
            let mw = &mm[(b * nw + w) * nq * nu..(b * nw + w + 1) * nq * nu];
            let sd = seed.map(|sv| (&sv[(idx * 2) * nu..(idx * 2 + 1) * nu], &sv[(idx * 2 + 1) * nu..(idx * 2 + 2) * nu]));
            let mut buf = vec![0.0f64; nu];
            unsafe {
                if mode == 1 {
                    let zl = Dual::<N>::var(zlast + t16[b], t16slot);
                    let pi = to_image::<Dual<N>>(pl, dl, zl, ncg[w], cg_t, cg_gap);
                    *po.at(i3) = pi[0].v; *po.at(i3 + 1) = pi[1].v; *po.at(i3 + 2) = 0.0;
                    for j in 0..2 { contract(&pi[j], mw, nq, sd, nu, &mut buf); for u in 0..nu { *dpo.at((idx * 2 + j) * nu + u) = buf[u]; } }
                    for j in 0..3 { *do_.at(i3 + j) = dl[j].v; }
                    let rk = ddrank[r];
                    if rk >= 0 {
                        let base = (b * ndd + rk as usize) * 3;
                        for j in 0..3 { contract(&dl[j], mw, nq, sd, nu, &mut buf); for u in 0..nu { *ddo.at((base + j) * nu + u) = buf[u]; } }
                    }
                } else {
                    for j in 0..3 { *po.at(i3 + j) = pl[j].v; *do_.at(i3 + j) = dl[j].v; }
                    if has_dp { for j in 0..3 { contract(&pl[j], mw, nq, sd, nu, &mut buf); for u in 0..nu { *dpo.at((i3 + j) * nu + u) = buf[u]; } } }
                    if has_dd { for j in 0..3 { contract(&dl[j], mw, nq, sd, nu, &mut buf); for u in 0..nu { *ddo.at((i3 + j) * nu + u) = buf[u]; } } }
                }
                if want_rs { for k in k0..k1 { *rsa.at(idx * ns + k) = rs[k].v; } }
            }
        }
    });
    if let Some(gr) = grp {
        let ngroups = g.ngroups;
        let rsv: &[f64] = if g.rs_out.is_null() { &rs_all } else { unsafe { sl(g.rs_out, nb * nr * ns) } };
        let (rmx, drm) = (SendPtr(g.rsmax), SendPtr(g.drsmax));
        let has_drs = !g.drsmax.is_null();
        (0..nb).into_par_iter().for_each(|b| {
            for gi in 0..ngroups {
                let mut arg = vec![usize::MAX; ns]; let mut best = vec![f64::NEG_INFINITY; ns];
                for r in 0..nr {
                    if gr[b * nr + r] != gi as i32 { continue; }
                    for k in k0..k1 { let v = rsv[(b * nr + r) * ns + k]; if v > best[k] { best[k] = v; arg[k] = r; } }
                }
                unsafe { for k in 0..ns { *rmx.at((b * ngroups + gi) * ns + k) = if arg[k] == usize::MAX { 0.0 } else { best[k] }; } }
                if nu > 0 && has_drs {
                    let mut done: Vec<usize> = Vec::new();
                    for k in k0..k1 {
                        let r = arg[k]; if r == usize::MAX || done.contains(&r) { continue; }
                        done.push(r);
                        let idx = b * nr + r; let i3 = idx * 3; let w = iw[idx] as usize;
                        let mut rs = [Dual::<N>::cst(0.0); NSMAX];
                        let px = Dual::<N>::var(p0[i3], sx); let py = Dual::<N>::var(p0[i3 + 1], sy);
                        trace_ray::<Dual<N>>(&s, b, w, [px, py, Dual::cst(p0[i3 + 2])], [Dual::cst(d0[i3]), Dual::cst(d0[i3 + 1]), Dual::cst(d0[i3 + 2])], k0, k1, &mut rs);
                        let mw = &mm[(b * nw + w) * nq * nu..(b * nw + w + 1) * nq * nu];
                        let sd = seed.map(|sv| (&sv[(idx * 2) * nu..(idx * 2 + 1) * nu], &sv[(idx * 2 + 1) * nu..(idx * 2 + 2) * nu]));
                        let mut buf = vec![0.0f64; nu];
                        for kk in k0..k1 {
                            if arg[kk] != r { continue; }
                            contract(&rs[kk], mw, nq, sd, nu, &mut buf);
                            unsafe { for u in 0..nu { *drm.at(((b * ngroups + gi) * ns + kk) * nu + u) = buf[u]; } }
                        }
                    }
                }
            }
        });
    }
    0
}

/// Main kernel (see Args).  mode 0 (raw): P_out, D_out (nb,nr,3) and, if nu > 0, dP / dD (nb,nr,3,nu).
/// mode 1 (image): P_out = (Pi_x, Pi_y, 0), D_out = direction at the last surface; if nu > 0: dP (nb,nr,2,nu) for the
///   image intercepts, dD (nb,ndd,3,nu) for the rays listed in dd_idx.  rs_out (nb,nr,ns) heights (may be null).
///   Group maxima (group >= 0): rsmax (nb,ng,ns) and, if nu > 0, drsmax (nb,ng,ns,nu) of the arg-max ray.
/// Returns 0, -1 if ns > NSMAX, -2 if nq + 2 exceeds the largest dual width.
#[unsafe(no_mangle)]
pub extern "C" fn lc_trace(g: *const Args) -> i32 {
    let g = unsafe { &*g };
    if g.ns > NSMAX { return -1; }
    if g.nu == 0 { return run::<1>(g); }
    match g.nq + 2 {
        n if n <= 19 => run::<19>(g),
        n if n <= 24 => run::<24>(g),
        n if n <= 32 => run::<32>(g),
        n if n <= 40 => run::<40>(g),
        n if n <= 48 => run::<48>(g),
        n if n <= 64 => run::<64>(g),
        _ => -2,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn lc_nthreads() -> usize { rayon::current_num_threads() }

#[unsafe(no_mangle)]
pub extern "C" fn lc_version() -> i32 { 2 }
