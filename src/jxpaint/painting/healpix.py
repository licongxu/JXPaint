"""Healpix tSZ painter, bit-for-bit matching paint_a10_y0true_2d_mpi.jl.

For each halo: find disc pixels (healpy query_disc, RING), compute the
chord-based angular distance to each pixel centre, clamp to theta_min, and for
pixels with theta < theta_max accumulate A_halo * y_t(log theta, log theta500)
into the map, where y_t is the bicubic beam-convolved shape table.

query_disc is called inclusive=True (a superset of centre-within-theta_max
pixels); the strict theta < theta_max test then reproduces XGPaint's
centre-within-radius painted set exactly.
"""
import os
import time
import multiprocessing as mp

import numpy as np
import healpy as hp
import jax
import jax.numpy as jnp

from .. import constants as C
from . import geometry as geom


def paint_catalogue(z, M_1e14, lon, lat, y0_true, shape_table,
                    nside=C.NSIDE, cosmo=None, chunk=4000, progress=False):
    """Paint a Compton-y map. Returns a RING-ordered float64 map (npix,)."""
    npix = hp.nside2npix(nside)
    out = np.zeros(npix, dtype=np.float64)

    ra, dec = geom.catalogue_to_radec(lon, lat)
    vec = geom.radec_to_vec(ra, dec)                      # (Nh, 3)
    th500 = geom.theta500_array(M_1e14, z, cosmo=cosmo)   # (Nh,)
    thmax = geom.theta_max_array(M_1e14, z, cosmo=cosmo)  # (Nh,)
    amp = geom.amplitude_array(y0_true)                   # (Nh,)
    log_th500 = np.log(th500)
    thmin = np.exp(shape_table.logtheta_min)              # exp(-16.5)

    Nh = len(M_1e14)
    for c0 in range(0, Nh, chunk):
        c1 = min(c0 + chunk, Nh)
        pix_list, logth_list, log5_list, amp_list = [], [], [], []
        for i in range(c0, c1):
            tmax = thmax[i]
            disc = hp.query_disc(nside, vec[i], tmax, inclusive=True, nest=False)
            if disc.size == 0:
                continue
            vp = np.asarray(hp.pix2vec(nside, disc, nest=False)).T   # (npd,3)
            d2 = np.sum((vp - vec[i]) ** 2, axis=1)
            theta = np.arccos(np.clip(1.0 - d2 / 2.0, -1.0, 1.0))
            theta = np.maximum(thmin, theta)
            keep = theta < tmax
            if not keep.any():
                continue
            pix_list.append(disc[keep])
            logth_list.append(np.log(theta[keep]))
            log5_list.append(np.full(int(keep.sum()), log_th500[i]))
            amp_list.append(np.full(int(keep.sum()), amp[i]))
        if not pix_list:
            continue
        pix = np.concatenate(pix_list)
        logth = np.concatenate(logth_list)
        log5 = np.concatenate(log5_list)
        amps = np.concatenate(amp_list)
        yt = np.asarray(shape_table.evaluate(logth, log5))
        np.add.at(out, pix, amps * yt)
        if progress:
            print(f"  painted halos {c1}/{Nh}", flush=True)
    return out


def _scatter_paint(coefs, lt_min, lt_max, l5_min, l5_max, dlt, dl5, n0, n1, fill,
                   vp, vh, thmax_h, logth500_h, amp_h, allpix, npix):
    """GPU kernel: chord distance -> bicubic gather -> masked scatter-add.

    All array args are jnp on the active device.  Mirrors BeamedShapeTable.evaluate
    and the painter's per-pixel math, fused for throughput.
    """
    thmin = jnp.exp(lt_min)
    d2 = jnp.sum((vp - vh) ** 2, axis=1)
    theta = jnp.arccos(jnp.clip(1.0 - d2 / 2.0, -1.0, 1.0))
    theta = jnp.maximum(thmin, theta)
    keep = theta < thmax_h

    lt = jnp.log(theta)
    l5 = logth500_h
    p0 = 1.0 + (lt - lt_min) / dlt
    p1 = 1.0 + (l5 - l5_min) / dl5
    inside = (lt >= lt_min) & (lt <= lt_max) & (l5 >= l5_min) & (l5 <= l5_max)
    p0c = jnp.clip(p0, 1.0, float(n0)); p1c = jnp.clip(p1, 1.0, float(n1))
    ix0 = jnp.clip(jnp.floor(p0c).astype(jnp.int32), 1, n0 - 1)
    ix1 = jnp.clip(jnp.floor(p1c).astype(jnp.int32), 1, n1 - 1)
    f0 = p0c - ix0; f1 = p1c - ix1

    def cw(f):
        f2 = f * f; f3 = f2 * f
        return jnp.stack([(1 - 3 * f + 3 * f2 - f3) / 6.0,
                          (3 * f3 - 6 * f2 + 4) / 6.0,
                          (-3 * f3 + 3 * f2 + 3 * f + 1) / 6.0,
                          f3 / 6.0], axis=-1)
    w0 = cw(f0); w1 = cw(f1)
    r0 = jnp.clip(ix0[:, None] - 1 + jnp.arange(4), 0, n0 + 1)
    r1 = jnp.clip(ix1[:, None] - 1 + jnp.arange(4), 0, n1 + 1)
    block = coefs[r0[:, :, None], r1[:, None, :]]            # (M,4,4)
    yt = jnp.einsum("ma,mab,mb->m", w0, block, w1)
    yt = jnp.where(inside, yt, fill)

    contrib = jnp.where(keep, amp_h * yt, 0.0)
    out = jnp.zeros(npix, dtype=jnp.float64).at[allpix].add(contrib)
    return out


_scatter_paint_jit = jax.jit(
    _scatter_paint, static_argnums=(1, 2, 3, 4, 5, 6, 7, 8, 9, 16))


def _disc_worker(args):
    """Worker: query_disc over a halo slice. Returns (pix_concat, counts).

    Uses only healpy/numpy (no JAX/CUDA), so it is safe under fork.
    """
    vec_s, thmax_s, nside = args
    out = []
    for i in range(len(thmax_s)):
        out.append(hp.query_disc(nside, vec_s[i], thmax_s[i],
                                 inclusive=True, nest=False))
    counts = np.fromiter((x.size for x in out), dtype=np.int64, count=len(out))
    pix = np.concatenate(out) if out else np.empty(0, dtype=np.int64)
    return pix, counts


def _assemble_discs(vec, thmax, nside, nproc):
    """Find disc pixels for all halos in parallel; return (allpix, allh)."""
    Nh = len(thmax)
    if nproc <= 1 or Nh < 2000:
        pix, counts = _disc_worker((vec, thmax, nside))
        return pix, np.repeat(np.arange(Nh, dtype=np.int64), counts)
    bounds = np.linspace(0, Nh, nproc + 1).astype(np.int64)
    args = [(vec[a:b], thmax[a:b], nside)
            for a, b in zip(bounds[:-1], bounds[1:])]
    ctx = mp.get_context("fork")
    with ctx.Pool(nproc) as pool:
        results = pool.map(_disc_worker, args)
    pix_parts, h_parts = [], []
    for (pix_s, counts_s), a, b in zip(results, bounds[:-1], bounds[1:]):
        pix_parts.append(pix_s)
        h_parts.append(np.repeat(np.arange(a, b, dtype=np.int64), counts_s))
    return np.concatenate(pix_parts), np.concatenate(h_parts)


def paint_catalogue_gpu(z, M_1e14, lon, lat, y0_true, shape_table,
                        nside=C.NSIDE, cosmo=None, gpu_chunk=20_000_000,
                        nproc=None, verbose=False):
    """Fast painter: host assembles disc contributions (parallel), GPU math.

    Returns a RING-ordered float64 numpy map.  Bit-for-bit equal to
    paint_catalogue (same chord/bicubic/scatter math, just fused on device).
    Disc-finding is parallelised over `nproc` fork workers; geometry is numpy
    so CUDA is not initialised before the fork.
    """
    if nproc is None:
        nproc = min(16, max(1, (os.cpu_count() or 2) - 1))
    npix = hp.nside2npix(nside)
    ra, dec = geom.catalogue_to_radec(lon, lat)
    vec = geom.radec_to_vec(ra, dec)
    th500 = geom.theta500_array(M_1e14, z, cosmo=cosmo)
    thmax = geom.theta_max_array(M_1e14, z, cosmo=cosmo)
    amp = geom.amplitude_array(y0_true)
    log_th500 = np.log(th500)
    Nh = len(M_1e14)

    t0 = time.time()
    allpix, allh = _assemble_discs(vec, thmax, nside, nproc)
    t_host_disc = time.time() - t0

    t0 = time.time()
    vp = np.asarray(hp.pix2vec(nside, allpix, nest=False)).T.copy()   # (M,3)
    t_pv = time.time() - t0

    st = shape_table
    coefs = st.coefs_device()    # uploads table to GPU once (after the fork)
    # device arrays
    t0 = time.time()
    out_acc = np.zeros(npix, dtype=np.float64)
    M = allpix.size
    for c0 in range(0, M, gpu_chunk):
        c1 = min(c0 + gpu_chunk, M)
        sl = slice(c0, c1)
        h = allh[sl]
        out_chunk = _scatter_paint_jit(
            coefs, st.lt_min, st.lt_max, st.l5_min, st.l5_max, st.dlt, st.dl5,
            st.n0, st.n1, st.fill,
            jnp.asarray(vp[sl]), jnp.asarray(vec[h]), jnp.asarray(thmax[h]),
            jnp.asarray(log_th500[h]), jnp.asarray(amp[h]),
            jnp.asarray(allpix[sl]), npix)
        out_acc += np.asarray(out_chunk)
    t_gpu = time.time() - t0
    if verbose:
        print(f"  host disc loop {t_host_disc:.1f}s  pix2vec {t_pv:.1f}s  "
              f"gpu {t_gpu:.1f}s  contributions={M/1e6:.0f}M", flush=True)
    return out_acc
