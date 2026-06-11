#!/usr/bin/env python
"""Stress-test the GPU-native painter: bit-for-bit across many catalogues,
stage profile, edge cases (poles / giant haloes), NaN/inf, timing stability.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import healpy as hp
import jax

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from jxpaint.profiles.shape_table import load_beamed_table
from jxpaint.painting.gpu_native import (paint_catalogue_gpu_native,
                                         build_ring_tables, disc_pixels,
                                         ring_pix2vec, _ring_tables_device)
from jxpaint.painting.healpix import paint_catalogue
from jxpaint.painting import geometry as geom
import jax.numpy as jnp

CAT = "/rds/rds-lxu/tsz_project/tsz_catalogue_benchmark/catalogue_bench_snr_{}.csv"
REF = "/rds/rds-lxu/tsz_project/tsz_benchmark_maps_scatter/map_bench_snr_{}_y0true.fits"
XGPAINT = 108.07


def validate_map(m, ref):
    assert np.all(np.isfinite(m)), "non-finite pixels!"
    assert m.min() >= 0.0, "negative pixels!"
    mask = np.abs(ref) > 1e-9
    rel = np.abs(m[mask] - ref[mask]) / np.abs(ref[mask])
    flux = abs(m.sum() - ref.sum()) / abs(ref.sum())
    return rel.max(), np.sqrt(np.mean(rel**2)), flux, np.count_nonzero(m), np.count_nonzero(ref)


def main():
    st = load_beamed_table()
    # warmup compile
    d = pd.read_csv(CAT.format(0), nrows=500)
    paint_catalogue_gpu_native(d.z.values, d.M.values, d.lon.values, d.lat.values,
                               d.y0_true.values, st)

    print("=== bit-for-bit across catalogues + timing ===")
    idxs = [0, 1, 2, 50, 137, 333, 500, 750, 999]
    worst = 0.0
    for i in idxs:
        if not os.path.exists(REF.format(i)):
            continue
        df = pd.read_csv(CAT.format(i))
        ts = []
        for _ in range(2):
            t0 = time.time()
            m = paint_catalogue_gpu_native(df.z.values, df.M.values, df.lon.values,
                                           df.lat.values, df.y0_true.values, st)
            ts.append(time.time() - t0)
        ref = hp.read_map(REF.format(i), dtype=np.float64)
        mx, rms, flux, nz, nzr = validate_map(m, ref)
        worst = max(worst, mx)
        ok = mx < 1e-5 and rms < 1e-6 and flux < 1e-5 and nz == nzr
        print(f"cat {i:3d}: warm {min(ts):.2f}s ({XGPAINT/min(ts):.0f}x)  "
              f"max_rel {mx:.2e} flux {flux:.2e} nz {'ok' if nz==nzr else f'{nz}!={nzr}'}  "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"worst max pixel rel err across all: {worst:.3e}\n")

    print("=== stage profile (cat 0) ===")
    df = pd.read_csv(CAT.format(0))
    nside = 1024
    Tdev = _ring_tables_device(nside)
    vec = geom.radec_to_vec(*geom.catalogue_to_radec(df.lon.values, df.lat.values))
    thmax = geom.theta_max_array(df.M.values, df.z.values)
    vec_d = jnp.asarray(vec); thmax_d = jnp.asarray(thmax)
    for _ in range(2):
        t0 = time.time()
        pix, halo = disc_pixels(vec_d, thmax_d, Tdev, nside)
        pix.block_until_ready()
        t_disc = time.time() - t0
    print(f"  disc_pixels (GPU): {t_disc:.3f}s  -> {pix.shape[0]/1e6:.0f}M candidate pixels")

    print("\n=== edge cases vs CPU reference painter (poles, giant, extreme z) ===")
    rng = np.random.default_rng(1)
    n = 200
    lat = np.concatenate([np.full(50, np.pi),       # dec=+90 (north pole)
                          np.full(50, 0.0),         # dec=-90 (south pole)
                          rng.uniform(0, np.pi, n - 100)])
    lon = rng.uniform(0, 2 * np.pi, n)
    M = np.concatenate([np.full(20, 26.0), rng.uniform(1, 5, n - 20)])  # giant + normal
    z = np.concatenate([np.full(10, 0.006), np.full(10, 2.8),
                        rng.uniform(0.05, 2.0, n - 20)])
    y0 = rng.uniform(1e-6, 1e-4, n)
    # artificial giant-heavy catalogue: raise the pixel caps accordingly
    m_gpu = paint_catalogue_gpu_native(z, M, lon, lat, y0, st,
                                       e_per_halo=300, n_per_halo=2000)
    m_cpu = paint_catalogue(z, M, lon, lat, y0, st)
    mask = m_cpu > 0
    rel = np.abs(m_gpu[mask] - m_cpu[mask]) / m_cpu[mask]
    extra = np.count_nonzero((m_gpu > 0) & (m_cpu == 0))
    missing = np.count_nonzero((m_gpu == 0) & (m_cpu > 0))
    print(f"  GPU vs CPU painter: max rel {rel.max():.2e}  extra px {extra}  missing px {missing}  "
          f"finite {np.all(np.isfinite(m_gpu))}  "
          f"{'PASS' if rel.max()<1e-10 and extra==0 and missing==0 else 'FAIL'}")

    print("\n=== memory / scale: 3x catalogue concatenated (~920k halos) ===")
    df = pd.read_csv(CAT.format(0))
    z3 = np.tile(df.z.values, 3); M3 = np.tile(df.M.values, 3)
    lon3 = np.tile(df.lon.values, 3); lat3 = np.tile(df.lat.values, 3)
    y03 = np.tile(df.y0_true.values, 3)
    t0 = time.time(); m3 = paint_catalogue_gpu_native(z3, M3, lon3, lat3, y03, st)
    print(f"  920k halos painted in {time.time()-t0:.2f}s  finite={np.all(np.isfinite(m3))}  "
          f"sum={m3.sum():.4f} (==3x single? {abs(m3.sum()-3*2.421744)/(.3*2.421744):.1e})")


if __name__ == "__main__":
    main()
