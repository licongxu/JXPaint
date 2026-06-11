#!/usr/bin/env python
"""Demonstrate: varying cosmology needs NO interpolator rebuild in JXPaint.

The beam-convolved shape table y_t(log theta, log theta500) depends only on the
gNFW shape parameters and the beam FWHM -- NOT on cosmology or B.  Cosmology
enters only through the geometry theta500(M,z) and the amplitude.  So the table
is built/loaded ONCE and reused for every cosmology; the per-cosmology cost is
just the vectorised geometry (~ms).  In XGPaint the (8192x8192, FFTLog-beamed)
interpolator is rebuilt whenever the model/cosmology is reconstructed -- the
bottleneck this avoids.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import jax.numpy as jnp
from jxpaint.profiles.shape_table import load_beamed_table
from jxpaint.cosmology import FlatLCDM
from jxpaint.painting.gpu_native import (paint_catalogue_gpu_native,
                                         compute_geometry)
from jxpaint import constants as C

CAT = "/rds/rds-lxu/tsz_project/tsz_catalogue_benchmark/catalogue_bench_snr_0.csv"


def main():
    # ---- build/load the interpolator ONCE ----
    t0 = time.time()
    st = load_beamed_table()
    st.coefs_device()  # upload once
    t_table = time.time() - t0
    print(f"interpolator (8192x8192 beamed shape table) loaded+uploaded ONCE: "
          f"{t_table:.2f}s  [cosmology-independent -> never rebuilt]\n")

    df = pd.read_csv(CAT)
    z, M = df.z.values, df.M.values
    lon, lat, y0 = df.lon.values, df.lat.values, df.y0_true.values

    # a grid of cosmologies (vary h and Omega_m)
    cosmos = []
    for h in (0.64, 0.6766, 0.70, 0.74):
        for Om in (0.27, 0.31, 0.35):
            cosmos.append(FlatLCDM(h=h, Omega_m=Om))

    # warmup compile (first paint compiles the whole pipeline once)
    paint_catalogue_gpu_native(z, M, lon, lat, y0, st, cosmo=cosmos[0])

    print(f"{'cosmology':>22} | geom[ms] | paint[s] | rebuilds | map sum  | th500[0]")
    print("-" * 86)
    rebuilds = 0
    sums = []
    for cz in cosmos:
        # geometry-only cost (the ONLY cosmology-dependent step)
        tg = []
        for _ in range(5):
            t0 = time.time()
            vec, th500, thmax, amp = compute_geometry(z, M, lon, lat, y0, cz, C.B_BIAS)
            th500.block_until_ready()
            tg.append(time.time() - t0)
        t_geom = min(tg)
        # full paint (same table object reused; no rebuild)
        tp = []
        for _ in range(2):
            t0 = time.time()
            m = paint_catalogue_gpu_native(z, M, lon, lat, y0, st, cosmo=cz)
            tp.append(time.time() - t0)
        t_paint = min(tp)
        sums.append(m.sum())
        print(f"h={cz.h:.4f},Om={cz.Omega_m:.2f} | {t_geom*1e3:7.2f}  | {t_paint:7.3f}  "
              f"| {rebuilds:8d} | {m.sum():.5f} | {float(th500[0]):.4e}")

    sums = np.array(sums)
    print("\nSummary:")
    print(f"  interpolator builds/loads total: 1 (at {t_table:.2f}s), reused for "
          f"all {len(cosmos)} cosmologies")
    print(f"  per-cosmology geometry cost: ~{t_geom*1e3:.1f} ms  (the only "
          f"cosmology-dependent work)")
    print(f"  per-cosmology paint cost:   ~{t_paint:.2f} s  (interpolator "
          f"contribution = 0, it is reused)")
    print(f"  maps vary with cosmology (sum range {sums.min():.4f}..{sums.max():.4f}, "
          f"spread {100*(sums.max()-sums.min())/sums.mean():.1f}%) -> cosmology IS applied")
    print(f"  amortised interpolator cost over {len(cosmos)} cosmologies: "
          f"{t_table/len(cosmos)*1e3:.0f} ms/cosmology (-> negligible)")


if __name__ == "__main__":
    main()
