#!/usr/bin/env python
"""Paint + validate catalogues {0,1,2} via the GPU painter; report speedup.

XGPaint baseline (reference/julia/time_xgpaint.jl, 8 threads): 123.75 s/catalogue.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import healpy as hp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from jxpaint.profiles.shape_table import load_beamed_table
from jxpaint.painting.healpix import paint_catalogue_gpu

CAT = "/rds/rds-lxu/tsz_project/tsz_catalogue_benchmark/catalogue_bench_snr_{}.csv"
REF = "/rds/rds-lxu/tsz_project/tsz_benchmark_maps_scatter/map_bench_snr_{}_y0true.fits"
# Ground truth: the actual production painter paint_a10_y0true_2d_mpi.jl
# self-reports "painted in 108.07 seconds" for catalogue 0 (8 threads, single
# rank), excluding the one-time 512MB shape-cache load.  (A faithful Julia
# re-implementation timed 123.75 s; we cite the lower, real number.)
XGPAINT_SECONDS = 108.07


def validate(m, ref):
    mask = np.abs(ref) > 1e-9
    rel = np.abs(m[mask] - ref[mask]) / np.abs(ref[mask])
    flux = abs(m.sum() - ref.sum()) / abs(ref.sum())
    return rel.max(), np.sqrt(np.mean(rel**2)), flux


def main():
    st = load_beamed_table()
    # warmup (compile)
    d = pd.read_csv(CAT.format(0), nrows=500)
    paint_catalogue_gpu(d.z.values, d.M.values, d.lon.values, d.lat.values,
                        d.y0_true.values, st)
    print(f"XGPaint baseline: {XGPAINT_SECONDS:.1f} s/catalogue (8 threads)\n")
    for i in (0, 1, 2):
        df = pd.read_csv(CAT.format(i))
        t0 = time.time()
        m = paint_catalogue_gpu(df.z.values, df.M.values, df.lon.values,
                                df.lat.values, df.y0_true.values, st, verbose=True)
        dt = time.time() - t0
        ref = hp.read_map(REF.format(i), dtype=np.float64)
        mx, rms, flux = validate(m, ref)
        ok = mx < 1e-5 and rms < 1e-6 and flux < 1e-5
        print(f"cat {i}: wall {dt:.1f}s  speedup {XGPAINT_SECONDS/dt:.1f}x  | "
              f"max_rel {mx:.2e} rms {rms:.2e} flux {flux:.2e}  "
              f"{'PASS' if ok else 'FAIL'}\n")


if __name__ == "__main__":
    main()
