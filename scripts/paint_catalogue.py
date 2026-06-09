#!/usr/bin/env python
"""Paint a Compton-y map from a benchmark catalogue (numpy/healpy reference path).

Usage: python scripts/paint_catalogue.py <index> [output.fits]
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import healpy as hp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from jxpaint.profiles.shape_table import load_beamed_table
from jxpaint.painting.healpix import paint_catalogue, paint_catalogue_gpu

CAT_DIR = "/rds/rds-lxu/tsz_project/tsz_catalogue_benchmark"
OUT_DIR = "/scratch/scratch-lxu/jxpaint/outputs"


def main():
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    idx = int(pos[0]) if pos else 0
    out_path = (pos[1] if len(pos) > 1
                else os.path.join(OUT_DIR, f"jxpaint_snr_{idx}_y0true.fits"))
    os.makedirs(OUT_DIR, exist_ok=True)
    cat = os.path.join(CAT_DIR, f"catalogue_bench_snr_{idx}.csv")
    df = pd.read_csv(cat)
    print(f"catalogue {idx}: {len(df)} halos", flush=True)
    use_cpu = "--cpu" in sys.argv
    st = load_beamed_table()
    t0 = time.time()
    if use_cpu:
        m = paint_catalogue(df.z.values, df.M.values, df.lon.values,
                            df.lat.values, df.y0_true.values, st, progress=True)
    else:
        m = paint_catalogue_gpu(df.z.values, df.M.values, df.lon.values,
                                df.lat.values, df.y0_true.values, st, verbose=True)
    dt = time.time() - t0
    print(f"painted in {dt:.1f}s; nonzero={np.count_nonzero(m)} "
          f"max={m.max():.6e} sum={m.sum():.6f}", flush=True)
    if os.path.exists(out_path):
        os.remove(out_path)
    hp.write_map(out_path, m, dtype=np.float64, overwrite=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
