#!/usr/bin/env python
"""Paint a Compton-y map from a benchmark catalogue.

Default: fully-GPU painter (gpu_native).  --hybrid: GPU kernel + CPU disc
(multiprocess).  --cpu: numpy/healpy reference painter.

Usage: python scripts/paint_catalogue.py <index> [output.fits] [--cpu|--hybrid]
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
from jxpaint.painting.gpu_native import paint_catalogue_gpu_native

CAT_DIR = "/rds/rds-lxu/tsz_project/tsz_catalogue_benchmark"
OUT_DIR = "/scratch/scratch-lxu/agent_dev/auto_research_agent/JXPaint/outputs"


def main():
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    idx = int(pos[0]) if pos else 0
    out_path = (pos[1] if len(pos) > 1
                else os.path.join(OUT_DIR, f"jxpaint_snr_{idx}_y0true.fits"))
    os.makedirs(OUT_DIR, exist_ok=True)
    cat = os.path.join(CAT_DIR, f"catalogue_bench_snr_{idx}.csv")
    df = pd.read_csv(cat)
    print(f"catalogue {idx}: {len(df)} halos", flush=True)
    st = load_beamed_table()
    t0 = time.time()
    if "--cpu" in sys.argv:
        m = paint_catalogue(df.z.values, df.M.values, df.lon.values,
                            df.lat.values, df.y0_true.values, st, progress=True)
    elif "--hybrid" in sys.argv:
        m = paint_catalogue_gpu(df.z.values, df.M.values, df.lon.values,
                                df.lat.values, df.y0_true.values, st, verbose=True)
    else:  # default: fully-GPU painter
        m = paint_catalogue_gpu_native(df.z.values, df.M.values, df.lon.values,
                                       df.lat.values, df.y0_true.values, st)
    dt = time.time() - t0
    print(f"painted in {dt:.1f}s; nonzero={np.count_nonzero(m)} "
          f"max={m.max():.6e} sum={m.sum():.6f}", flush=True)
    if os.path.exists(out_path):
        os.remove(out_path)
    hp.write_map(out_path, m, dtype=np.float64, overwrite=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
