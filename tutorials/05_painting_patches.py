"""Tutorial 5: painting a 10 x 10 degree catalogue patch.

Run from the repository root:

    PYTHONPATH=src python tutorials/05_painting_patches.py

For a FLAMINGO-style CSV with Mass_Msun, Redshift, RA_deg, and DEC_deg columns,
you can also use the command-line script directly:

    PYTHONPATH=src python scripts/paint_catalogue.py --csv L2p8_m9_lightcone0_filtered_halos.csv --patch --center-ra-deg 0 --center-dec-deg 0 --width-deg 10 --height-deg 10 --npix 256 --output-path patch_00.npz

This file is intentionally a plain Python tutorial so it can be copied into
batch scripts more easily than a notebook.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jxpaint.painting.patch import paint_catalogue_patch, write_patch
from jxpaint.profiles.shape_table import load_beamed_table


CAT_DIR = "/rds/rds-lxu/tsz_project/tsz_catalogue_benchmark"
OUT_DIR = "/scratch/scratch-lxu/agent_dev/auto_research_agent/JXPaint/outputs"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cat_path = os.path.join(CAT_DIR, "catalogue_bench_snr_0.csv")
    df = pd.read_csv(cat_path)
    shape_table = load_beamed_table()

    patch, meta = paint_catalogue_patch(
        df.z.values,
        df.M.values,
        df.lon.values,
        df.lat.values,
        df.y0_true.values,
        shape_table,
        center_ra_deg=0.0,
        center_dec_deg=0.0,
        width_deg=10.0,
        height_deg=10.0,
        nx=256,
        ny=256,
        progress=True,
    )

    out_npz = os.path.join(OUT_DIR, "catalogue_bench_snr_0_patch_10x10deg_256px.npz")
    write_patch(out_npz, patch, meta)
    print(f"wrote {out_npz}")
    print(f"shape={patch.shape}, candidate_halos={meta['candidate_halos']}, "
          f"nonzero={np.count_nonzero(patch)}, max={patch.max():.6e}")

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(
        patch,
        origin="lower",
        extent=[meta["x_deg"][0], meta["x_deg"][-1],
                meta["y_deg"][0], meta["y_deg"][-1]],
        cmap="magma",
    )
    ax.set_xlabel("RA offset from centre [deg]")
    ax.set_ylabel("Dec offset from centre [deg]")
    ax.set_title("JXPaint 10 x 10 degree Compton-y patch")
    fig.colorbar(im, ax=ax, label="Compton-y")
    out_png = os.path.join(OUT_DIR, "catalogue_bench_snr_0_patch_10x10deg_256px.png")
    fig.savefig(out_png, dpi=160)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
