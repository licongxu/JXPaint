"""Demo: Arnaud gNFW profile with B=1 for full-sky and patch painting.

This is the profile requested for Celia's FLAMINGO catalogue workflow.  It uses
Arnaud gNFW normalization with no hydrostatic mass bias (`B=1`) and works with a
CSV containing `Mass_Msun`, `Redshift`, `RA_deg`, and `DEC_deg` columns.

Examples from the repository root:

    # Full-sky HEALPix map.  Uses the fast GPU painter by default.
    PYTHONPATH=src python scripts/paint_catalogue.py       --csv L2p8_m9_lightcone0_filtered_halos.csv       --profile arnaud-b1       --output-path flamingo_arnaud_b1_fullsky.fits

    # One 10 deg x 10 deg patch with 256 x 256 pixels.
    PYTHONPATH=src python scripts/paint_catalogue.py       --csv L2p8_m9_lightcone0_filtered_halos.csv       --profile arnaud-b1       --patch       --center-ra-deg 0       --center-dec-deg 0       --width-deg 10       --height-deg 10       --npix 256       --output-path flamingo_arnaud_b1_patch_00.npz

For 10 patches, repeat the patch command with different `--center-ra-deg`,
`--center-dec-deg`, and `--output-path` values.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jxpaint.painting.patch import paint_catalogue_patch, write_patch
from jxpaint.profiles.custom_gnfw import ArnaudGNFWPressureProfile
from jxpaint.profiles.shape_table import load_beamed_table


CSV = "L2p8_m9_lightcone0_filtered_halos.csv"
OUT_DIR = "/scratch/scratch-lxu/agent_dev/auto_research_agent/JXPaint/outputs"


def load_flamingo_csv(path):
    raw = pd.read_csv(path)
    profile = ArnaudGNFWPressureProfile()
    out = pd.DataFrame()
    out["z"] = raw["Redshift"]
    out["M"] = raw["Mass_Msun"] / 1e14
    out["lon"] = np.deg2rad(raw["RA_deg"].to_numpy())
    out["lat"] = np.deg2rad(raw["DEC_deg"].to_numpy()) + np.pi / 2.0
    out["y0_true"] = np.asarray(profile.y0_arnaud(out["M"].to_numpy(),
                                                   out["z"].to_numpy()))
    return out


def paint_demo_patch(csv_path=CSV, center_ra_deg=0.0, center_dec_deg=0.0):
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_flamingo_csv(csv_path)
    shape_table = load_beamed_table()
    patch, meta = paint_catalogue_patch(
        df.z.values,
        df.M.values,
        df.lon.values,
        df.lat.values,
        df.y0_true.values,
        shape_table,
        center_ra_deg=center_ra_deg,
        center_dec_deg=center_dec_deg,
        width_deg=10.0,
        height_deg=10.0,
        nx=256,
        ny=256,
        bias_B=1.0,
        progress=True,
    )
    out_npz = os.path.join(OUT_DIR, "flamingo_arnaud_b1_patch_00.npz")
    write_patch(out_npz, patch, meta)
    print(f"wrote {out_npz}")

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
    ax.set_title("Arnaud gNFW B=1, 10 x 10 deg patch")
    fig.colorbar(im, ax=ax, label="Compton-y")
    out_png = os.path.join(OUT_DIR, "flamingo_arnaud_b1_patch_00.png")
    fig.savefig(out_png, dpi=160)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    if not os.path.exists(CSV):
        raise SystemExit(
            f"Put {CSV!r} in this directory, or use the command examples in "
            "this file with --csv pointing to your catalogue."
        )
    paint_demo_patch(CSV)
