#!/usr/bin/env python
"""Paint a Compton-y map from a benchmark catalogue.

Default: fully-GPU painter (gpu_native).  --hybrid: GPU kernel + CPU disc
(multiprocess).  --cpu: numpy/healpy reference painter.  --patch paints a
rectangular flat-sky cutout (for example 10 x 10 degrees).

Usage:
  python scripts/paint_catalogue.py <index> [output.fits] [--cpu|--hybrid]
  python scripts/paint_catalogue.py <index> --patch --center-ra-deg 0 --center-dec-deg 0 --npix 256
  python scripts/paint_catalogue.py --csv halos.csv --profile arnaud-b1 --patch --center-ra-deg 0 --center-dec-deg 0 --npix 256 --output-path patch_00.npz
"""
import argparse
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
from jxpaint.painting.patch import paint_catalogue_patch, write_patch
from jxpaint.profiles.custom_gnfw import (
    ArnaudGNFWPressureProfile,
    CustomGNFWPressureProfile,
)
from jxpaint import constants as C

CAT_DIR = "/rds/rds-lxu/tsz_project/tsz_catalogue_benchmark"
OUT_DIR = "/scratch/scratch-lxu/agent_dev/auto_research_agent/JXPaint/outputs"


def selected_bias(args):
    if args.bias_B is not None:
        return args.bias_B
    if args.profile == "arnaud-b1":
        return 1.0
    return C.B_BIAS


def selected_profile(args):
    if args.profile == "arnaud-b1":
        return ArnaudGNFWPressureProfile()
    return CustomGNFWPressureProfile(B=selected_bias(args))


def load_catalogue(args):
    if args.csv is None:
        cat = os.path.join(CAT_DIR, f"catalogue_bench_snr_{args.index}.csv")
        return pd.read_csv(cat), f"catalogue {args.index}"

    df = pd.read_csv(args.csv)
    out = pd.DataFrame()
    if "z" in df:
        out["z"] = df["z"]
    elif "Redshift" in df:
        out["z"] = df["Redshift"]
    else:
        raise ValueError("CSV needs a z or Redshift column")

    if "M" in df:
        out["M"] = df["M"]
    elif "Mass_Msun" in df:
        out["M"] = df["Mass_Msun"] / 1e14
    else:
        raise ValueError("CSV needs an M (1e14 Msun) or Mass_Msun column")

    if "lon" in df and "lat" in df:
        out["lon"] = df["lon"]
        out["lat"] = df["lat"]
    elif "RA_deg" in df and "DEC_deg" in df:
        out["lon"] = np.deg2rad(df["RA_deg"].to_numpy())
        out["lat"] = np.deg2rad(df["DEC_deg"].to_numpy()) + np.pi / 2.0
    else:
        raise ValueError("CSV needs lon/lat in radians or RA_deg/DEC_deg columns")

    if "y0_true" in df:
        out["y0_true"] = df["y0_true"]
    else:
        profile = selected_profile(args)
        out["y0_true"] = np.asarray(profile.y0_arnaud(
            out["M"].to_numpy(), out["z"].to_numpy()))
    return out, args.csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", nargs="?", type=int, default=0)
    parser.add_argument("output", nargs="?")
    parser.add_argument("--csv", help="input CSV; supports M/z/lon/lat/y0_true or FLAMINGO Mass_Msun/Redshift/RA_deg/DEC_deg")
    parser.add_argument("--output-path", help="output path, useful with --csv")
    parser.add_argument("--profile", choices=["xgpaint", "arnaud-b1"], default="xgpaint",
                        help="pressure normalization/profile for CSV y0 generation")
    parser.add_argument("--bias-B", type=float, help="mass-bias B used in y0 generation and painting geometry")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--cpu", action="store_true")
    mode.add_argument("--hybrid", action="store_true")
    mode.add_argument("--patch", action="store_true")
    parser.add_argument("--center-ra-deg", type=float)
    parser.add_argument("--center-dec-deg", type=float)
    parser.add_argument("--width-deg", type=float, default=10.0)
    parser.add_argument("--height-deg", type=float, default=10.0)
    parser.add_argument("--pixel-size-arcmin", type=float, default=1.0)
    parser.add_argument("--npix", type=int, help="square patch pixel count, e.g. 256")
    parser.add_argument("--nx", type=int, help="patch pixel count in x/RA offset")
    parser.add_argument("--ny", type=int, help="patch pixel count in y/Dec offset")
    args = parser.parse_args()

    idx = args.index
    out_path = args.output_path or args.output
    if out_path is None:
        pix_tag = f"_{args.npix}px" if args.patch and args.npix else ""
        suffix = (f"patch_{args.width_deg:g}x{args.height_deg:g}deg{pix_tag}.npz"
                  if args.patch else "y0true.fits")
        out_path = os.path.join(OUT_DIR, f"jxpaint_snr_{idx}_{suffix}")
    os.makedirs(OUT_DIR, exist_ok=True)
    bias_B = selected_bias(args)
    df, label = load_catalogue(args)
    print(f"{label}: {len(df)} halos  profile={args.profile}  B={bias_B:g}", flush=True)
    st = load_beamed_table()
    t0 = time.time()
    if args.patch:
        if args.center_ra_deg is None or args.center_dec_deg is None:
            raise SystemExit("--patch requires --center-ra-deg and --center-dec-deg")
        nx = args.npix if args.npix is not None else args.nx
        ny = args.npix if args.npix is not None else args.ny
        m, meta = paint_catalogue_patch(
            df.z.values, df.M.values, df.lon.values, df.lat.values,
            df.y0_true.values, st, center_ra_deg=args.center_ra_deg,
            center_dec_deg=args.center_dec_deg, width_deg=args.width_deg,
            height_deg=args.height_deg, pixel_size_arcmin=args.pixel_size_arcmin,
            nx=nx, ny=ny, progress=True, bias_B=bias_B)
        dt = time.time() - t0
        print(f"painted patch in {dt:.1f}s; shape={m.shape} "
              f"candidate_halos={meta['candidate_halos']} "
              f"nonzero={np.count_nonzero(m)} max={m.max():.6e} "
              f"sum={m.sum():.6f}", flush=True)
        write_patch(out_path, m, meta)
        print(f"wrote {out_path}", flush=True)
        return
    if args.cpu:
        m = paint_catalogue(df.z.values, df.M.values, df.lon.values,
                            df.lat.values, df.y0_true.values, st, progress=True,
                            bias_B=bias_B)
    elif args.hybrid:
        m = paint_catalogue_gpu(df.z.values, df.M.values, df.lon.values,
                                df.lat.values, df.y0_true.values, st, verbose=True,
                                bias_B=bias_B)
    else:  # default: fully-GPU painter
        m = paint_catalogue_gpu_native(df.z.values, df.M.values, df.lon.values,
                                       df.lat.values, df.y0_true.values, st,
                                       bias_B=bias_B)
    dt = time.time() - t0
    print(f"painted in {dt:.1f}s; nonzero={np.count_nonzero(m)} "
          f"max={m.max():.6e} sum={m.sum():.6f}", flush=True)
    if os.path.exists(out_path):
        os.remove(out_path)
    hp.write_map(out_path, m, dtype=np.float64, overwrite=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
