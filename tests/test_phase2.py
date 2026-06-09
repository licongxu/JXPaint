"""Phase 2/3 fast checks: geometry + painter self-consistency on a subset.

The authoritative validation is the full-catalogue bit-for-bit match vs
map_bench_snr_{0,1,2}_y0true.fits (scripts/benchmark_gpu.py: max pixel rel err
~7e-11, RMS ~1.5e-12, flux ~6e-15, all PASS; 6.9-7.4x faster than XGPaint).
This module runs a cheap subset check that the GPU and CPU painters agree and
that geometry constants match the painter.  Run: python tests/test_phase2.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

CAT = "/rds/rds-lxu/tsz_project/tsz_catalogue_benchmark/catalogue_bench_snr_0.csv"


def test_theta_max_uses_unbiased_delta200():
    """theta_max theta2 must use unbiased Delta=200 radius, not Delta=500/B^(1/3)."""
    from jxpaint.painting import geometry as geom
    from jxpaint import constants as C
    # massive low-z halo where theta2 dominates the floor
    M, z = 26.0, 0.05
    tmax = float(geom.theta_max_array(np.array([M]), np.array([z]))[0])
    th500 = float(geom.theta500_array(np.array([M]), np.array([z]))[0])
    # the floor is 20 arcmin; this halo must exceed it via the Delta=200 radius
    floor = 2 * C.FWHM_ARCMIN * (np.pi / 10800.0)
    assert tmax > floor, "massive halo theta_max should exceed the beam floor"
    # biased theta2 (wrong) would be 4*th500; correct uses larger Delta=200 radius
    assert tmax > 4 * th500, "theta_max must use unbiased Delta=200 radius (larger)"
    print(f"  theta_max={np.degrees(tmax):.3f}deg > 4*theta500={np.degrees(4*th500):.3f}deg  OK")


def test_gpu_cpu_painter_agree():
    """GPU and CPU painters produce identical maps on a halo subset."""
    if not os.path.exists(CAT):
        print("  SKIP: catalogue not present"); return
    import pandas as pd
    from jxpaint.profiles.shape_table import load_beamed_table
    from jxpaint.painting.healpix import paint_catalogue, paint_catalogue_gpu
    df = pd.read_csv(CAT, nrows=3000)
    st = load_beamed_table()
    a = paint_catalogue(df.z.values, df.M.values, df.lon.values, df.lat.values,
                        df.y0_true.values, st)
    b = paint_catalogue_gpu(df.z.values, df.M.values, df.lon.values,
                            df.lat.values, df.y0_true.values, st)
    mask = np.abs(a) > 0
    rel = np.abs(a[mask] - b[mask]) / np.abs(a[mask])
    print(f"  GPU vs CPU painter max rel err = {rel.max():.3e} (npix={mask.sum()})")
    assert rel.max() < 1e-12


if __name__ == "__main__":
    fails = 0
    for fn in [test_theta_max_uses_unbiased_delta200, test_gpu_cpu_painter_agree]:
        try:
            print(f"[{fn.__name__}]"); fn(); print("  PASS")
        except AssertionError as e:
            fails += 1; print(f"  FAIL: {e}")
    print("\n", "ALL PASS" if fails == 0 else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
