#!/usr/bin/env python
"""Validate a JXPaint map against an XGPaint reference map.

Usage: python scripts/validate_map.py <jxpaint.fits> <reference.fits>
Metrics (goal tolerances):
  max pixel rel error                 < 1e-5
  RMS rel error (y_ref > 1e-9)        < 1e-6
  total flux rel error                < 1e-5
"""
import sys
import numpy as np
import healpy as hp


def main():
    a = hp.read_map(sys.argv[1], dtype=np.float64)
    b = hp.read_map(sys.argv[2], dtype=np.float64)
    assert a.shape == b.shape, f"shape mismatch {a.shape} {b.shape}"

    diff = a - b
    absdiff = np.abs(diff)

    # max pixel rel error over pixels where ref is meaningfully nonzero
    mask = np.abs(b) > 1e-9
    relpix = absdiff[mask] / np.abs(b[mask])
    max_rel = relpix.max()
    rms_rel = np.sqrt(np.mean(relpix ** 2))

    flux_a, flux_b = a.sum(), b.sum()
    flux_rel = abs(flux_a - flux_b) / abs(flux_b)

    # global absolute diagnostics
    print(f"pixels: {a.size}  ref nonzero(>1e-9): {mask.sum()}")
    print(f"max |abs diff|            = {absdiff.max():.3e}")
    print(f"max pixel rel error       = {max_rel:.3e}   (tol 1e-5)  "
          f"{'PASS' if max_rel < 1e-5 else 'FAIL'}")
    print(f"RMS rel error (y>1e-9)    = {rms_rel:.3e}   (tol 1e-6)  "
          f"{'PASS' if rms_rel < 1e-6 else 'FAIL'}")
    print(f"total flux rel error      = {flux_rel:.3e}   (tol 1e-5)  "
          f"{'PASS' if flux_rel < 1e-5 else 'FAIL'}")
    print(f"  flux jxpaint={flux_a:.8f} ref={flux_b:.8f}")

    # worst pixel detail
    iworst = np.argmax(absdiff * mask)
    print(f"worst pixel {iworst}: jx={a[iworst]:.6e} ref={b[iworst]:.6e} "
          f"reldiff={absdiff[iworst]/max(abs(b[iworst]),1e-300):.3e}")
    ok = (max_rel < 1e-5) and (rms_rel < 1e-6) and (flux_rel < 1e-5)
    print("RESULT:", "ALL PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
