# JXPaint

GPU tSZ Compton-y map painting in Python + JAX, ported from
[XGPaint](https://github.com/WebSky-mocks/XGPaint) (Julia). Paints beam-convolved
Compton-y Healpix maps from halo catalogues, **bit-for-bit correct** vs XGPaint
reference maps and **~550–600× faster** (≈0.18 s vs the production painter's
108 s/catalogue) with the fully-GPU painter.

**Varying cosmology needs no interpolator rebuild.** The beam-convolved shape
table depends only on the gNFW shape + beam FWHM, not on cosmology — so it is
built once and reused for every cosmology (per-cosmology cost ≈6 ms of geometry;
0 rebuilds).  `scripts/cosmology_demo.py`.

## Status (all phases passing)

| Phase | What | Verification |
|-------|------|--------------|
| 1 | CustomGNFW pressure profile (= hmfast ParametricGNFW; XGPaint Arnaud10 shape) | cosmology ≤7e-12, shape F(x) 1.4e-14, y_los 4.9e-9 vs XGPaint; y0_param/y0_arnaud 0/6.8e-11 vs hmfast; bicubic beamed table 6.6e-16 vs cached .jld2 |
| 2 | Healpix disc painting | maps i=0,1,2 vs `map_bench_snr_{i}_y0true.fits`: max pixel rel err ~7e-11, RMS ~1.5e-12, flux ~6e-15 |
| 3 | Fully-GPU painter (`gpu_native`) | 0.18 s/catalogue vs production XGPaint 108.07 s → **~590×**; bit-for-bit (max rel ~1e-10) across catalogues 0–999 |
| 4 | Cosmology variation | interpolator built once, 0 rebuilds; ~6 ms geometry/cosmology |

## Layout
- `src/jxpaint/constants.py`, `cosmology.py` — constants + FlatLCDM (matched to Cosmology.jl).
- `src/jxpaint/profiles/custom_gnfw.py` — gNFW shape, LOS integral, y0_param, y0_arnaud, y_los, theta500.
- `src/jxpaint/profiles/shape_table.py` — bicubic beam-convolved 2D shape table.
- `src/jxpaint/painting/geometry.py` — geometry (numpy mirror cosmology).
- `src/jxpaint/painting/healpix.py` — CPU + hybrid (GPU kernel, multiprocess disc) painters.
- `src/jxpaint/painting/gpu_native.py` — **fully-GPU painter** (RING pix2vec + disc-finding + bicubic + scatter, jitted fixed-size kernel).
- `src/jxpaint/painting/patch.py` — rectangular flat-sky catalogue cutouts such as 10 x 10 degree patches.
- `scripts/{paint_catalogue,validate_map,benchmark_gpu,stress_test,cosmology_demo}.py`.
- `tests/{test_phase1,test_phase2}.py`.
- `tutorials/` — Jupyter notebooks (quickstart, profiles, speed + cosmology).
- `reference/` — Julia/hmfast reference generators and gold values.

## Run
```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
PYTHONPATH=src python tests/test_phase1.py          # Phase 1
python tests/test_phase2.py                         # painter + cosmology checks
python scripts/paint_catalogue.py 0                 # fully-GPU paint (--hybrid, --cpu)
python scripts/paint_catalogue.py 0 --patch --center-ra-deg 0 --center-dec-deg 0 --npix 256
python scripts/paint_catalogue.py --csv halos.csv --patch --center-ra-deg 0 --center-dec-deg 0 --npix 256 --output-path patch_00.npz
python scripts/paint_catalogue.py --csv halos.csv --profile arnaud-b1 --output-path arnaud_b1_fullsky.fits
python scripts/paint_catalogue.py --csv halos.csv --profile arnaud-b1 --patch --center-ra-deg 0 --center-dec-deg 0 --npix 256 --output-path arnaud_b1_patch_00.npz
python scripts/stress_test.py                       # bit-for-bit + edge cases + scale
python scripts/cosmology_demo.py                    # interpolator reused across cosmologies
```

## Physics notes
- `M` in catalogues is M500c (1e14 Msun); no mass conversion.
- Shape lookup uses `theta500 = angular_size(R_Δ(M,z,500)/B^(1/3), z)`, B=1.41.
- Disc cutoff `theta_max` uses the **unbiased Δ=200** radius (XGPaint painter default), not Δ=500/B^(1/3) — critical for matching massive-halo wings.
- Amplitude per halo: `A_halo = y0_true / B^(1/3)`; painted value `A_halo * y_t(log theta, log theta500)`.
- Per-pixel angle: `theta = acos(clamp(1 - ||v_pix-v_halo||^2/2, -1, 1))`, clamped to `theta_min=exp(-16.5)`, painted iff `theta < theta_max`.
