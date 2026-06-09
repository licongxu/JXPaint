# JXPaint

GPU tSZ Compton-y map painting in Python + JAX, ported from
[XGPaint](https://github.com/WebSky-mocks/XGPaint) (Julia). Paints beam-convolved
Compton-y Healpix maps from halo catalogues, **bit-for-bit correct** vs XGPaint
reference maps and **6.9–7.4× faster**.

## Status (all phases passing)

| Phase | What | Verification |
|-------|------|--------------|
| 1 | CustomGNFW pressure profile (= hmfast ParametricGNFW; XGPaint Arnaud10 shape) | cosmology ≤7e-12, shape F(x) 1.4e-14, y_los 4.9e-9 vs XGPaint; y0_param/y0_arnaud 0/6.8e-11 vs hmfast; bicubic beamed table 6.6e-16 vs cached .jld2 |
| 2 | Healpix disc painting | maps i=0,1,2 vs `map_bench_snr_{i}_y0true.fits`: max pixel rel err ~7e-11, RMS ~1.5e-12, flux ~6e-15 |
| 3 | GPU acceleration | 16.8–18.0 s/catalogue vs XGPaint 123.75 s → 6.9–7.4× |

## Layout
- `src/jxpaint/constants.py`, `cosmology.py` — constants + FlatLCDM (matched to Cosmology.jl).
- `src/jxpaint/profiles/custom_gnfw.py` — gNFW shape, LOS integral, y0_param, y0_arnaud, y_los, theta500.
- `src/jxpaint/profiles/shape_table.py` — bicubic beam-convolved 2D shape table.
- `src/jxpaint/painting/{geometry,healpix}.py` — painting geometry + CPU/GPU painters.
- `scripts/{paint_catalogue,validate_map,benchmark_gpu}.py`.
- `tests/{test_phase1,test_phase2}.py`.
- `reference/` — Julia/hmfast reference generators and gold values.

## Run
```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
PYTHONPATH=src python tests/test_phase1.py          # Phase 1
python tests/test_phase2.py                         # Phase 2/3 fast checks
python scripts/paint_catalogue.py 0                 # GPU paint (add --cpu for CPU)
python scripts/benchmark_gpu.py                     # speedup vs XGPaint
```

## Physics notes
- `M` in catalogues is M500c (1e14 Msun); no mass conversion.
- Shape lookup uses `theta500 = angular_size(R_Δ(M,z,500)/B^(1/3), z)`, B=1.41.
- Disc cutoff `theta_max` uses the **unbiased Δ=200** radius (XGPaint painter default), not Δ=500/B^(1/3) — critical for matching massive-halo wings.
- Amplitude per halo: `A_halo = y0_true / B^(1/3)`; painted value `A_halo * y_t(log theta, log theta500)`.
- Per-pixel angle: `theta = acos(clamp(1 - ||v_pix-v_halo||^2/2, -1, 1))`, clamped to `theta_min=exp(-16.5)`, painted iff `theta < theta_max`.
