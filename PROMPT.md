# Goal — JXPaint: GPU tSZ map painting in Python + JAX

Build **JXPaint** to paint Compton-$y$ maps from halo catalogues. Must be **much faster than XGPaint** (JAX + GPU: JIT, `vmap`, batched painting) and **always correct** vs XGPaint reference maps.

- **Stack:** Python + JAX only (no Julia in deliverable). Read Julia for physics/validation.
- **Paths:** absolute paths everywhere.
- **Physics:** `jax_enable_x64 = True`. GPU by default; no silent CPU fallback.

## References (read-only)

| Role | Path |
|------|------|
| XGPaint (truth) | `/home/lxu/.julia/dev/XGPaint` |
| Prior JAX code | `/scratch/scratch-lxu/painting_code/GODMAX` |
| Custom GNFW (hmfast) | `/scratch/scratch-lxu/agent_dev/auto_research_agent/hmfast/src/hmfast/halos/profiles/pressure.py` |
| Reference maps | `/rds/rds-lxu/tsz_project/tsz_benchmark_maps_scatter/map_bench_snr_{i}_y0true.fits` |
| Catalogues | `/rds/rds-lxu/tsz_project/tsz_catalogue_benchmark/catalogue_bench_snr_{i}.csv` |
| Julia painter | `/scratch/scratch-lxu/tsz_cnc_scatter/paint_with_scatter/paint_a10_y0true_2d_mpi.jl` |

Catalogue: `z, M, lon, lat, snr, snr_true, y0_true, amp_noscatter`. `M` in $10^{14} M_\odot$ (= $M_{500c}$). Coords: `dec = lat - π/2`, `ra = rem(lon + π, 2π) - π`.

## Phase 1 — Custom GNFW pressure profile (first)

Implement `CustomGNFWPressureProfile` (= hmfast `ParametricGNFWPressureProfile`; XGPaint shape = `Arnauld10ThermalSZProfile`).

**gNFW shape (Arnaud 2010):** defaults `P0=8.130, c500=1.156, alpha=1.0620, beta=5.4807, gamma=0.3292, B=1.41`.

**Parametric amplitude:** $y_0^{\rm param} = 10^{A_{\rm SZ}} (M_{500c} h/B / 0.7{\times}3{\times}10^{14})^{\alpha_{\rm SZ}} E(z)^2 (h/0.7)^{-1/2}$ with defaults `A_SZ=-4.97`, `alpha_SZ=0.7867`. Rescale by `ratio = y0_param / y0_arnaud` (use `shape_integral=0.470502095` from hmfast).

**Benchmark painting mode:** $y(\theta) = A_{\rm halo}\, y_t(\theta, \theta_{500})$ with $A_{\rm halo} = \texttt{y0\_true} / B^{1/3}$ and $y_t$ a precomputed **10 arcmin beam-convolved** 2D table in $(\log\theta, \log\theta_{500})$.

**Cosmology:** `h=0.6766, Ob0h2=0.02242, Oc0h2=0.1193`.

Deliver: JIT-friendly JAX PyTree; `y_los`, `y0_param`, `y0_arnaud`, `theta500`, `build_beamed_shape_table`, `evaluate_shape`.

**Phase 1 tests (pass before Phase 2):** vs hmfast rel err $<10^{-6}$; vs XGPaint 1D $<10^{-5}$; 2D shape table vs cached `.jld2` $<10^{-4}$.

## Phase 2 — Healpix painting

Match Julia painter: Healpix RING `nside=1024`; disc painting via 3D chord distance; $\theta_{\max}=\min(\texttt{compute\_θmax}, 5°)$. Output to `/scratch/scratch-lxu/agent_dev/auto_research_agent/JXPaint/outputs/`. Target **≥5× speedup** vs XGPaint on `catalogue_bench_snr_0.csv`.

## Validation (XGPaint = reference)

For `i ∈ {0,1,2}`: compare JXPaint map to `map_bench_snr_{i}_y0true.fits`.

| Metric | Tolerance |
|--------|-----------|
| max pixel rel error | $< 10^{-5}$ |
| RMS rel error ($y_{\rm ref}>10^{-9}$) | $< 10^{-6}$ |
| total flux | rel err $< 10^{-5}$ |

Fix geometry ($R_{500}$, $B^{1/3}$, beam, $\theta_{\min}$) before optimizing speed.

## Layout

`src/jxpaint/{cosmology,constants,profiles/custom_gnfw,profiles/shape_table,painting/{healpix,geometry}}`, `tests/`, `scripts/{paint_catalogue,validate_map}.py`.

## Out of scope

CIB/kSZ/RSZ, hmfast HaloModel, foregrounds, masking, large FITS in git.

## Workflow

1. Read XGPaint (`profiles_y.jl`, `a10_fast_profile.jl`, `profiles.jl`).
2. Phase 1 profile + tests → pass.
3. Phase 2 paint → validate maps `i=0,1,2`.
4. Then GPU optimizations. Log runs in `progress.md`.

**Rule:** If unsure about $M_{500}$, radius units, or `y0_true` meaning, resolve against Julia reference first.
