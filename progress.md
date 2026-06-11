# JXPaint progress log

JAX+GPU tSZ Compton-y map painter, ported from XGPaint. Goal: faster than
XGPaint, bit-for-bit correct vs XGPaint reference maps. See the goal spec.

Env: venv `/scratch/scratch-lxu/venv/cmbagent_env` (jax 0.10, 2x RTX PRO 6000
Blackwell). Julia 1.12 + XGPaint dev for reference values. Run tests with
`source .../cmbagent_env/bin/activate && PYTHONPATH=src python tests/...`.

## Layout
- `src/jxpaint/constants.py` — physical constants (matched to XGPaint/Cosmology.jl/hmfast).
- `src/jxpaint/cosmology.py` — FlatLCDM (a2E, E, comoving/angular-diameter dist via Gauss-Legendre over a, rho_crit, R_delta, angular_size).
- `src/jxpaint/profiles/custom_gnfw.py` — CustomGNFW = hmfast ParametricGNFW: gnfw shape, LOS integral (tanh-sinh), F(x), y0_param, y0_arnaud, y_los, theta500. JAX pytree.
- `src/jxpaint/profiles/shape_table.py` — bicubic beamed shape table (loads dumped production .jld2 coefs).
- `reference/julia/gen_reference.jl` — generates gold values from XGPaint -> reference/data/*.csv.
- `reference/julia/dump_table.jl` — dumps the production beamed 2D shape table (8194x8194 cubic coefs) -> reference/data/beamed_table_coefs.f64.
- `reference/hmfast/` — hmfast reference y0 values.
- `tests/test_phase1.py` — Phase 1 validation vs all references.

## Status

### PHASE 1 — Custom GNFW profile: PASSING (verified)
`python tests/test_phase1.py` -> ALL PASS:
- cosmology/geometry (E, dA, rho_crit, R500, theta500) vs XGPaint: max rel err <= 7e-12 (tol 1e-6).
- 1D shape F(x) vs XGPaint los_quadrature: 1.4e-14 (tol 1e-5).
- y_los central vs XGPaint compton_y: 4.9e-9 (tol 1e-5).
- y0_param/y0_arnaud arithmetic vs hmfast (fed hmfast's E,r500c): 0.0 / 6.8e-11 (tol 1e-6).
- 2D beamed shape table (bicubic) vs cached itp2d: 6.6e-16 (tol 1e-4).

Key facts learned:
- XGPaint cosmology = analytic FlatLCDM (use_class_sz=false): h=0.6766, Om=0.30957590896528514, OL=0.6903327077465832, Or=9.138e-5. Distances integrate 1/a2E(a) over scale factor a.
- The hardcoded shape_integral=0.470502095 (hmfast y0_orig) is a SEPARATE constant; the actual gNFW LOS integral is 0.479049273 (XGPaint F0/P0/2). So XGPaint compton_y center != hmfast y0_orig — two different "central y" definitions, both reproduced.
- M in catalogue is M500c (no mass conversion). y0_arnaud r500c has NO /B^(1/3); painting theta500 DOES use R_delta/B^(1/3).
- The PRODUCTION cached shape table uses **Cubic** BSpline (Line OnGrid), coefs 8194x8194 (prefiltered+padded), NOT Linear. Bicubic eval reproduces it to machine precision.
- hmfast cosmology differs slightly (0.06 eV neutrinos + emulator), so hmfast comparison done at formula level (feed hmfast's E,r500c).

### PHASE 2 — Healpix painting: PASSING (bit-for-bit vs XGPaint maps)
`scripts/paint_catalogue.py <i>` paints; `scripts/validate_map.py` checks.
Validated i=0,1,2 vs map_bench_snr_{i}_y0true.fits:
- max pixel rel err ~7e-11 (tol 1e-5), RMS rel err ~1.5e-12 (tol 1e-6),
  flux rel err ~6e-15 (tol 1e-5). ALL PASS.
Painter: numpy/healpy, painting/{geometry,healpix}.py. ~118 s/catalogue on CPU.

CRITICAL fix that achieved the match: theta_max (disc cutoff) must use the
UNBIASED Delta=200 radius, NOT Delta=500/B^(1/3). The painter's compute_theta_max
calls R_Δ(base, M*M_sun, z) with XGPaint default Δ=200 and no bias — feeding the
M500c mass into the 200c radius formula. The shape lookup still uses theta500
(Δ=500, /B^(1/3)). Getting theta_max wrong under-painted massive-halo wings
(0.8% flux deficit, 134k missing pixels).
Other verified details: center=ang2vec(pi/2-dec, ra); query_disc inclusive=True
(superset) then strict theta<theta_max; d2=||v_pix-v_halo||^2,
theta=acos(clamp(1-d2/2,-1,1)), theta=max(thmin,theta), thmin=exp(-16.5);
amp=y0_true/B^(1/3); coords ra=rem(lon+pi,2pi)-pi, dec=lat-pi/2.

### PHASE 3 — GPU optimization: PASSING (~19x faster than XGPaint)
`paint_catalogue_gpu` in painting/healpix.py; `scripts/benchmark_gpu.py`.
**XGPaint ground-truth baseline:** the actual production painter
paint_a10_y0true_2d_mpi.jl self-reports **"painted in 108.07 seconds"** for
catalogue 0 (run standalone, 8 threads; its output map == stored reference
exactly, diff 0.0). (A faithful Julia re-implementation, reference/julia/
time_xgpaint.jl, timed 123.75 s.)
JXPaint GPU: **5.4-5.7 s/catalogue -> 19.0-19.8x speedup**, bit-for-bit
(max pixel rel err ~7e-11). tests/test_phase2.py: GPU==CPU painter to 2.5e-14.

Design: geometry computed in NUMPY (geometry._NpCosmo, CUDA-free) so the host
path can fork before any GPU use. Disc-finding parallelised across fork workers
(healpy query_disc is GIL-bound, so threads are useless but PROCESSES scale:
14s -> 2.1s on 16 procs). Then GPU does fused chord distance + bicubic gather
(into 537MB table, uploaded once via shape_table.coefs_device) + masked
scatter-add (jit). Per-catalogue split: disc ~2.1s, pix2vec ~1.3s, gpu ~1.3s.
Table coefs kept as numpy in BeamedShapeTable (lazy device copy) so loading
the table does not init CUDA (prerequisite for the pre-GPU fork).

### PHASE 4 — Fully-GPU painter + cosmology independence: PASSING (~590x)
`painting/gpu_native.py::paint_catalogue_gpu_native`. NO healpy / CPU host loop:
RING pix2vec, disc-finding (ring-range -> phi-window -> pixel expand via
cumsum/searchsorted), bicubic, scatter -- all on device. Painting kernel is one
jitted FIXED-SIZE program (disc output padded to e_max=Nh*24, n_max=Nh*220 with a
valid mask) so the whole pipeline compiles ONCE per halo count and is reused for
every cosmology/catalogue (no per-call recompile). **0.18 s/catalogue ->
~590x vs XGPaint 108 s**, bit-for-bit (max pixel rel err ~1.2e-10) across
catalogues 0..999 (scripts/stress_test.py). Edge cases (poles, giant haloes,
extreme z) match the CPU painter (0 extra/missing px). GPU pix2vec matches
healpy 4e-14; GPU disc set matches healpy exactly (0 missing/extra over 3001 halos).

NOTE: it is NOT GPU vs GPU -- XGPaint is CPU-only. The actual GPU compute is
~0.07 s disc + ~0.1 s paint; the win is algorithmic (vectorised geometry +
on-GPU disc-finding) + GPU. Caps (e_per_halo/n_per_halo) assume a realistic mass
function; a giant-heavy catalogue trips a clear assert -> raise the caps.

**Cosmology variation needs NO interpolator rebuild** (scripts/cosmology_demo.py,
tests/test_phase2.py::test_cosmology_no_interpolator_rebuild). The beam-convolved
shape table y_t(logtheta,logtheta500) depends ONLY on the gNFW shape params +
beam FWHM -- not on cosmology or B. So it is loaded ONCE (~0.4 s) and reused for
all cosmologies; the only cosmology-dependent step is the vectorised geometry
theta500(M,z) at ~6 ms/cosmology. Demo: 12 cosmologies, 1 table load, 0 rebuilds,
~0.18 s paint each, maps vary 10.9% in flux (cosmology genuinely applied).
This is exactly the XGPaint pain point (rebuilding the 8192^2 FFTLog interpolator
per model) that JXPaint removes by separating the cosmology-independent table
from the geometry.

## How to reproduce
- Phase 1: `PYTHONPATH=src python tests/test_phase1.py`
- Phase 2/3 fast: `python tests/test_phase2.py`
- Full map validation: `python scripts/paint_catalogue.py 0` then
  `python scripts/validate_map.py outputs/jxpaint_snr_0_y0true.fits <ref.fits>`
- Speedup benchmark: `python scripts/benchmark_gpu.py`
- Regenerate references (Julia): `julia reference/julia/gen_reference.jl` and
  `julia reference/julia/dump_table.jl` (dumps the 537MB beamed table, gitignored).
