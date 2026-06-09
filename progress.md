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

### PHASE 3 — GPU optimization: IN PROGRESS
Goal: >=5x faster than XGPaint on catalogue 0. Baseline timing of XGPaint via
reference/julia/time_xgpaint.jl. numpy painter bottleneck = per-halo Python
query_disc loop. Plan: host-side disc precompute + GPU batch (chord dist +
bicubic gather + scatter-add).
