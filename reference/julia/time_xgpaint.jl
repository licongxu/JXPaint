# Time XGPaint painting catalogue_bench_snr_0 (single process, no MPI),
# reusing the exact production painter logic, to set the speedup baseline.
using XGPaint, CSV, DataFrames, Healpix, Interpolations, JLD2

const CACHE = "/scratch/scratch-lxu/tsz_cnc_scatter/paint_with_scatter/cached_a10_beamed2d_shape_ultrahighpres.jld2"
const CAT = "/rds/rds-lxu/tsz_project/tsz_catalogue_benchmark/catalogue_bench_snr_0.csv"
const h=0.6766; const Ob0h2=0.02242; const Oc0h2=0.1193; const B=1.41
const Omega_b=Ob0h2/h^2; const Omega_c=Oc0h2/h^2

# Minimal reproduction of the painter's per-halo loop (matches paint_a10_y0true_2d_mpi.jl)
obj = JLD2.load(CACHE, "y_shape_beamed2d")   # BeamConvolvedA10_2D
itp2d = obj.itp2d
base  = obj.base
θmin  = exp(-16.5)   # logtheta grid lower edge (verified from the dumped table)
println("theta_min = ", θmin)

# Build a callable that paints with y0_true amplitude using the 2D shape.
# We mimic CachedBeamedA10_2DShape behaviour inline.
mutable struct Shape2D{I,Bs}; itp2d::I; base::Bs; lastM::Float64; lastz::Float64; l5::Float64; θmin::Float64; filled::Bool; end
XGPaint.compute_θmin(s::Shape2D) = s.θmin
XGPaint.R_Δ(s::Shape2D, args...; kwargs...) = XGPaint.R_Δ(s.base, args...; kwargs...)
XGPaint.angular_size(s::Shape2D, args...; kwargs...) = XGPaint.angular_size(s.base, args...; kwargs...)
function XGPaint.compute_θmax(s::Shape2D, M_Δ, z; FWHM=10, mult=4, kwargs...)
    r = XGPaint.R_Δ(s.base, M_Δ, z; kwargs...)
    theta1 = 2*FWHM*(π/10800)
    theta2 = mult*XGPaint.angular_size(s.base, r, z; kwargs...)
    return oftype(theta2, max(theta1, theta2))
end
@inline function (s::Shape2D)(θ::Real, M500::Real, z::Real; kwargs...)
    if !s.filled || M500 != s.lastM || z != s.lastz
        R500 = XGPaint.R_Δ(s.base, M500*XGPaint.M_sun, z, 500; kwargs...)/s.base.B^(1/3)
        θ500 = XGPaint.angular_size(s.base, R500, z; kwargs...)
        s.l5 = log(Float64(θ500)); s.lastM=M500; s.lastz=z; s.filled=true
    end
    θeff = max(s.θmin, θ)
    return s.itp2d(log(Float64(θeff)), s.l5) + 0
end

shape = Shape2D(itp2d, base, NaN, NaN, NaN, θmin, false)

df = CSV.read(CAT, DataFrame)
nside = 1024
θmax_ws = deg2rad(5.0)
out_map = HealpixMap{Float64,RingOrder}(nside)
ws = HealpixProfileWorkspace(nside, θmax_ws)

function paint!(out_map, ws, shape, df, θmax_ws, B)
    halo_mass = df.M .* 1e14
    ra = rem.(df.lon .+ π, 2π) .- π
    dec = df.lat .- π/2
    y0 = df.y0_true
    wser = XGPaint.wrapserialworkspace(ws, 1)
    fill!(out_map.pixels, 0.0)
    @inbounds for i in eachindex(halo_mass)
        Mh=halo_mass[i]; z=df.z[i]; α=ra[i]; δ=dec[i]
        amp = Float64(y0[i]); (!isfinite(amp) || amp==0.0) && continue
        amp /= B^(1/3)
        θmax = XGPaint.compute_θmax(shape, Mh*XGPaint.M_sun, z; FWHM=10)
        θmax = min(Float64(θmax), θmax_ws)
        XGPaint.profile_paint!(out_map, wser, shape, Mh, z, α, δ, θmax, amp)
    end
    return out_map
end

# warmup (compile) on a tiny slice
paint!(out_map, ws, shape, df[1:100,:], θmax_ws, B)
# timed full run
t0 = time(); paint!(out_map, ws, shape, df, θmax_ws, B); dt = time()-t0
println("XGPAINT_FULL_PAINT_SECONDS = ", round(dt, digits=3))
println("nthreads = ", Threads.nthreads(), "  nonzero=", count(!=(0.0), out_map.pixels),
        "  sum=", sum(out_map.pixels))
