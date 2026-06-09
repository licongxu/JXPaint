# Dump the production beam-convolved 2D shape table from the cache to raw binary,
# so JXPaint can load and bilinearly-interpolate the exact same y_t(logtheta, logtheta500).
using XGPaint
using JLD2
using Printf

const CACHE = "/scratch/scratch-lxu/tsz_cnc_scatter/paint_with_scatter/cached_a10_beamed2d_shape_ultrahighpres.jld2"
const OUT = "/scratch/scratch-lxu/jxpaint/reference/data"
mkpath(OUT)

obj = JLD2.load(CACHE, "y_shape_beamed2d")
itp2d = obj.itp2d
println("typeof(itp2d) = ", typeof(itp2d))

# Navigate FilledExtrapolation -> ScaledInterpolation -> BSplineInterpolation
ext = itp2d
scaled = getfield(ext, :itp)
println("typeof(scaled) = ", typeof(scaled))
ranges = scaled.ranges
logthetas = collect(ranges[1])
logtheta500s = collect(ranges[2])
A = parent(scaled)   # underlying BSpline interpolation
coefs = A.coefs
println("size(coefs) = ", size(coefs))
println("logtheta:  ", first(logthetas), " .. ", last(logthetas), "  N=", length(logthetas))
println("logtheta500: ", first(logtheta500s), " .. ", last(logtheta500s), "  N=", length(logtheta500s))
println("filled extrapolation value = ", getfield(ext, :fillvalue))

# sanity: compare a direct itp2d() eval to a manual bilinear on coefs at a sample
lt, l5 = -5.0, -6.0
println("itp2d(", lt, ",", l5, ") = ", itp2d(lt, l5))

println("coefs axes = ", axes(coefs))
P = parent(coefs)                 # plain Matrix, size (8194, 8194), 1-based
println("size(parent) = ", size(P), "  typeof = ", typeof(P))
# Write raw Float64 binary (Julia is column-major; numpy will read order='F')
open(joinpath(OUT, "beamed_table_coefs.f64"), "w") do io
    write(io, Array{Float64}(P))
end
open(joinpath(OUT, "beamed_table_axes.txt"), "w") do io
    println(io, "logtheta_min,logtheta_max,N_logtheta")
    @printf(io, "%.17g,%.17g,%d\n", first(logthetas), last(logthetas), length(logthetas))
    println(io, "logtheta500_min,logtheta500_max,N_logtheta500")
    @printf(io, "%.17g,%.17g,%d\n", first(logtheta500s), last(logtheta500s), length(logtheta500s))
    println(io, "fillvalue")
    @printf(io, "%.17g\n", Float64(getfield(ext, :fillvalue)))
    println(io, "parent_n0,parent_n1,order,interp")
    println(io, "$(size(P,1)),$(size(P,2)),column_major_F,cubic_line_ongrid_offset0")
end
println("DUMPED beamed_table_coefs.f64 (", size(coefs,1), "x", size(coefs,2), ") and axes.txt")
