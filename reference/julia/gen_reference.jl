# Generate gold-standard reference values from XGPaint for the JXPaint port.
# Run:  julia --project=... reference/julia/gen_reference.jl
# Outputs CSV files into reference/data/.
#
# Matches the benchmark painter paint_a10_y0true_2d_mpi.jl:
#   h=0.6766, Ob0h2=0.02242, Oc0h2=0.1193, B=1.41, FWHM=10 arcmin,
#   Arnauld10ThermalSZProfile(Omega_c, Omega_b, h, B), use_class_sz=false (analytic).

import Pkg
for pkg in ("Unitful", "UnitfulAstro")
    try
        @eval using $(Symbol(pkg))
    catch
        Pkg.add(pkg)
        @eval using $(Symbol(pkg))
    end
end
using XGPaint
using JLD2
using Printf
using Unitful
using UnitfulAstro

const OUT = "/scratch/scratch-lxu/agent_dev/auto_research_agent/JXPaint/reference/data"
mkpath(OUT)

# ---- painter cosmology defaults ----
const h      = 0.6766
const Ob0h2  = 0.02242
const Oc0h2  = 0.1193
const B      = 1.41
const Omega_b = Ob0h2 / h^2
const Omega_c = Oc0h2 / h^2
const FWHM_rad = 10.0 * (pi/180) / 60

model = XGPaint.Arnauld10ThermalSZProfile(Omega_c=Omega_c, Omega_b=Omega_b, h=h, B=B)
println("Model built. cosmo = ", model.cosmo)
println("Omega_m = ", model.cosmo.Ω_m, "  Omega_L = ", model.cosmo.Ω_Λ,
        "  Omega_r = ", model.cosmo.Ω_r, "  h = ", model.cosmo.h)

Msun = XGPaint.M_sun
Pefac = XGPaint.P_e_factor
println("M_sun = ", Msun, "  P_e_factor = ", Pefac)


# =====================================================================
# 1) Cosmology / geometry table over (M, z)
#    Columns: M[1e14 Msun], z, Ez, rho_crit[kg/m3], dA[Mpc], R500[Mpc], theta500[rad]
# =====================================================================
Ms = [0.5, 1.0, 1.4481393230306108, 2.0, 5.0, 10.0, 30.0]   # in 1e14 Msun
zs = [0.05, 0.1615681644444623, 0.3, 0.5, 0.7065646395805434, 1.0, 1.5, 2.0]

open(joinpath(OUT, "cosmo_geometry.csv"), "w") do io
    println(io, "M_1e14,z,Ez,rho_crit_kgm3,dA_Mpc,R500_Mpc,theta500_rad")
    for Mv in Ms, zv in zs
        Ez = XGPaint.Cosmology.E(model.cosmo, zv)
        rhoc = ustrip(uconvert(Unitful.u"kg/m^3", XGPaint.ρ_crit(model, zv)))
        dA = ustrip(uconvert(Unitful.u"Mpc", XGPaint.Cosmology.angular_diameter_dist(model.cosmo, zv)))
        R500u = XGPaint.R_Δ(model, Mv*1e14*Msun, zv, 500) / model.B^(1/3)
        R500 = ustrip(uconvert(Unitful.u"Mpc", R500u))
        th500 = XGPaint.angular_size(model, R500u, zv)
        @printf(io, "%.10g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                Mv, zv, Ez, rhoc, dA, R500, th500)
    end
end
println("wrote cosmo_geometry.csv")

# =====================================================================
# 2) 1D unbeamed shape F(x) = P0 * 2 * int_0^inf gnfw(sqrt(y^2+x^2)) dy
#    x = theta / theta500.  Columns: x, F
# =====================================================================
par = XGPaint.get_params(model, 1e14*Msun, 0.0)
println("get_params: P0=", par.P₀, " xc=", par.xc, " alpha=", par.α,
        " beta=", par.β, " gamma=", par.γ)
xs = exp10.(range(-6, 2, length=200))
open(joinpath(OUT, "shape_1d.csv"), "w") do io
    println(io, "x,F")
    for xv in xs
        F = par.P₀ * XGPaint._nfw_profile_los_quadrature(xv, par.xc, par.α, par.β, par.γ)
        @printf(io, "%.17g,%.17g\n", xv, F)
    end
end
println("wrote shape_1d.csv")

# shape integral check: F(0+) / P0 / 2 should equal 0.470502095
F0 = par.P₀ * XGPaint._nfw_profile_los_quadrature(1e-12, par.xc, par.α, par.β, par.γ)
open(joinpath(OUT, "scalars.csv"), "w") do io
    println(io, "name,value")
    @printf(io, "P0,%.17g\n", par.P₀)
    @printf(io, "xc,%.17g\n", par.xc)
    @printf(io, "alpha,%.17g\n", par.α)
    @printf(io, "beta,%.17g\n", par.β)
    @printf(io, "gamma,%.17g\n", par.γ)
    @printf(io, "F_at_zero,%.17g\n", F0)
    @printf(io, "shape_integral_from_F0,%.17g\n", F0/par.P₀/2)
    @printf(io, "M_sun_kg,%.17g\n", ustrip(Msun))
    @printf(io, "P_e_factor,%.17g\n", ustrip(Pefac))
    @printf(io, "Omega_m,%.17g\n", model.cosmo.Ω_m)
    @printf(io, "Omega_L,%.17g\n", model.cosmo.Ω_Λ)
    @printf(io, "Omega_r,%.17g\n", model.cosmo.Ω_r)
    @printf(io, "h,%.17g\n", model.cosmo.h)
    @printf(io, "FWHM_rad,%.17g\n", FWHM_rad)
end
println("wrote scalars.csv")

# =====================================================================
# 3) Central Compton-y (Arnaud) for cross-check: compton_y at small r
#    Columns: M_1e14, z, y0_arnaud
# =====================================================================
open(joinpath(OUT, "y0_arnaud.csv"), "w") do io
    println(io, "M_1e14,z,y0_arnaud_center")
    for Mv in Ms, zv in zs
        # compton_y(model, r, M_500, z): r is the angle theta (rad). Use tiny theta -> central.
        yc = XGPaint.compton_y(model, 1e-10, Mv*1e14*Msun, zv)
        @printf(io, "%.10g,%.17g,%.17g\n", Mv, zv, yc)
    end
end
println("wrote y0_arnaud.csv")

# =====================================================================
# 4) 2D beam-convolved shape table samples from the production cache
#    itp2d(logtheta, logtheta500).  Columns: logtheta, logtheta500, yt
# =====================================================================
const CACHE = "/scratch/scratch-lxu/tsz_cnc_scatter/paint_with_scatter/cached_a10_beamed2d_shape_ultrahighpres.jld2"
if isfile(CACHE)
    obj = JLD2.load(CACHE, "y_shape_beamed2d")
    itp2d = obj.itp2d
    # sample a grid within the table support: logtheta in [-16,2], logtheta500 in [-9.5,2.5]
    lts  = collect(range(-16.0, 2.0, length=120))
    lt5s = collect(range(-9.5, 2.5, length=40))
    open(joinpath(OUT, "shape2d_beamed.csv"), "w") do io
        println(io, "logtheta,logtheta500,yt")
        for lt in lts, lt5 in lt5s
            v = itp2d(lt, lt5)
            @printf(io, "%.17g,%.17g,%.17g\n", lt, lt5, v)
        end
    end
    println("wrote shape2d_beamed.csv")
else
    println("WARNING: cache not found, skipping shape2d_beamed.csv")
end

println("ALL REFERENCE FILES WRITTEN to ", OUT)
