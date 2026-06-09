"""Physical constants, matched bit-for-bit to XGPaint / Cosmology.jl / hmfast.

All SI unless noted.  Enables float64 in JAX on import (physics requirement).
"""
import jax

jax.config.update("jax_enable_x64", True)

# --- fundamental / astro constants (XGPaint XGPaint.jl + Unitful/UnitfulAstro) ---
M_SUN_KG = 1.98847e30                  # XGPaint.M_sun
MPC_M = 3.0856775814913673e22          # Unitful u"Mpc" in metres
MPC_CM = MPC_M * 100.0
G_SI = 6.67430e-11                     # CODATA2018 gravitational constant [m^3 kg^-1 s^-2]
C_KM_S = 299792.458                    # speed of light [km/s]; hubble_dist0 = 2997.92458/h Mpc

# Compton-y conversion (XGPaint: P_e_factor = sigma_T / (m_e c^2), SI [s^2/kg])
P_E_FACTOR_SI = 8.125531675591423e-16

# hmfast parametric-y0 constants (tszpower convention)
SIGMA_T_CM2 = 6.6524587e-25            # Thomson cross-section [cm^2]
M_E_C2_EV = 510998.95                  # electron rest energy [eV]
SHAPE_INTEGRAL = 0.470502095           # GNFW dimensionless shape integral (hardcoded in tszpower)

# --- background cosmology (paint_a10_y0true_2d_mpi.jl defaults) ---
H_LITTLE = 0.6766
OB0H2 = 0.02242
OC0H2 = 0.1193
TCMB = 2.7255
NEFF = 3.046
B_BIAS = 1.41                          # hydrostatic mass-bias factor

# gNFW Arnaud10 shape parameters (XGPaint get_params; hmfast equivalents in parens)
P0_GNFW = 8.130
C500_GNFW = 1.156                      # XGPaint xc = 1/c500
ALPHA_GNFW = 1.0620
BETA_GNFW = 5.4807                     # hmfast sign (XGPaint uses -beta)
GAMMA_GNFW = 0.3292                    # hmfast sign (XGPaint uses -gamma)

# parametric amplitude defaults (hmfast ParametricGNFWPressureProfile)
A_SZ = -4.97
ALPHA_SZ = 0.7867

# beam / painting
FWHM_ARCMIN = 10.0
FWHM_RAD = FWHM_ARCMIN * (3.141592653589793 / 180.0) / 60.0
NSIDE = 1024
THETA_MAX_DEG = 5.0


def omega_m() -> float:
    return (OB0H2 + OC0H2) / H_LITTLE**2


def omega_r() -> float:
    omega_g = 4.48131e-7 * TCMB**4 / H_LITTLE**2
    omega_n = NEFF * omega_g * (7.0 / 8.0) * (4.0 / 11.0) ** (4.0 / 3.0)
    return omega_g + omega_n


def omega_lambda() -> float:
    return 1.0 - omega_m() - omega_r()
