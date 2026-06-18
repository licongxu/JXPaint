"""CustomGNFW pressure profile = hmfast ParametricGNFWPressureProfile,
with the XGPaint Arnaud10 gNFW shape and line-of-sight Compton-y.

Conventions (all verified against source):
  * gNFW 3D dimensionless shape (hmfast sign convention):
        P(s) = (c500 s)^{-gamma} (1 + (c500 s)^alpha)^{(gamma-beta)/alpha}
    with s = r / r500c.  Equivalent to XGPaint x_bar^g (1+x_bar^a)^((b-g)/a)
    using x_bar = s*c500, g=-gamma, b=-beta.
  * 1D projected shape:  F(x) = P0 * 2 \\int_0^\\infty P(sqrt(y^2 + x^2)) dy,
    x = theta / theta500.  Matches XGPaint shape_1d reference exactly.
  * Arnaud central y0 (hmfast y0_orig), r500c WITHOUT /B^{1/3}:
        y0_arnaud = 2 (sigma_T/m_e c^2) P0 P500c r500c_cm * shape_integral
  * Parametric y0 (hmfast):
        y0_param = 10^{A_SZ} (m500c h/B / (0.7*3e14))^{alpha_SZ} E(z)^2 (h/0.7)^{-1/2}
  * Painting geometry theta500 uses R500 = R_delta(M,z,500) / B^{1/3}.

M is M500c.  In the public API masses are passed as M_1e14 (units of 1e14 Msun).
"""
import numpy as np
import jax
import jax.numpy as jnp

from .. import constants as C
from ..cosmology import FlatLCDM, default_cosmology

# --- tanh-sinh (double-exponential) nodes for \int_0^\infty f(y) dy ---
# Substitution y = exp((pi/2) sinh(t)), dy = y * (pi/2) cosh(t) dt, t in [-T, T].
_TS_T = 4.0
_TS_N = 400
_ts_t = np.linspace(-_TS_T, _TS_T, _TS_N)
_ts_y = np.exp((np.pi / 2.0) * np.sinh(_ts_t))
_ts_w = (_ts_t[1] - _ts_t[0]) * _ts_y * (np.pi / 2.0) * np.cosh(_ts_t)
_TS_Y = jnp.asarray(_ts_y)            # (N,)
_TS_W = jnp.asarray(_ts_w)            # (N,)

_M_PIVOT = 0.7 * 3e14                 # 0.7 * 3e14 Msun (hmfast pivot)


class CustomGNFWPressureProfile:
    """JIT-friendly gNFW pressure / Compton-y profile (registered JAX PyTree)."""

    def __init__(self, A_SZ=C.A_SZ, alpha_SZ=C.ALPHA_SZ,
                 P0=C.P0_GNFW, c500=C.C500_GNFW, alpha=C.ALPHA_GNFW,
                 beta=C.BETA_GNFW, gamma=C.GAMMA_GNFW, B=C.B_BIAS,
                 cosmo: FlatLCDM = None):
        self.A_SZ = A_SZ
        self.alpha_SZ = alpha_SZ
        self.P0 = P0
        self.c500 = c500
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.B = B
        self.cosmo = default_cosmology() if cosmo is None else cosmo

    # ---- PyTree registration ----
    def _tree_flatten(self):
        children = (self.A_SZ, self.alpha_SZ, self.P0, self.c500,
                    self.alpha, self.beta, self.gamma, self.B, self.cosmo)
        return children, None

    @classmethod
    def _tree_unflatten(cls, aux, children):
        obj = cls.__new__(cls)
        (obj.A_SZ, obj.alpha_SZ, obj.P0, obj.c500, obj.alpha,
         obj.beta, obj.gamma, obj.B, obj.cosmo) = children
        return obj

    # ---- gNFW 3D dimensionless shape (no P0) ----
    def gnfw(self, s):
        sc = self.c500 * s
        return sc ** (-self.gamma) * (1.0 + sc ** self.alpha) ** (
            (self.gamma - self.beta) / self.alpha)

    # ---- 1D projected LOS shape  los(x) = 2 \int_0^\infty gnfw(sqrt(y^2+x^2)) dy ----
    def los_integral(self, x):
        """2 * \\int_0^\\infty gnfw(sqrt(y^2+x^2)) dy, vmappable over x."""
        x = jnp.asarray(x, dtype=jnp.float64)
        s = jnp.sqrt(_TS_Y[..., :] ** 2 + x[..., None] ** 2)   # (..., N)
        integ = self.gnfw(s)
        return 2.0 * jnp.sum(integ * _TS_W, axis=-1)

    def F(self, x):
        """Projected dimensionless shape F(x) = P0 * los_integral(x)."""
        return self.P0 * self.los_integral(x)

    def evaluate_shape(self, theta, theta500):
        """Unbeamed projected shape at angle theta given theta500: F(theta/theta500)."""
        return self.F(jnp.asarray(theta) / jnp.asarray(theta500))

    # ---- geometry ----
    def theta500(self, M_1e14, z):
        """Angular R500 [rad] with hydrostatic-bias radius R_delta/B^{1/3}."""
        M_kg = jnp.asarray(M_1e14) * 1e14 * C.M_SUN_KG
        R500 = self.cosmo.R_delta(M_kg, z, 500) / self.B ** (1.0 / 3.0)
        return self.cosmo.angular_size(R500, z)

    def r500c_Mpc(self, M_1e14, z, apply_bias=False):
        M_kg = jnp.asarray(M_1e14) * 1e14 * C.M_SUN_KG
        R = self.cosmo.R_delta(M_kg, z, 500)
        return R / self.B ** (1.0 / 3.0) if apply_bias else R

    # ---- normalizations ----
    def _P500c_eVcm3(self, M_1e14, z):
        """Arnaud P_500c [eV/cm^3] (hmfast / XGPaint A10_normalization)."""
        h = self.cosmo.h
        Ez = self.cosmo.E(z)
        m_tilde = jnp.asarray(M_1e14) * 1e14 * h / self.B
        return (1.65 * (h / 0.7) ** 2 * Ez ** (8.0 / 3.0)
                * (m_tilde / _M_PIVOT) ** (2.0 / 3.0 + 0.12)
                * (0.7 / h) ** 1.5)

    def y0_arnaud(self, M_1e14, z):
        """Central Arnaud Compton-y0 (hmfast y0_orig); r500c WITHOUT /B^{1/3}."""
        P500c = self._P500c_eVcm3(M_1e14, z)
        r500c_cm = self.r500c_Mpc(M_1e14, z, apply_bias=False) * C.MPC_CM
        return (2.0 * (C.SIGMA_T_CM2 / C.M_E_C2_EV)
                * self.P0 * P500c * r500c_cm * C.SHAPE_INTEGRAL)

    def y0_param(self, M_1e14, z):
        """Parametric central Compton-y0 (hmfast compute_y0_parametric)."""
        h = self.cosmo.h
        Ez = self.cosmo.E(z)
        m_tilde = jnp.asarray(M_1e14) * 1e14 * h / self.B
        return ((10.0 ** self.A_SZ)
                * (m_tilde / _M_PIVOT) ** self.alpha_SZ
                * Ez ** 2 * (h / 0.7) ** (-0.5))

    def ratio(self, M_1e14, z):
        return self.y0_param(M_1e14, z) / self.y0_arnaud(M_1e14, z)

    # ---- formula-only variants taking explicit cosmology inputs ----
    # (used to test arithmetic equivalence vs hmfast under hmfast's own E, r500c)
    def y0_param_from_E(self, M_1e14, Ez):
        h = self.cosmo.h
        m_tilde = jnp.asarray(M_1e14) * 1e14 * h / self.B
        return ((10.0 ** self.A_SZ) * (m_tilde / _M_PIVOT) ** self.alpha_SZ
                * jnp.asarray(Ez) ** 2 * (h / 0.7) ** (-0.5))

    def y0_arnaud_from_E_r(self, M_1e14, Ez, r500c_Mpc):
        h = self.cosmo.h
        m_tilde = jnp.asarray(M_1e14) * 1e14 * h / self.B
        P500c = (1.65 * (h / 0.7) ** 2 * jnp.asarray(Ez) ** (8.0 / 3.0)
                 * (m_tilde / _M_PIVOT) ** (2.0 / 3.0 + 0.12) * (0.7 / h) ** 1.5)
        r500c_cm = jnp.asarray(r500c_Mpc) * C.MPC_CM
        return (2.0 * (C.SIGMA_T_CM2 / C.M_E_C2_EV)
                * self.P0 * P500c * r500c_cm * C.SHAPE_INTEGRAL)

    # ---- full line-of-sight Compton-y (XGPaint compton_y; R500 uses /B^{1/3}) ----
    def y_los(self, theta, M_1e14, z):
        """Arnaud Compton-y at angle theta [rad] for halo (M_1e14, z).

        Matches XGPaint compton_y: C_norm * F(theta/theta500) * R500 * P_e_factor,
        with R500 = R_delta/B^{1/3}.  Returned dimensionless.
        """
        P500c = self._P500c_eVcm3(M_1e14, z)                 # eV/cm^3
        R500_cm = self.r500c_Mpc(M_1e14, z, apply_bias=True) * C.MPC_CM
        th500 = self.theta500(M_1e14, z)
        Fx = self.F(jnp.asarray(theta) / th500)
        return (C.SIGMA_T_CM2 / C.M_E_C2_EV) * P500c * R500_cm * Fx


class ArnaudGNFWPressureProfile(CustomGNFWPressureProfile):
    """Arnaud gNFW pressure profile with no hydrostatic mass bias (B=1).

    This is the simple Arnaud parameterization used by hmfast-style catalogue
    painting when the requested mass bias is unity.
    """

    def __init__(self, cosmo: FlatLCDM = None):
        super().__init__(B=1.0, cosmo=cosmo)


def arnaud_gnfw_b1_profile(cosmo: FlatLCDM = None):
    """Return the simple Arnaud gNFW profile with B=1."""
    return ArnaudGNFWPressureProfile(cosmo=cosmo)


jax.tree_util.register_pytree_node(
    CustomGNFWPressureProfile,
    lambda o: o._tree_flatten(),
    CustomGNFWPressureProfile._tree_unflatten,
)


# register FlatLCDM as a pytree so it can ride inside the profile under jit
def _cosmo_flatten(c):
    return (c.h, c.Omega_m, c.Omega_r, c.Omega_L), None


def _cosmo_unflatten(aux, children):
    h, Om, Or, OL = children
    return FlatLCDM(h=h, Omega_m=Om, Omega_r=Or, Omega_L=OL)


jax.tree_util.register_pytree_node(FlatLCDM, _cosmo_flatten, _cosmo_unflatten)
