"""Flat-LCDM background, replicating Cosmology.jl (as used by XGPaint, analytic path).

Reference (Cosmology.jl/src/Cosmology.jl):
    a2E(a)              = sqrt(Omega_r + Omega_m*a + Omega_L*a^4)
    E(z)                = a2E(a)/a^2,  a = 1/(1+z)
    hubble_dist0        = 2997.92458 / h   [Mpc]
    comoving_radial_dist(z) = hubble_dist0 * \\int_{a(z)}^{1} da / a2E(a)
    angular_diameter_dist(z) = comoving_radial_dist(z) / (1+z)   (flat)

The radial integral is done in scale factor `a` exactly as Cosmology.jl's
`Z(c,z) = quadgk(a -> 1/a2E(c,a), a(z), 1)`.  We use a high-order fixed
Gauss-Legendre rule; the integrand is smooth and monotonic so ~64 nodes give
>1e-12 accuracy, matching QuadGK's default rtol.
"""
import numpy as np
import jax
import jax.numpy as jnp

from . import constants as C

# Gauss-Legendre nodes/weights on [-1, 1] (host-side, then pushed to device).
_GL_N = 96
_gl_x, _gl_w = np.polynomial.legendre.leggauss(_GL_N)
_GL_X = jnp.asarray(_gl_x)
_GL_W = jnp.asarray(_gl_w)


class FlatLCDM:
    """Immutable flat-LCDM background matching Cosmology.jl FlatLCDM."""

    def __init__(self, h=C.H_LITTLE, Omega_m=None, Omega_r=None, Omega_L=None):
        self.h = h
        self.Omega_m = C.omega_m() if Omega_m is None else Omega_m
        self.Omega_r = C.omega_r() if Omega_r is None else Omega_r
        self.Omega_L = (1.0 - self.Omega_m - self.Omega_r) if Omega_L is None else Omega_L
        self.hubble_dist0_Mpc = 2997.92458 / self.h          # D_H0 [Mpc]
        # H0 in SI [1/s]: 100 h km/s/Mpc
        self.H0_si = 100.0 * self.h * 1000.0 / C.MPC_M

    # ---- dimensionless expansion ----
    def a2E(self, a):
        return jnp.sqrt(self.Omega_r + self.Omega_m * a + self.Omega_L * a**4)

    def E(self, z):
        a = 1.0 / (1.0 + z)
        return self.a2E(a) / a**2

    # ---- distances ----
    def _Z(self, z):
        """Z(z) = \\int_{a(z)}^{1} da / a2E(a) via Gauss-Legendre on [a(z), 1]."""
        z = jnp.asarray(z, dtype=jnp.float64)
        a_lo = 1.0 / (1.0 + z)
        a_hi = jnp.ones_like(a_lo)
        half = 0.5 * (a_hi - a_lo)              # (...,)
        mid = 0.5 * (a_hi + a_lo)
        # nodes: shape (..., N)
        a_nodes = mid[..., None] + half[..., None] * _GL_X
        integrand = 1.0 / self.a2E(a_nodes)
        return half * jnp.sum(integrand * _GL_W, axis=-1)

    def comoving_radial_dist(self, z):
        """Comoving radial distance D_C(z) [Mpc]."""
        return self.hubble_dist0_Mpc * self._Z(z)

    def angular_diameter_dist(self, z):
        """Angular-diameter distance d_A(z) [Mpc] (flat)."""
        return self.comoving_radial_dist(z) / (1.0 + z)

    # ---- densities / radii ----
    def rho_crit(self, z):
        """Critical density [kg/m^3]: 3 H(z)^2 / (8 pi G)."""
        Hz = self.H0_si * self.E(z)
        return 3.0 * Hz**2 / (8.0 * jnp.pi * C.G_SI)

    def R_delta(self, M_kg, z, delta=500):
        """R_delta [Mpc] from M_delta [kg]: (M / ((4pi/3) delta rho_crit))^(1/3)."""
        r_m = (M_kg / ((4.0 * jnp.pi / 3.0) * delta * self.rho_crit(z))) ** (1.0 / 3.0)
        return r_m / C.MPC_M

    def angular_size(self, R_Mpc, z):
        """Angular size [rad] = atan(R / d_A), both in Mpc (physical)."""
        return jnp.arctan2(R_Mpc, self.angular_diameter_dist(z))


def default_cosmology() -> FlatLCDM:
    return FlatLCDM()
