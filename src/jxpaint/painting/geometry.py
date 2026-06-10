"""Painting geometry, matching paint_a10_y0true_2d_mpi.jl + XGPaint profiles.jl.

Coordinate convention (catalogue -> sky):
    ra  = rem(lon + pi, 2pi) - pi
    dec = lat - pi/2
Healpix center vector for a halo: ang2vec(theta_colat = pi/2 - dec, phi = ra).
Per-pixel angular distance (chord): theta = acos(clamp(1 - d^2/2, -1, 1)),
then theta = max(theta_min, theta); paint iff theta < theta_max.
theta_max = min( max(2*FWHM*pi/10800, 4*theta500), 5deg ).
Amplitude: A_halo = y0_true / B^(1/3).
"""
import numpy as np

from .. import constants as C

_TWO_PI = 2.0 * np.pi

# --- numpy mirror of cosmology.FlatLCDM (host-side, CUDA-free) ---
# Identical formulas to src/jxpaint/cosmology.py (validated vs XGPaint to ~7e-12),
# kept in numpy so the painting host path never initialises CUDA before the
# fork-based multiprocessing disc loop.
_GL_N = 96
_GL_X, _GL_W = np.polynomial.legendre.leggauss(_GL_N)


class _NpCosmo:
    def __init__(self, h=C.H_LITTLE, Omega_m=None, Omega_r=None, Omega_L=None):
        self.h = h
        self.Omega_m = C.omega_m() if Omega_m is None else Omega_m
        self.Omega_r = C.omega_r() if Omega_r is None else Omega_r
        self.Omega_L = (1.0 - self.Omega_m - self.Omega_r) if Omega_L is None else Omega_L
        self.hubble_dist0_Mpc = 2997.92458 / self.h
        self.H0_si = 100.0 * self.h * 1000.0 / C.MPC_M

    def a2E(self, a):
        return np.sqrt(self.Omega_r + self.Omega_m * a + self.Omega_L * a**4)

    def E(self, z):
        a = 1.0 / (1.0 + z)
        return self.a2E(a) / a**2

    def _Z(self, z):
        z = np.asarray(z, float)
        a_lo = 1.0 / (1.0 + z)
        half = 0.5 * (1.0 - a_lo)
        mid = 0.5 * (1.0 + a_lo)
        a_nodes = mid[..., None] + half[..., None] * _GL_X
        return half * np.sum((1.0 / self.a2E(a_nodes)) * _GL_W, axis=-1)

    def angular_diameter_dist(self, z):
        return self.hubble_dist0_Mpc * self._Z(z) / (1.0 + z)

    def rho_crit(self, z):
        Hz = self.H0_si * self.E(z)
        return 3.0 * Hz**2 / (8.0 * np.pi * C.G_SI)

    def R_delta(self, M_kg, z, delta=500):
        r_m = (M_kg / ((4.0 * np.pi / 3.0) * delta * self.rho_crit(z))) ** (1.0 / 3.0)
        return r_m / C.MPC_M

    def angular_size(self, R_Mpc, z):
        return np.arctan2(R_Mpc, self.angular_diameter_dist(z))


def default_cosmology():
    return _NpCosmo()


def catalogue_to_radec(lon, lat):
    """(lon, lat) [rad] -> (ra, dec) [rad]."""
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    ra = np.remainder(lon + np.pi, _TWO_PI) - np.pi
    dec = lat - np.pi / 2.0
    return ra, dec


def radec_to_vec(ra, dec):
    """Unit vector(s) for halo centres: ang2vec(pi/2 - dec, ra)."""
    theta = np.pi / 2.0 - np.asarray(dec, float)      # colatitude
    phi = np.asarray(ra, float)
    st = np.sin(theta)
    return np.stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)], axis=-1)


def theta500_array(M_1e14, z, cosmo=None, B=C.B_BIAS):
    """Angular R500 [rad] = angular_size(R_delta(M500c)/B^(1/3), z), vectorized."""
    cosmo = default_cosmology() if cosmo is None else cosmo
    M_kg = np.asarray(M_1e14, float) * 1e14 * C.M_SUN_KG
    R500 = np.asarray(cosmo.R_delta(M_kg, z, 500)) / B ** (1.0 / 3.0)   # Mpc
    return np.asarray(cosmo.angular_size(R500, z))


def theta_max_array(M_1e14, z, cosmo=None, fwhm_arcmin=C.FWHM_ARCMIN, mult=4,
                    cap_deg=C.THETA_MAX_DEG):
    """theta_max [rad] = min( max(2*FWHM*pi/10800, mult*angular_size(R_200, z)), cap ).

    Matches the painter's compute_θmax: r = R_Δ(base, M*M_sun, z) with the
    XGPaint default Δ=200 and NO /B^(1/3) bias (note: the M500c mass is fed into
    the 200c radius formula, exactly as the Julia painter does).  This is the
    disc cutoff only; the shape lookup still uses the Δ=500 biased theta500.
    """
    cosmo = default_cosmology() if cosmo is None else cosmo
    M_kg = np.asarray(M_1e14, float) * 1e14 * C.M_SUN_KG
    R200 = np.asarray(cosmo.R_delta(M_kg, z, 200))           # Mpc, unbiased
    theta1 = 2.0 * fwhm_arcmin * (np.pi / 10800.0)
    theta2 = mult * np.asarray(cosmo.angular_size(R200, z))
    tm = np.maximum(theta1, theta2)
    return np.minimum(tm, np.deg2rad(cap_deg))


def amplitude_array(y0_true, B=C.B_BIAS):
    """A_halo = y0_true / B^(1/3)."""
    return np.asarray(y0_true, float) / B ** (1.0 / 3.0)
