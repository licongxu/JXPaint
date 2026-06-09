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
from ..cosmology import default_cosmology

_TWO_PI = 2.0 * np.pi


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
