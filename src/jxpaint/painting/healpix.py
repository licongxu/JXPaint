"""Healpix tSZ painter, bit-for-bit matching paint_a10_y0true_2d_mpi.jl.

For each halo: find disc pixels (healpy query_disc, RING), compute the
chord-based angular distance to each pixel centre, clamp to theta_min, and for
pixels with theta < theta_max accumulate A_halo * y_t(log theta, log theta500)
into the map, where y_t is the bicubic beam-convolved shape table.

query_disc is called inclusive=True (a superset of centre-within-theta_max
pixels); the strict theta < theta_max test then reproduces XGPaint's
centre-within-radius painted set exactly.
"""
import numpy as np
import healpy as hp

from .. import constants as C
from . import geometry as geom


def paint_catalogue(z, M_1e14, lon, lat, y0_true, shape_table,
                    nside=C.NSIDE, cosmo=None, chunk=4000, progress=False):
    """Paint a Compton-y map. Returns a RING-ordered float64 map (npix,)."""
    npix = hp.nside2npix(nside)
    out = np.zeros(npix, dtype=np.float64)

    ra, dec = geom.catalogue_to_radec(lon, lat)
    vec = geom.radec_to_vec(ra, dec)                      # (Nh, 3)
    th500 = geom.theta500_array(M_1e14, z, cosmo=cosmo)   # (Nh,)
    thmax = geom.theta_max_array(M_1e14, z, cosmo=cosmo)  # (Nh,)
    amp = geom.amplitude_array(y0_true)                   # (Nh,)
    log_th500 = np.log(th500)
    thmin = np.exp(shape_table.logtheta_min)              # exp(-16.5)

    Nh = len(M_1e14)
    for c0 in range(0, Nh, chunk):
        c1 = min(c0 + chunk, Nh)
        pix_list, logth_list, log5_list, amp_list = [], [], [], []
        for i in range(c0, c1):
            tmax = thmax[i]
            disc = hp.query_disc(nside, vec[i], tmax, inclusive=True, nest=False)
            if disc.size == 0:
                continue
            vp = np.asarray(hp.pix2vec(nside, disc, nest=False)).T   # (npd,3)
            d2 = np.sum((vp - vec[i]) ** 2, axis=1)
            theta = np.arccos(np.clip(1.0 - d2 / 2.0, -1.0, 1.0))
            theta = np.maximum(thmin, theta)
            keep = theta < tmax
            if not keep.any():
                continue
            pix_list.append(disc[keep])
            logth_list.append(np.log(theta[keep]))
            log5_list.append(np.full(int(keep.sum()), log_th500[i]))
            amp_list.append(np.full(int(keep.sum()), amp[i]))
        if not pix_list:
            continue
        pix = np.concatenate(pix_list)
        logth = np.concatenate(logth_list)
        log5 = np.concatenate(log5_list)
        amps = np.concatenate(amp_list)
        yt = np.asarray(shape_table.evaluate(logth, log5))
        np.add.at(out, pix, amps * yt)
        if progress:
            print(f"  painted halos {c1}/{Nh}", flush=True)
    return out
