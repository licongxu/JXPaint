"""Flat-sky rectangular patch painter.

This module keeps the same halo geometry and profile lookup used by the
full-sky HEALPix painter, but accumulates into a local tangent-plane image.
It is intended for catalogue cutouts such as 10 x 10 degree patches.
"""
import numpy as np

from . import geometry as geom


def _as_unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def patch_pixel_vectors(center_ra, center_dec, width_deg, height_deg,
                        pixel_size_arcmin=None, nx=None, ny=None):
    """Return tangent-plane pixel vectors and coordinate axes.

    Parameters are angular.  ``center_ra`` and ``center_dec`` are radians.
    The returned image axes are offsets from the patch centre in degrees:
    ``x_deg`` increases east/right and ``y_deg`` increases north/up.
    """
    width = np.deg2rad(width_deg)
    height = np.deg2rad(height_deg)
    if nx is None and ny is None:
        if pixel_size_arcmin is None:
            pixel_size_arcmin = 1.0
        pix_x = pix_y = np.deg2rad(pixel_size_arcmin / 60.0)
        nx = int(np.ceil(width / pix_x))
        ny = int(np.ceil(height / pix_y))
    else:
        if nx is None:
            nx = ny
        if ny is None:
            ny = nx
        nx = int(nx)
        ny = int(ny)
        pix_x = width / nx
        pix_y = height / ny
    if nx <= 0 or ny <= 0:
        raise ValueError("patch dimensions and pixel counts must be positive")

    x = (np.arange(nx, dtype=np.float64) + 0.5 - nx / 2.0) * pix_x
    y = (np.arange(ny, dtype=np.float64) + 0.5 - ny / 2.0) * pix_y
    xx, yy = np.meshgrid(x, y)

    c = np.array([
        np.cos(center_dec) * np.cos(center_ra),
        np.cos(center_dec) * np.sin(center_ra),
        np.sin(center_dec),
    ], dtype=np.float64)
    east = np.array([-np.sin(center_ra), np.cos(center_ra), 0.0],
                    dtype=np.float64)
    north = np.array([
        -np.sin(center_dec) * np.cos(center_ra),
        -np.sin(center_dec) * np.sin(center_ra),
        np.cos(center_dec),
    ], dtype=np.float64)
    vec = _as_unit(c + xx[..., None] * east + yy[..., None] * north)
    return vec, np.rad2deg(x), np.rad2deg(y)


def paint_catalogue_patch(z, M_1e14, lon, lat, y0_true, shape_table,
                          center_ra_deg, center_dec_deg, width_deg=10.0,
                          height_deg=10.0, pixel_size_arcmin=1.0,
                          nx=None, ny=None, cosmo=None, progress=False):
    """Paint a rectangular flat-sky Compton-y patch.

    Catalogue ``lon`` and ``lat`` follow the usual JXPaint/XGPaint convention
    in radians.  The patch centre is supplied as RA/Dec in degrees.  Returns
    ``(image, metadata)`` where ``image`` has shape ``(ny, nx)`` and metadata
    contains coordinate axes and patch settings.
    """
    center_ra = np.deg2rad(center_ra_deg)
    center_dec = np.deg2rad(center_dec_deg)
    pix_vec, x_deg, y_deg = patch_pixel_vectors(
        center_ra, center_dec, width_deg, height_deg,
        pixel_size_arcmin=pixel_size_arcmin, nx=nx, ny=ny)
    image = np.zeros(pix_vec.shape[:2], dtype=np.float64)

    ra, dec = geom.catalogue_to_radec(lon, lat)
    halo_vec = geom.radec_to_vec(ra, dec)
    th500 = geom.theta500_array(M_1e14, z, cosmo=cosmo)
    thmax = geom.theta_max_array(M_1e14, z, cosmo=cosmo)
    amp = geom.amplitude_array(y0_true)
    log_th500 = np.log(th500)
    thmin = np.exp(shape_table.logtheta_min)

    half_diag = np.deg2rad(0.5 * np.hypot(width_deg, height_deg))
    center_vec = geom.radec_to_vec(np.array([center_ra]), np.array([center_dec]))[0]
    sep_center = np.arccos(np.clip(halo_vec @ center_vec, -1.0, 1.0))
    candidates = np.nonzero(sep_center <= (half_diag + thmax))[0]

    pix_rad = max(np.diff(np.deg2rad(x_deg)).mean() if x_deg.size > 1 else np.deg2rad(width_deg),
                  np.diff(np.deg2rad(y_deg)).mean() if y_deg.size > 1 else np.deg2rad(height_deg))
    x_rad = np.deg2rad(x_deg)
    y_rad = np.deg2rad(y_deg)

    for n, i in enumerate(candidates, start=1):
        x0, y0 = _project_offset(center_ra, center_dec, ra[i], dec[i])
        r = thmax[i] + 2.0 * pix_rad
        ix = np.nonzero(np.abs(x_rad - x0) <= r)[0]
        iy = np.nonzero(np.abs(y_rad - y0) <= r)[0]
        if ix.size == 0 or iy.size == 0:
            continue

        sub = pix_vec[np.ix_(iy, ix)]
        d2 = np.sum((sub - halo_vec[i]) ** 2, axis=-1)
        theta = np.arccos(np.clip(1.0 - d2 / 2.0, -1.0, 1.0))
        theta = np.maximum(thmin, theta)
        keep = theta < thmax[i]
        if not keep.any():
            continue

        vals = amp[i] * np.asarray(
            shape_table.evaluate(np.log(theta[keep]),
                                 np.full(int(keep.sum()), log_th500[i])))
        patch = image[np.ix_(iy, ix)]
        patch[keep] += vals
        image[np.ix_(iy, ix)] = patch
        if progress and (n % 1000 == 0 or n == candidates.size):
            print(f"  patch painted candidate halos {n}/{candidates.size}",
                  flush=True)

    meta = {
        "center_ra_deg": float(center_ra_deg),
        "center_dec_deg": float(center_dec_deg),
        "width_deg": float(width_deg),
        "height_deg": float(height_deg),
        "pixel_size_arcmin": float(np.mean([width_deg * 60.0 / len(x_deg),
                                             height_deg * 60.0 / len(y_deg)])),
        "nx": int(len(x_deg)),
        "ny": int(len(y_deg)),
        "x_deg": x_deg,
        "y_deg": y_deg,
        "candidate_halos": int(candidates.size),
    }
    return image, meta


def _project_offset(center_ra, center_dec, ra, dec):
    """Small-angle east/north offset of a sky position from the patch centre."""
    dra = np.remainder(ra - center_ra + np.pi, 2.0 * np.pi) - np.pi
    x = dra * np.cos(center_dec)
    y = dec - center_dec
    return x, y


def write_patch(path, image, metadata):
    """Write a patch as NPZ, or as FITS when the path ends in .fits/.fit."""
    lower = path.lower()
    if lower.endswith((".fits", ".fit")):
        try:
            from astropy.io import fits
        except ImportError as exc:
            raise ImportError("writing FITS patches requires astropy") from exc
        hdr = fits.Header()
        hdr["CRVAL1"] = metadata["center_ra_deg"]
        hdr["CRVAL2"] = metadata["center_dec_deg"]
        hdr["CDELT1"] = -metadata["pixel_size_arcmin"] / 60.0
        hdr["CDELT2"] = metadata["pixel_size_arcmin"] / 60.0
        hdr["CTYPE1"] = "RA---TAN"
        hdr["CTYPE2"] = "DEC--TAN"
        hdr["BUNIT"] = "Compton-y"
        fits.PrimaryHDU(image, header=hdr).writeto(path, overwrite=True)
        return

    np.savez_compressed(
        path, y=image, x_deg=metadata["x_deg"], y_deg=metadata["y_deg"],
        center_ra_deg=metadata["center_ra_deg"],
        center_dec_deg=metadata["center_dec_deg"],
        width_deg=metadata["width_deg"], height_deg=metadata["height_deg"],
        pixel_size_arcmin=metadata["pixel_size_arcmin"],
        nx=metadata["nx"], ny=metadata["ny"],
        candidate_halos=metadata["candidate_halos"])
