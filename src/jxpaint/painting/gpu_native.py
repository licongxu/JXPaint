"""End-to-end GPU painting: RING pix2vec + disc-finding + paint, all in JAX.

Removes the CPU host bottleneck (healpy query_disc loop + pix2vec) from
paint_catalogue_gpu.  Ring geometry tables (per-ring start pixel, length,
z, phi0, dphi) are tiny (4*nside-1 entries) and built once on the host with
healpy (exact), then everything runs on the device.

Disc-finding reproduces the painted pixel SET as a superset of
centre-within-theta_max pixels (generous phi window + 1-ring margin); the exact
chord-distance theta < theta_max test then selects the same pixels XGPaint paints.
"""
from functools import partial

import numpy as np
import healpy as hp
import jax
import jax.numpy as jnp


def build_ring_tables(nside):
    """Per-ring (r=1..4nside-1) tables: start pixel, npix, z, phi0, dphi.

    phi0/z read from healpy for each ring's first pixel (exact, avoids parity
    subtleties).  Returns a dict of numpy arrays indexed by ring-1 (0-based).
    """
    nring = 4 * nside - 1
    r = np.arange(1, nring + 1)
    npix_r = np.where(r < nside, 4 * r,
             np.where(r <= 3 * nside, 4 * nside, 4 * (4 * nside - r)))
    ncap = 2 * nside * (nside - 1)
    start = np.empty(nring, dtype=np.int64)
    # north cap r=1..nside-1: start = 2 r (r-1)
    north = r < nside
    start[north] = 2 * r[north] * (r[north] - 1)
    # equatorial r=nside..3nside
    eq = (r >= nside) & (r <= 3 * nside)
    start[eq] = ncap + (r[eq] - nside) * 4 * nside
    # south cap r=3nside+1..4nside-1
    south = r > 3 * nside
    rs = 4 * nside - r[south]            # mirror ring index (1..nside-1)
    npix_tot = 12 * nside * nside
    start[south] = npix_tot - 2 * rs * (rs + 1)
    # z and phi0 from healpy for the first pixel of each ring
    theta0, phi0 = hp.pix2ang(nside, start, nest=False)
    z = np.cos(theta0)
    dphi = 2.0 * np.pi / npix_r
    return {
        "start": start.astype(np.int64),
        "npix": npix_r.astype(np.int64),
        "z": z.astype(np.float64),
        "phi0": phi0.astype(np.float64),
        "dphi": dphi.astype(np.float64),
        "nring": nring,
    }


def ring_pix2vec(ipix, tables_dev, nside):
    """RING pix2vec on device: ipix (int array) -> (..,3) unit vectors.

    Uses ring tables; finds each pixel's ring by searchsorted on start indices.
    """
    start = tables_dev["start"]
    npix = tables_dev["npix"]
    zt = tables_dev["z"]
    phi0 = tables_dev["phi0"]
    dphi = tables_dev["dphi"]
    # ring index = last r with start[r] <= ipix
    ri = jnp.searchsorted(start, ipix, side="right") - 1
    ri = jnp.clip(ri, 0, start.shape[0] - 1)
    j = ipix - start[ri]
    phi = phi0[ri] + j * dphi[ri]
    z = zt[ri]
    st = jnp.sqrt(jnp.maximum(0.0, 1.0 - z * z))
    return jnp.stack([st * jnp.cos(phi), st * jnp.sin(phi), z], axis=-1)


def disc_pixels(vec, thmax, tables_dev, nside, margin_pix=2.0, margin_ring=1):
    """Find disc pixels for all halos on the device.

    Returns (allpix, allhalo): flat int arrays of candidate pixel indices and
    the halo each came from.  A superset of centre-within-thmax pixels (generous
    phi window + ring margin); caller filters by exact chord theta < thmax.
    """
    start = tables_dev["start"]; npix_r = tables_dev["npix"]
    zt = tables_dev["z"]; phi0 = tables_dev["phi0"]; dphi = tables_dev["dphi"]
    nring = start.shape[0]
    neg_z = -zt                                   # increasing in ring index

    zc = vec[:, 2]
    phic = jnp.arctan2(vec[:, 1], vec[:, 0])
    thc = jnp.arccos(jnp.clip(zc, -1.0, 1.0))
    sinc = jnp.sqrt(jnp.maximum(0.0, 1.0 - zc * zc))
    cosr = jnp.cos(thmax)

    # ring range per halo (z decreasing; neg_z increasing)
    zhi = jnp.cos(jnp.maximum(0.0, thc - thmax))
    zlo = jnp.cos(jnp.minimum(jnp.pi, thc + thmax))
    rlo = jnp.searchsorted(neg_z, -zhi, side="left") - margin_ring
    rhi = jnp.searchsorted(neg_z, -zlo, side="right") - 1 + margin_ring
    rlo = jnp.clip(rlo, 0, nring - 1)
    rhi = jnp.clip(rhi, 0, nring - 1)
    n_rings = rhi - rlo + 1                         # (Nh,)

    # halos -> (halo, ring) entries
    off_h = jnp.concatenate([jnp.zeros(1, jnp.int64), jnp.cumsum(n_rings)])
    E = int(off_h[-1])
    eidx = jnp.arange(E, dtype=jnp.int64)
    h_of = jnp.searchsorted(off_h, eidx, side="right") - 1
    ring = rlo[h_of] + (eidx - off_h[h_of])         # ring index (0-based) per entry

    # per-entry phi window -> pixel count
    z_r = zt[ring]; sin_r = jnp.sqrt(jnp.maximum(0.0, 1.0 - z_r * z_r))
    denom = sinc[h_of] * sin_r
    cosd = (cosr[h_of] - zc[h_of] * z_r) / jnp.where(denom > 1e-12, denom, 1.0)
    whole = (denom <= 1e-12) | (cosd <= -1.0)
    dphi_win = jnp.where(whole, jnp.pi, jnp.arccos(jnp.clip(cosd, -1.0, 1.0)))
    dphw = dphi[ring]
    jhw = dphi_win / dphw + margin_pix              # half-width in pixels (+margin)
    jc = (phic[h_of] - phi0[ring]) / dphw
    jmin = jnp.ceil(jc - jhw).astype(jnp.int64)
    jmax = jnp.floor(jc + jhw).astype(jnp.int64)
    count = jnp.clip(jmax - jmin + 1, 0, npix_r[ring])   # (E,)

    # entries -> pixels
    off_e = jnp.concatenate([jnp.zeros(1, jnp.int64), jnp.cumsum(count)])
    N = int(off_e[-1])
    pidx = jnp.arange(N, dtype=jnp.int64)
    k = jnp.searchsorted(off_e, pidx, side="right") - 1
    local = pidx - off_e[k]
    j = jmin[k] + local
    pix = start[ring[k]] + jnp.mod(j, npix_r[ring[k]])
    halo = h_of[k]
    return pix, halo


def disc_pixels_fixed(vec, thmax, tables_dev, nside, e_max, n_max,
                      margin_pix=2.0, margin_ring=1):
    """Fixed-size disc-finding: same as disc_pixels but returns (pix, halo, valid)
    of fixed length n_max (padded), so the pipeline compiles ONCE and is reused
    for every cosmology / catalogue of the same halo count (no per-call recompile).

    Returns also (E_actual, N_actual) so the caller can assert the caps suffice.
    """
    start = tables_dev["start"]; npix_r = tables_dev["npix"]
    zt = tables_dev["z"]; phi0 = tables_dev["phi0"]; dphi = tables_dev["dphi"]
    nring = start.shape[0]; Nh = thmax.shape[0]
    neg_z = -zt

    zc = vec[:, 2]
    phic = jnp.arctan2(vec[:, 1], vec[:, 0])
    thc = jnp.arccos(jnp.clip(zc, -1.0, 1.0))
    sinc = jnp.sqrt(jnp.maximum(0.0, 1.0 - zc * zc))
    cosr = jnp.cos(thmax)

    zhi = jnp.cos(jnp.maximum(0.0, thc - thmax))
    zlo = jnp.cos(jnp.minimum(jnp.pi, thc + thmax))
    rlo = jnp.clip(jnp.searchsorted(neg_z, -zhi, side="left") - margin_ring, 0, nring - 1)
    rhi = jnp.clip(jnp.searchsorted(neg_z, -zlo, side="right") - 1 + margin_ring, 0, nring - 1)
    n_rings = rhi - rlo + 1

    off_h = jnp.concatenate([jnp.zeros(1, jnp.int64), jnp.cumsum(n_rings)])
    E_actual = off_h[-1]
    eidx = jnp.arange(e_max, dtype=jnp.int64)
    e_valid = eidx < E_actual
    h_of = jnp.clip(jnp.searchsorted(off_h, eidx, side="right") - 1, 0, Nh - 1)
    ring = jnp.clip(rlo[h_of] + (eidx - off_h[h_of]), 0, nring - 1)

    z_r = zt[ring]; sin_r = jnp.sqrt(jnp.maximum(0.0, 1.0 - z_r * z_r))
    denom = sinc[h_of] * sin_r
    cosd = (cosr[h_of] - zc[h_of] * z_r) / jnp.where(denom > 1e-12, denom, 1.0)
    whole = (denom <= 1e-12) | (cosd <= -1.0)
    dphi_win = jnp.where(whole, jnp.pi, jnp.arccos(jnp.clip(cosd, -1.0, 1.0)))
    dphw = dphi[ring]
    jhw = dphi_win / dphw + margin_pix
    jc = (phic[h_of] - phi0[ring]) / dphw
    jmin = jnp.ceil(jc - jhw).astype(jnp.int64)
    jmax = jnp.floor(jc + jhw).astype(jnp.int64)
    count = jnp.where(e_valid, jnp.clip(jmax - jmin + 1, 0, npix_r[ring]), 0)

    off_e = jnp.concatenate([jnp.zeros(1, jnp.int64), jnp.cumsum(count)])
    N_actual = off_e[-1]
    pidx = jnp.arange(n_max, dtype=jnp.int64)
    valid = pidx < N_actual
    k = jnp.clip(jnp.searchsorted(off_e, pidx, side="right") - 1, 0, e_max - 1)
    local = pidx - off_e[k]
    j = jmin[k] + local
    pix = start[ring[k]] + jnp.mod(j, npix_r[ring[k]])
    halo = jnp.where(valid, h_of[k], 0)
    pix = jnp.where(valid, pix, 0)
    return pix, halo, valid, E_actual, N_actual


# cached ring tables per nside (host build is instant; device copy reused)
_RING_CACHE = {}


def _ring_tables_device(nside):
    if nside not in _RING_CACHE:
        T = build_ring_tables(nside)
        _RING_CACHE[nside] = {
            k: (jnp.asarray(v) if isinstance(v, np.ndarray) else v)
            for k, v in T.items()}
    return _RING_CACHE[nside]


def compute_geometry(z, M_1e14, lon, lat, y0_true, cosmo, B):
    """All per-halo geometry on device: centre vectors, theta500, theta_max, amp.

    This is the ONLY cosmology-dependent step.  The beam-convolved shape table is
    cosmology-independent, so varying cosmology only re-runs this (vectorised,
    ~ms) -- no interpolator rebuild.
    """
    from .. import constants as C
    z = jnp.asarray(z, jnp.float64); M = jnp.asarray(M_1e14, jnp.float64)
    lon = jnp.asarray(lon, jnp.float64); lat = jnp.asarray(lat, jnp.float64)
    ra = jnp.mod(lon + jnp.pi, 2 * jnp.pi) - jnp.pi
    thc = jnp.pi / 2.0 - (lat - jnp.pi / 2.0)
    stc = jnp.sin(thc)
    vec = jnp.stack([stc * jnp.cos(ra), stc * jnp.sin(ra), jnp.cos(thc)], axis=-1)
    M_kg = M * 1e14 * C.M_SUN_KG
    R500 = cosmo.R_delta(M_kg, z, 500) / B ** (1.0 / 3.0)
    th500 = cosmo.angular_size(R500, z)
    R200 = cosmo.R_delta(M_kg, z, 200)
    theta1 = 2.0 * C.FWHM_ARCMIN * (jnp.pi / 10800.0)
    thmax = jnp.minimum(jnp.maximum(theta1, 4.0 * cosmo.angular_size(R200, z)),
                        jnp.deg2rad(C.THETA_MAX_DEG))
    amp = jnp.asarray(y0_true, jnp.float64) / B ** (1.0 / 3.0)
    return vec, th500, thmax, amp


def _cubic_w(f):
    f2 = f * f; f3 = f2 * f
    return jnp.stack([(1 - 3 * f + 3 * f2 - f3) / 6.0, (3 * f3 - 6 * f2 + 4) / 6.0,
                      (-3 * f3 + 3 * f2 + 3 * f + 1) / 6.0, f3 / 6.0], axis=-1)


@partial(jax.jit, static_argnames=("lt_min", "lt_max", "l5_min", "l5_max",
                                   "dlt", "dl5", "n0", "n1", "fill", "npix"))
def _paint_chunk(pix_c, valid_c, halo_c, vec, thmax, log5, amp, coefs,
                 start, npixr, zt, phi0, dphi,
                 lt_min, lt_max, l5_min, l5_max, dlt, dl5, n0, n1, fill, npix):
    """One fixed-size painted chunk -> map. Compiled once, reused everywhere."""
    # ring pix2vec (inline)
    ri = jnp.clip(jnp.searchsorted(start, pix_c, side="right") - 1, 0, start.shape[0] - 1)
    j = pix_c - start[ri]
    phi = phi0[ri] + j * dphi[ri]
    zp = zt[ri]
    stp = jnp.sqrt(jnp.maximum(0.0, 1.0 - zp * zp))
    vp = jnp.stack([stp * jnp.cos(phi), stp * jnp.sin(phi), zp], axis=-1)
    # chord distance
    vh = vec[halo_c]
    d2 = jnp.sum((vp - vh) ** 2, axis=1)
    thmin = jnp.exp(lt_min)
    theta = jnp.maximum(thmin, jnp.arccos(jnp.clip(1.0 - d2 / 2.0, -1.0, 1.0)))
    keep = valid_c & (theta < thmax[halo_c])
    # bicubic shape lookup
    lt = jnp.log(theta); l5 = log5[halo_c]
    p0 = 1.0 + (lt - lt_min) / dlt; p1 = 1.0 + (l5 - l5_min) / dl5
    inside = (lt >= lt_min) & (lt <= lt_max) & (l5 >= l5_min) & (l5 <= l5_max)
    p0c = jnp.clip(p0, 1.0, float(n0)); p1c = jnp.clip(p1, 1.0, float(n1))
    ix0 = jnp.clip(jnp.floor(p0c).astype(jnp.int32), 1, n0 - 1)
    ix1 = jnp.clip(jnp.floor(p1c).astype(jnp.int32), 1, n1 - 1)
    w0 = _cubic_w(p0c - ix0); w1 = _cubic_w(p1c - ix1)
    r0 = jnp.clip(ix0[:, None] - 1 + jnp.arange(4), 0, n0 + 1)
    r1 = jnp.clip(ix1[:, None] - 1 + jnp.arange(4), 0, n1 + 1)
    block = coefs[r0[:, :, None], r1[:, None, :]]
    yt = jnp.where(inside, jnp.einsum("ma,mab,mb->m", w0, block, w1), fill)
    contrib = jnp.where(keep, amp[halo_c] * yt, 0.0)
    return jnp.zeros(npix, dtype=jnp.float64).at[pix_c].add(contrib)


def paint_catalogue_gpu_native(z, M_1e14, lon, lat, y0_true, shape_table,
                               nside=1024, cosmo=None, chunk=None,
                               e_per_halo=24, n_per_halo=220):
    """Fully on-GPU painter: geometry + disc-finding + pix2vec + bicubic + scatter.

    No healpy / CPU host loop.  Fixed-size disc output + a jitted fixed-size paint
    kernel mean the whole pipeline compiles ONCE for a given halo count and is
    reused for every cosmology (and catalogue of that size) with no recompile and
    no interpolator rebuild.  Bit-for-bit equal to the reference painter.

    e_per_halo / n_per_halo set the (halo,ring)-entry and pixel padding caps as
    multiples of the halo count; raise them if the assert below trips.
    """
    from ..cosmology import default_cosmology
    from .. import constants as C
    if cosmo is None:
        cosmo = default_cosmology()
    Tdev = _ring_tables_device(nside)
    npix = 12 * nside * nside
    Nh = len(M_1e14)
    e_max = int(Nh * e_per_halo)
    n_max = int(Nh * n_per_halo)
    if chunk is None:
        chunk = n_max                                  # single fixed-size chunk
    else:
        n_max = ((n_max + chunk - 1) // chunk) * chunk  # multiple of chunk (fixed slices)

    vec, th500, thmax, amp = compute_geometry(z, M_1e14, lon, lat, y0_true,
                                              cosmo, C.B_BIAS)
    log5 = jnp.log(th500)
    pix, halo, valid, E_act, N_act = disc_pixels_fixed(
        vec, thmax, Tdev, nside, e_max, n_max)
    E_act = int(E_act); N_act = int(N_act)
    assert E_act <= e_max and N_act <= n_max, (
        f"disc caps too small: E={E_act}/{e_max}, N={N_act}/{n_max}; "
        "raise e_per_halo/n_per_halo")
    coefs = shape_table.coefs_device()
    st = shape_table
    out = jnp.zeros(npix, dtype=jnp.float64)
    for c0 in range(0, n_max, chunk):
        sl = slice(c0, c0 + chunk)
        out = out + _paint_chunk(
            pix[sl], valid[sl], halo[sl], vec, thmax, log5, amp, coefs,
            Tdev["start"], Tdev["npix"], Tdev["z"], Tdev["phi0"], Tdev["dphi"],
            lt_min=st.lt_min, lt_max=st.lt_max, l5_min=st.l5_min, l5_max=st.l5_max,
            dlt=st.dlt, dl5=st.dl5, n0=st.n0, n1=st.n1, fill=st.fill, npix=npix)
    out.block_until_ready()
    return np.asarray(out)
