"""TruncatedParametricGNFWPressureProfile

Subclass of hmfast's ParametricGNFWPressureProfile that applies JXPaint's
per-halo theta_max profile truncation, WITHOUT modifying hmfast source.

    theta_max(M, z) = min( max(2*FWHM, 4*theta_R200), cap_deg )     (JXPaint geometry)

The truncation is applied at the u_k level as a multiplicative ratio

    u_k_trunc(k, M, z) = u_k_full(k, M, z) * T(k, M, z)

where T = u2d_trunc / u2d_full is computed from the 2D line-of-sight projected
Compton-y profile (cylinder cutoff at theta_max, matching the painter) via a
mcfit Hankel transform (nu=0).  u_k_full is the parent's exact hmfast u_k, so
the absolute normalization is unchanged; only the truncation shape is folded in.

T is precomputed on a coarse (M, z, q) grid (q = k * R500_c dimensionless) and
interpolated inside u_k.  Using the parent's u_k for the full profile means the
trispectrum (trispectrum_1h_masked) is also corrected when this profile is used.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import mcfit

from hmfast.halos.mass_definition import MassDefinition
from hmfast.halos.profiles import ParametricGNFWPressureProfile


class TruncatedParametricGNFWPressureProfile(ParametricGNFWPressureProfile):
    def __init__(self, *args, fwhm_arcmin: float = 10.0, cap_deg: float = 5.0,
                 n_mz: int = 40, nperp: int = 512, ns: int = 1024,
                 nr: int = 6144, **kwargs):
        super().__init__(*args, **kwargs)
        self.fwhm_arcmin = float(fwhm_arcmin)
        self.cap_deg = float(cap_deg)
        self.n_mz = int(n_mz)
        self.nperp = int(nperp)
        self.ns = int(ns)
        self.nr = int(nr)
        self._cache = None  # filled by precompute()

    # ---- precompute T(k, M, z) on a coarse grid ----
    def precompute(self, halo_model, m_min, m_max, z_min, z_max):
        cosmo = halo_model.cosmology
        Nm = Nz = self.n_mz
        mg = np.geomspace(m_min, m_max, Nm)
        zg = np.geomspace(z_min, z_max, Nz)
        mg_j = jnp.asarray(mg); zg_j = jnp.asarray(zg)

        mdef500 = MassDefinition(500, "critical")
        R500 = np.asarray(mdef500.r_delta(cosmo, mg_j, zg_j))          # (Nm,Nz) phys
        R500_c = R500 * (1.0 + zg[None, :])                            # comoving
        DA = np.asarray(cosmo.angular_diameter_distance(zg_j))[None, :]
        # vectorized theta_max (JXPaint: min(max(2FWHM, 4 theta_R200), cap); M500c->R200)
        mdef200 = MassDefinition(200, "critical")
        R200 = np.asarray(mdef200.r_delta(cosmo, mg_j, zg_j))          # (Nm,Nz)
        theta_R200 = np.arctan2(R200, DA)
        two_fwhm = 2.0 * np.deg2rad(self.fwhm_arcmin / 60.0)
        theta_max = np.minimum(np.maximum(two_fwhm, 4.0 * theta_R200),
                               np.deg2rad(self.cap_deg))               # (Nm,Nz)
        Rmax_c = (DA * theta_max) * (1.0 + zg[None, :])                # comoving cutoff
        rmax_ratio = Rmax_c / R500_c                                   # in R500 units

        # 3D P_e on a comoving log r grid
        r_grid = np.geomspace(1e-3, 1e4, self.nr)
        pe = np.asarray(super().u_r(halo_model, jnp.asarray(r_grid), mg_j, zg_j))  # (Nr,Nm,Nz)

        # 2D projected y2d(r_perp; M,z) via LOS integral in R500 units
        rp = np.logspace(np.log10(1e-3), np.log10(1e4), self.nperp)    # r_perp in R500 units
        # log-symmetric s grid (dense at the profile core)
        smax = 80.0
        s_half = np.logspace(np.log10(1e-3), np.log10(smax), self.ns // 2)
        s_grid = np.concatenate([-s_half[::-1], s_half])               # (Ns,) R500 units
        RP, SG = np.meshgrid(rp, s_grid, indexing="ij")
        G = np.sqrt(RP**2 + SG**2)                                     # (Nperp,Ns)
        R3all = R500_c[:, :, None, None] * G[None, None, :, :]         # (Nm,Nz,Nperp,Ns)
        y2d = np.zeros((Nm, Nz, self.nperp))
        for im in range(Nm):
            for iz in range(Nz):
                P3 = np.interp(R3all[im, iz], r_grid, pe[:, im, iz], left=0.0, right=0.0)
                y2d[im, iz] = np.trapezoid(P3, s_grid, axis=1) * R500_c[im, iz]
        y2d_full = y2d
        y2d_trunc = y2d * (rp[None, None, :] <= rmax_ratio[:, :, None])

        # mcfit Hankel (nu=0): F(q) = 2pi int y2d(rp) J0(q rp) rp drp
        han = mcfit.Hankel(rp, nu=0, lowring=True)
        q_native, F_full_flat = han(y2d_full.reshape(-1, self.nperp))
        _, F_trunc_flat = han(y2d_trunc.reshape(-1, self.nperp))
        F_full = F_full_flat.reshape(Nm, Nz, self.nperp)
        F_trunc = F_trunc_flat.reshape(Nm, Nz, self.nperp)

        self._cache = {
            "logm_grid": jnp.asarray(np.log(mg)),
            "z_grid": jnp.asarray(zg),
            "q_native": jnp.asarray(q_native),
            "R500_c_grid": jnp.asarray(R500_c),
            "T_cache": jnp.asarray(F_trunc / np.where(np.abs(F_full) > 0, F_full, 1.0)),
        }
        return self

    # ---- jax-traceable trilinear interp of T over (logm, z, q) ----
    def _interp_T(self, logm, z, q):
        lg = self._cache["logm_grid"]; zg = self._cache["z_grid"]; qg = self._cache["q_native"]
        Tc = self._cache["T_cache"]                                  # (Nm_c,Nz_c,Nq)

        def idx1d(xp, x):
            i = jnp.searchsorted(xp, x, side="right") - 1
            return jnp.clip(i, 0, xp.shape[0] - 2)

        il = idx1d(lg, logm)                                         # (Nm,)
        iz = idx1d(zg, z)                                            # (Nz,)
        iq = idx1d(qg, q)                                            # (Nk,Nm,Nz)
        wl = (logm - lg[il]) / (lg[il + 1] - lg[il])                # (Nm,)
        wz = (z - zg[iz]) / (zg[iz + 1] - zg[iz])                   # (Nz,)
        wq = (q - qg[iq]) / (qg[iq + 1] - qg[iq])                   # (Nk,Nm,Nz)

        il_b = il[None, :, None]; iz_b = iz[None, None, :]; iq_b = iq
        wl_b = wl[None, :, None]; wz_b = wz[None, None, :]
        c000 = Tc[il_b, iz_b, iq_b];       c001 = Tc[il_b, iz_b, iq_b + 1]
        c010 = Tc[il_b, iz_b + 1, iq_b];   c011 = Tc[il_b, iz_b + 1, iq_b + 1]
        c100 = Tc[il_b + 1, iz_b, iq_b];   c101 = Tc[il_b + 1, iz_b, iq_b + 1]
        c110 = Tc[il_b + 1, iz_b + 1, iq_b]; c111 = Tc[il_b + 1, iz_b + 1, iq_b + 1]
        T = (c000*(1-wl_b)*(1-wz_b)*(1-wq) + c001*(1-wl_b)*(1-wz_b)*wq +
             c010*(1-wl_b)*wz_b*(1-wq)     + c011*(1-wl_b)*wz_b*wq +
             c100*wl_b*(1-wz_b)*(1-wq)     + c101*wl_b*(1-wz_b)*wq +
             c110*wl_b*wz_b*(1-wq)         + c111*wl_b*wz_b*wq)
        T = jnp.where(q > qg[-1], 1.0, T)                           # high-q -> no truncation
        T_lo = Tc[il_b, iz_b, 0]
        T = jnp.where(q < qg[0], T_lo, T)                           # low-q -> ell->0 limit
        return T

    # ---- u_k with truncation folded in (jax-traceable) ----
    def u_k(self, halo_model, k, m, z):
        u_full = super().u_k(halo_model, k, m, z)                  # (Nk, Nm, Nz) exact
        if self._cache is None:
            return u_full
        k = jnp.atleast_1d(k); m = jnp.atleast_1d(m); z = jnp.atleast_1d(z)
        mdef500 = MassDefinition(500, "critical")
        R500_c = mdef500.r_delta(halo_model.cosmology, m, z) * (1.0 + z)[None, :]  # (Nm,Nz)
        q = k[:, None, None] * R500_c[None, :, :]                  # (Nk,Nm,Nz)
        T = self._interp_T(jnp.log(m), z, q)
        return u_full * T


# ---- pytree registration (parent u_r/u_k are @jax.jit; cl_1h_masked is jitted
# so the tracer/profile is flattened -> _cache arrays MUST be leaves, not aux) ----
def _trunc_flatten(obj):
    c = obj._cache
    if c is None:                                   # during precompute, before cache is set
        z1 = jnp.zeros((1,))
        logm_g = z_g = q_g = R500_g = Tc = z1
    else:
        logm_g, z_g, q_g, R500_g, Tc = (
            c["logm_grid"], c["z_grid"], c["q_native"], c["R500_c_grid"], c["T_cache"])
    leaves = (obj.A_SZ, obj.alpha_SZ, obj.P0, obj.c500, obj.alpha,
              obj.beta, obj.gamma, obj.B,
              logm_g, z_g, q_g, R500_g, Tc)
    aux = (tuple(obj._x.tolist()), obj._hankel, obj.fwhm_arcmin, obj.cap_deg,
           obj.n_mz, obj.nperp, obj.ns, obj.nr)
    return leaves, aux


def _trunc_unflatten(aux, leaves):
    x_tuple, hankel, fwhm, cap, nmz, nperp, ns, nr = aux
    (A_SZ, alpha_SZ, P0, c500, alpha, beta, gamma, B,
     logm_grid, z_grid, q_native, R500_c_grid, T_cache) = leaves
    obj = TruncatedParametricGNFWPressureProfile.__new__(TruncatedParametricGNFWPressureProfile)
    obj.A_SZ, obj.alpha_SZ, obj.P0, obj.c500, obj.alpha = A_SZ, alpha_SZ, P0, c500, alpha
    obj.beta, obj.gamma, obj.B = beta, gamma, B
    obj._x = np.array(x_tuple)
    obj._hankel = hankel
    obj.fwhm_arcmin = fwhm
    obj.cap_deg = cap
    obj.n_mz = nmz
    obj.nperp = nperp
    obj.ns = ns
    obj.nr = nr
    obj._cache = {
        "logm_grid": logm_grid, "z_grid": z_grid, "q_native": q_native,
        "R500_c_grid": R500_c_grid, "T_cache": T_cache,
    }
    return obj


jax.tree_util.register_pytree_node(
    TruncatedParametricGNFWPressureProfile, _trunc_flatten, _trunc_unflatten)
