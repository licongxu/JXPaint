"""
Diagnostics: why does the painted Compton-y map (from catalogue_bench_snr_0.csv)
have LOWER power than the hmfast C_ell^yy theory?

(1) Abundance check: Tinker-2008 predicted N(>1e14 Msun) full sky vs catalogue.
(2) Single-halo y_ell for (5e14,0.5) and (1e15,0.3).
(3) Amplitude cross-check: catalogue y0_true / amp_noscatter vs hmfast central y.
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
sys.path.insert(0, "/scratch/scratch-lxu/agent_dev/auto_research_agent/hmfast/src")

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import pandas as pd

from hmfast.cosmology import Cosmology
from hmfast.halos.halo_model import HaloModel
from hmfast.halos.mass_definition import MassDefinition
from hmfast.halos.concentration import D08Concentration
from hmfast.halos.profiles.pressure import ParametricGNFWPressureProfile
from hmfast.tracers.tsz import tSZTracer
from hmfast.utils import Const

CAT = "/rds/rds-lxu/tsz_project/tsz_catalogue_benchmark/catalogue_bench_snr_0.csv"
OUT_NPZ = "/scratch/scratch-lxu/agent_dev/auto_research_agent/JXPaint/reference/hmfast/y_ell_single.npz"

FOURPI = 4.0 * np.pi  # full sky steradians
ZMIN, ZMAX = 0.006, 2.81


def build():
    cosmo = Cosmology(emulator_set="lcdm:v1", H0=67.66,
                      omega_b=0.02242, omega_cdm=0.1193)
    mdef = MassDefinition(delta=500, reference="critical")
    hm = HaloModel(cosmology=cosmo, mass_definition=mdef,
                   concentration=D08Concentration(), convert_masses=True)
    profile = ParametricGNFWPressureProfile(
        A_SZ=-4.97, alpha_SZ=0.7867,
        P0=8.130, c500=1.156, alpha=1.0620, beta=5.4807, gamma=0.3292, B=1.41)
    tracer = tSZTracer(profile=profile)
    return cosmo, hm, profile, tracer


def chi_of_z(cosmo, z):
    return cosmo.angular_diameter_distance(z) * (1.0 + z)


# ---------------------------------------------------------------------------
# (1) ABUNDANCE
# ---------------------------------------------------------------------------
def abundance(cosmo, hm, df):
    print("\n" + "=" * 70)
    print("(1) ABUNDANCE CHECK  (Tinker-2008 vs catalogue)")
    print("=" * 70)

    # Fine grids for integration. M500c 1e14 .. ~10^15.6 (catalogue cap region).
    z = jnp.asarray(np.linspace(ZMIN, ZMAX, 120))
    m = jnp.asarray(np.logspace(14.0, 15.6, 80))
    logm = jnp.log(m)

    dndlnm = np.asarray(hm.halo_mass_function.halo_mass_function(hm, m, z))  # (Nm,Nz)
    dV = np.asarray(cosmo.comoving_volume_element(z))                        # Mpc^3/sr
    zn = np.asarray(z); mn = np.asarray(m); logmn = np.asarray(logm)

    # dN/dz over full sky = 4pi * dV/dzdOmega * int dlnM dn/dlnM
    n_per_Mpc3 = np.trapz(dndlnm, x=logmn, axis=0)         # (Nz,) number/Mpc^3
    dNdz_fullsky = FOURPI * dV * n_per_Mpc3                # (Nz,)
    N_pred = np.trapz(dNdz_fullsky, x=zn)
    print("Predicted N(M500c>1e14, full sky, z in [%.3f,%.2f]) = %.4e"
          % (ZMIN, ZMAX, N_pred))
    print("Catalogue actual N                                  = %d" % len(df))
    print("Ratio predicted/catalogue                            = %.3f"
          % (N_pred / len(df)))

    # dN/dz in z bins (predicted vs catalogue counts)
    print("\n  dN/dz comparison (counts in z-bins):")
    zbins = np.array([ZMIN, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, ZMAX])
    print("   z_lo   z_hi   N_pred     N_cat     pred/cat")
    zcat = df["z"].values
    for a, b in zip(zbins[:-1], zbins[1:]):
        sel = (zn >= a) & (zn <= b)
        if sel.sum() < 2:
            # finer local integration
            zz = np.linspace(a, b, 20)
            dndlnm_l = np.asarray(hm.halo_mass_function.halo_mass_function(
                hm, m, jnp.asarray(zz)))
            dV_l = np.asarray(cosmo.comoving_volume_element(jnp.asarray(zz)))
            npm = np.trapz(dndlnm_l, x=logmn, axis=0)
            Npred_bin = np.trapz(FOURPI * dV_l * npm, x=zz)
        else:
            Npred_bin = np.trapz(dNdz_fullsky[sel], x=zn[sel])
        Ncat_bin = int(((zcat >= a) & (zcat < b)).sum())
        print("  %.3f  %.3f  %.3e  %8d   %.3f"
              % (a, b, Npred_bin, Ncat_bin, Npred_bin / max(Ncat_bin, 1)))

    # dN/dlnM in mass bins (predicted full sky vs catalogue)
    print("\n  dN/dlnM comparison (counts in logM bins, M in 1e14 Msun units):")
    # catalogue M is in units of 1e14 Msun -> physical logM = log10(M*1e14)
    logM_cat = np.log10(df["M"].values * 1e14)
    mbins = np.array([14.0, 14.2, 14.4, 14.6, 14.8, 15.0, 15.4])
    print("   logMlo logMhi  N_pred     N_cat     pred/cat")
    for a, b in zip(mbins[:-1], mbins[1:]):
        mm = np.logspace(a, b, 30)
        dndlnm_b = np.asarray(hm.halo_mass_function.halo_mass_function(
            hm, jnp.asarray(mm), z))           # (30,Nz)
        # integrate over lnM then over z*dV*4pi
        npm_z = np.trapz(dndlnm_b, x=np.log(mm), axis=0)   # (Nz,)
        Npred_bin = np.trapz(FOURPI * dV * npm_z, x=zn)
        Ncat_bin = int(((logM_cat >= a) & (logM_cat < b)).sum())
        print("  %.2f  %.2f   %.3e  %8d   %.3f"
              % (a, b, Npred_bin, Ncat_bin, Npred_bin / max(Ncat_bin, 1)))


# ---------------------------------------------------------------------------
# (2) SINGLE-HALO y_ell
# ---------------------------------------------------------------------------
def single_halo_yell(cosmo, hm, profile, tracer):
    print("\n" + "=" * 70)
    print("(2) SINGLE-HALO y_ell")
    print("=" * 70)

    ell = np.unique(np.logspace(1, np.log10(3000), 40).astype(int))
    l = jnp.asarray(ell.astype(float))

    cases = [("5e14_z0p5", 5e14, 0.5), ("1e15_z0p3", 1e15, 0.3)]
    results = {}
    theta500 = {}

    for name, M, zc in cases:
        zc_arr = jnp.asarray([zc])
        m_arr = jnp.asarray([M])
        chi = float(chi_of_z(cosmo, zc_arr)[0])
        k = (np.asarray(ell) + 0.5) / chi            # Mpc^-1
        kern = float(tracer.kernel(cosmo, zc_arr)[0])
        # u_k -> (Nk, Nm, Nz)
        uk = np.asarray(profile.u_k(hm, jnp.asarray(k), m_arr, zc_arr))[:, 0, 0]
        y_ell = kern * uk
        results[name] = y_ell

        # theta500 with B^(1/3) dilution applied to r500c (as prompt requests)
        # mass-def here is 500c so r_delta(M500c)=r500c (physical Mpc)
        r500c = float(hm.mass_definition.r_delta(cosmo, m_arr, zc_arr)[0, 0])
        B = profile.B
        r500c_dil = r500c / B ** (1.0 / 3.0)
        dA = float(np.atleast_1d(cosmo.angular_diameter_distance(zc_arr))[0])
        # angular size of physical r500c: theta = r_phys / dA ; r_phys = r500c/(1+z)
        theta_rad = (r500c_dil / (1.0 + zc)) / dA
        theta500_arcmin = theta_rad * (180.0 / np.pi) * 60.0
        theta500[name] = theta500_arcmin

        # central value at small l, and half-max angular scale
        y0_small = y_ell[0]
        half = 0.5 * y_ell[0]
        idx = np.where(y_ell <= half)[0]
        if len(idx):
            lhalf = ell[idx[0]]
            theta_half_arcmin = (np.pi / lhalf) * (180.0 / np.pi) * 60.0  # ~180deg/l in arcmin
            theta_half_arcmin = (180.0 * 60.0) / lhalf
        else:
            lhalf, theta_half_arcmin = np.nan, np.nan

        print("\n  Halo %s  (M500c=%.2e Msun, z=%.2f):" % (name, M, zc))
        print("    chi(z) = %.2f Mpc, kernel = %.4e, r500c=%.4f Mpc, theta500(B-dil)=%.3f arcmin"
              % (chi, kern, r500c, theta500_arcmin))
        print("    y_ell(l_min=%d) = %.4e" % (ell[0], y0_small))
        print("    half-max at l ~ %s  (theta ~ %.2f arcmin)"
              % (str(lhalf), theta_half_arcmin))
        print("    %6s  %12s" % ("ell", "y_ell"))
        for i in range(len(ell)):
            if i % 5 == 0 or i == len(ell) - 1:
                print("    %6d  %.4e" % (ell[i], y_ell[i]))

    np.savez(OUT_NPZ,
             l=ell,
             y_ell_5e14_z0p5=results["5e14_z0p5"],
             y_ell_1e15_z0p3=results["1e15_z0p3"],
             theta500_5e14_z0p5_arcmin=theta500["5e14_z0p5"],
             theta500_1e15_z0p3_arcmin=theta500["1e15_z0p3"])
    print("\nSaved -> %s" % OUT_NPZ)
    return results, theta500


# ---------------------------------------------------------------------------
# (3) AMPLITUDE CROSS-CHECK
# ---------------------------------------------------------------------------
def central_y_hmfast(cosmo, hm, profile, M, zc):
    """Central Compton-y y0 = (sigma_T/m_e c^2) * int dl_phys P_e(r).

    Re-use the profile's own y0_param formula (the parametric amplitude IS the
    central y by construction of ParametricGNFWPressureProfile). Compute both
    y0_param (parametric central y) and y0_orig (Arnaud central y) directly
    from the analytic expressions in the profile.
    """
    from hmfast.halos.mass_definition import convert_m_delta
    m_arr = jnp.asarray([M]); z_arr = jnp.asarray([zc])
    h = cosmo.H0 / 100.0

    mdo = hm.mass_definition
    md500 = MassDefinition(500, "critical")
    c_old = hm.concentration.c_delta(hm, m_arr, z_arr)
    m500c = convert_m_delta(cosmo, m_arr, z_arr, mdo, md500, c_old=c_old)
    m500c = np.asarray(m500c).reshape(-1)[0]

    r500c = float(md500.r_delta(cosmo, jnp.asarray([m500c]), z_arr)[0, 0])
    H = float(np.atleast_1d(cosmo.hubble_parameter(z_arr))[0]); E_z = H / cosmo.H0
    B = profile.B
    m500c_tilde = m500c * h / B

    P_500c_arnaud = (1.65 * (h / 0.7) ** 2 * E_z ** (8.0 / 3.0)
                     * (m500c_tilde / (0.7 * 3e14)) ** (2.0 / 3.0 + 0.12)
                     * (0.7 / h) ** 1.5)

    sigma_T_cm2 = 6.6524587e-25
    m_e_c2_eV = 510998.95
    shape_integral = 0.470502095
    mpc_to_cm = Const._Mpc_over_m_ * 100.0
    r500c_cm = r500c * mpc_to_cm

    y0_orig = (2.0 * (sigma_T_cm2 / m_e_c2_eV) * profile.P0
               * P_500c_arnaud * r500c_cm * shape_integral)
    y0_param = ((10.0 ** profile.A_SZ)
                * (m500c_tilde / (0.7 * 3e14)) ** profile.alpha_SZ
                * E_z ** 2 * (h / 0.7) ** (-0.5))
    return y0_orig, y0_param, m500c


def amplitude_check(cosmo, hm, profile, df):
    print("\n" + "=" * 70)
    print("(3) AMPLITUDE CROSS-CHECK (catalogue y0_true / amp_noscatter vs hmfast)")
    print("=" * 70)

    Mphys = df["M"].values * 1e14
    z = df["z"].values
    y0t = df["y0_true"].values
    ampns = df["amp_noscatter"].values

    cases = [(5e14, 0.5), (1e15, 0.3)]
    for M, zc in cases:
        # select catalogue halos near (M,z): within +/-0.15 dex in M, +/-0.05 in z
        sel = (np.abs(np.log10(Mphys) - np.log10(M)) < 0.10) & (np.abs(z - zc) < 0.06)
        n = sel.sum()
        y0_orig, y0_param, m500c = central_y_hmfast(cosmo, hm, profile, M, zc)
        print("\n  Near (M=%.2e, z=%.2f): %d catalogue halos (M500c_conv=%.3e)" % (M, zc, n, m500c))
        if n > 0:
            print("    catalogue mean y0_true       = %.4e (std %.2e)" % (y0t[sel].mean(), y0t[sel].std()))
            print("    catalogue mean amp_noscatter = %.4e (std %.2e)" % (ampns[sel].mean(), ampns[sel].std()))
        print("    hmfast y0_param (parametric central y) = %.4e" % y0_param)
        print("    hmfast y0_orig  (Arnaud central y)     = %.4e" % y0_orig)
        if n > 0:
            print("    ratio  cat_amp_noscatter / y0_param   = %.3f" % (ampns[sel].mean() / y0_param))
            print("    ratio  cat_amp_noscatter / y0_orig    = %.3f" % (ampns[sel].mean() / y0_orig))
            print("    ratio  cat_y0_true       / y0_param   = %.3f" % (y0t[sel].mean() / y0_param))


def main():
    print("Loading catalogue ...")
    df = pd.read_csv(CAT)
    print("Catalogue: %d rows, cols=%s" % (len(df), list(df.columns)))
    print("  z range  : %.4f .. %.4f" % (df["z"].min(), df["z"].max()))
    print("  M(1e14) range: %.4f .. %.4f  (logM_phys %.3f .. %.3f)"
          % (df["M"].min(), df["M"].max(),
             np.log10(df["M"].min() * 1e14), np.log10(df["M"].max() * 1e14)))

    cosmo, hm, profile, tracer = build()
    abundance(cosmo, hm, df)
    single_halo_yell(cosmo, hm, profile, tracer)
    amplitude_check(cosmo, hm, profile, df)


if __name__ == "__main__":
    main()
