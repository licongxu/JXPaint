"""
Generate the theoretical thermal-SZ angular auto power spectrum C_ell^{yy}
using the hmfast halo-model code, for comparison against a painted Compton-y map.

The halo-model machinery (Limber projection, 1-halo and 2-halo terms, the
sigma_T/(m_e c^2) Compton-y prefactor and the 1/(1+z) geometric kernel) is
already implemented in hmfast:

  - The pressure profile `u_k` returns the projected Fourier pressure profile
    u_ell, which already carries the 4*pi*(1+z)*r_delta/ell_delta^2 geometry
    and the Mpc->m conversion, but NOT the sigma_T/(m_e c^2) factor.
  - The Compton-y prefactor sigma_T/(m_e c^2) and the 1/(1+z) factor are
    supplied by tSZTracer.kernel.
  - HaloModel.cl_1h / cl_2h perform the full Limber projection:
        C_l^{1h} = int dz (dV/dzdOmega) W(z)^2 int dlnM (dn/dlnM) |u_ell|^2
        C_l^{2h} = int dz (dV/dzdOmega) W(z)^2 P_lin(k=(l+0.5)/chi, z)
                   [int dlnM (dn/dlnM) b(M,z) u_ell]^2

so we simply build the halo model, wrap the parametric GNFW pressure profile
in a tSZTracer, and call cl_1h / cl_2h.
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
sys.path.insert(0, "/scratch/scratch-lxu/agent_dev/auto_research_agent/hmfast/src")

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from hmfast.cosmology import Cosmology
from hmfast.halos.halo_model import HaloModel
from hmfast.halos.mass_definition import MassDefinition
from hmfast.halos.concentration import D08Concentration
from hmfast.halos.profiles.pressure import ParametricGNFWPressureProfile
from hmfast.tracers.tsz import tSZTracer

OUT = "/scratch/scratch-lxu/agent_dev/auto_research_agent/JXPaint/reference/hmfast/cl_yy_hmfast.npz"


def main():
    # --- Cosmology (FLAMINGO L2p8) ---
    cosmo = Cosmology(emulator_set="lcdm:v1", H0=67.66,
                      omega_b=0.02242, omega_cdm=0.1193)

    # Native mass definition: M500c (catalogue is M500c mass-limited).
    mdef_500c = MassDefinition(delta=500, reference="critical")
    hm = HaloModel(cosmology=cosmo,
                   mass_definition=mdef_500c,
                   concentration=D08Concentration(),
                   convert_masses=True)

    # --- tSZ pressure profile + tracer ---
    profile = ParametricGNFWPressureProfile(
        A_SZ=-4.97, alpha_SZ=0.7867,
        P0=8.130, c500=1.156, alpha=1.0620, beta=5.4807, gamma=0.3292,
        B=1.41,
    )
    tracer = tSZTracer(profile=profile)

    # --- Integration grids ---
    # Mass: M500c from 1e14 Msun upward (catalogue mass limit). Physical Msun.
    m = jnp.asarray(np.logspace(14.0, 15.8, 64))
    # Redshift: catalogue range.
    z = jnp.asarray(np.linspace(0.005, 2.81, 80))
    # Multipoles.
    ell = np.unique(np.logspace(1, np.log10(3000), 60).astype(int))
    l = jnp.asarray(ell.astype(float))

    print("Grids: Nl=%d, Nm=%d, Nz=%d" % (len(ell), m.shape[0], z.shape[0]))
    print("m range  [Msun] : %.3e .. %.3e" % (float(m[0]), float(m[-1])))
    print("z range          : %.3f .. %.3f" % (float(z[0]), float(z[-1])))
    print("ell range        : %d .. %d" % (ell[0], ell[-1]))

    # --- Compute C_ell^{yy} ---
    cl_1h = np.asarray(hm.cl_1h(tracer, None, l, m, z))
    cl_2h = np.asarray(hm.cl_2h(tracer, None, l, m, z))
    cl_tot = cl_1h + cl_2h

    # --- Save ---
    np.savez(OUT, ell=ell, cl_1h=cl_1h, cl_2h=cl_2h, cl_tot=cl_tot)
    print("\nSaved -> %s" % OUT)

    # --- Print a few values ---
    print("\n  ell      Cl_1h        Cl_2h        Cl_tot       l^2 Cl/2pi")
    for i in range(len(ell)):
        if i % 6 == 0 or i == len(ell) - 1:
            d = ell[i] ** 2 * cl_tot[i] / (2 * np.pi)
            print("%6d  %.4e  %.4e  %.4e  %.4e"
                  % (ell[i], cl_1h[i], cl_2h[i], cl_tot[i], d))

    # --- Sanity checks ---
    print("\nSanity checks:")
    finite = np.all(np.isfinite(cl_tot))
    positive = np.all(cl_1h > 0) and np.all(cl_2h > 0)
    print("  all finite           : %s" % finite)
    print("  cl_1h, cl_2h positive : %s" % positive)

    dl = ell ** 2 * cl_tot / (2 * np.pi)
    pk_ell = ell[int(np.argmax(dl))]
    print("  peak of l^2 Cl/2pi at ell = %d (tSZ expected ~2000-4000)" % pk_ell)
    print("  l^2 Cl/2pi at peak       = %.3e (expect ~1e-12)" % dl.max())

    # 1-halo should dominate at high ell, 2-halo at low ell for tSZ.
    lo = ell < 100
    hi = ell > 1000
    if lo.any():
        print("  2h/1h at ell<100  (median) = %.2f"
              % np.median((cl_2h[lo] / cl_1h[lo])))
    if hi.any():
        print("  2h/1h at ell>1000 (median) = %.3f"
              % np.median((cl_2h[hi] / cl_1h[hi])))

    ok = finite and positive and (1000 <= pk_ell <= 6000)
    print("\nOVERALL physical sanity: %s" % ("PASS" if ok else "CHECK"))


if __name__ == "__main__":
    main()
