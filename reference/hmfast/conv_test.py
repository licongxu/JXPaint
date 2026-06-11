"""
Convergence test for the hmfast tSZ C_ell^{yy} computation in gen_cl_yy.py.

Question: are Nm=80 (log) and Nz=80 (LINEAR) converged to ~1%? Linear z-spacing
may under-resolve the low-z clusters that dominate the tSZ 1-halo term.

Strategy: compute UNbeamed cl_1h, cl_2h on a fixed ell grid for several
(Nm, Nz, z-spacing) settings, and report fractional changes relative to a
high-resolution reference (Nm=320 log, Nz=320 log).

NOTE: this script does NOT modify hmfast source.
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

# Fixed multipole grid for the convergence test.
ELL = np.array([30, 100, 300, 600, 1000, 1500, 2000], dtype=float)

# Mass / redshift bounds (match gen_cl_yy.py).
M_LO, M_HI = 1e14, 1e16
Z_LO, Z_HI = 0.005, 2.81


def mass_grid(Nm):
    return np.logspace(np.log10(M_LO), np.log10(M_HI), Nm)


def z_grid(Nz, spacing):
    if spacing == "lin":
        return np.linspace(Z_LO, Z_HI, Nz)
    elif spacing == "log":
        # log-spaced in z
        return np.logspace(np.log10(Z_LO), np.log10(Z_HI), Nz)
    elif spacing == "log1p":
        # log-spaced in (1+z)
        return np.expm1(np.linspace(np.log1p(Z_LO), np.log1p(Z_HI), Nz))
    else:
        raise ValueError(spacing)


def build_model():
    cosmo = Cosmology(emulator_set="lcdm:v1", H0=67.66,
                      omega_b=0.02242, omega_cdm=0.1193)
    mdef_500c = MassDefinition(delta=500, reference="critical")
    hm = HaloModel(cosmology=cosmo,
                   mass_definition=mdef_500c,
                   concentration=D08Concentration(),
                   convert_masses=True,
                   hm_consistency=False)
    profile = ParametricGNFWPressureProfile(
        A_SZ=-4.97, alpha_SZ=0.7867,
        P0=8.130, c500=1.156, alpha=1.0620, beta=5.4807, gamma=0.3292,
        B=1.41,
    )
    tracer = tSZTracer(profile=profile)
    return hm, tracer


def compute(hm, tracer, Nm, Nz, spacing):
    m = jnp.asarray(mass_grid(Nm))
    z = jnp.asarray(z_grid(Nz, spacing))
    l = jnp.asarray(ELL)
    cl_1h = np.asarray(hm.cl_1h(tracer, None, l, m, z))
    cl_2h = np.asarray(hm.cl_2h(tracer, None, l, m, z))
    return cl_1h, cl_2h


def fmt_row(label, vals_1h, ref_1h, vals_2h, ref_2h):
    f1 = (vals_1h / ref_1h - 1.0) * 100.0
    f2 = (vals_2h / ref_2h - 1.0) * 100.0
    s1 = "  ".join("%+7.2f" % v for v in f1)
    s2 = "  ".join("%+7.2f" % v for v in f2)
    return "%-22s 1h: %s\n%-22s 2h: %s" % (label, s1, "", s2)


def main():
    hm, tracer = build_model()

    print("Convergence test for hmfast C_ell^yy (UNbeamed)")
    print("ell grid:", ELL.astype(int).tolist())
    print("M range: %.0e .. %.0e (log)   z range: %.3f .. %.3f"
          % (M_LO, M_HI, Z_LO, Z_HI))
    print()

    # --- Reference: highest resolution, log z ---
    print("Computing reference Nm=320 log, Nz=320 log ...")
    ref_1h, ref_2h = compute(hm, tracer, 320, 320, "log")
    print("  ref cl_1h:", " ".join("%.4e" % v for v in ref_1h))
    print("  ref cl_2h:", " ".join("%.4e" % v for v in ref_2h))
    print()

    # Also reference with linear z at high res (to test spacing bias).
    print("Computing reference Nm=320 log, Nz=320 LINEAR ...")
    refL_1h, refL_2h = compute(hm, tracer, 320, 320, "lin")
    print("  linear-ref cl_1h:", " ".join("%.4e" % v for v in refL_1h))
    print("  linear-ref cl_2h:", " ".join("%.4e" % v for v in refL_2h))
    print()

    hdr = "ell:                     " + "  ".join("%7d" % e for e in ELL.astype(int))
    print("Fractional change (%) vs reference (Nm=320 log, Nz=320 log):")
    print(hdr)
    print("-" * len(hdr))

    # --- Mass convergence (Nz fixed high = 256 log) ---
    print("\n[Mass convergence: Nz=256 log fixed]")
    for Nm in [40, 80, 160, 320]:
        c1, c2 = compute(hm, tracer, Nm, 256, "log")
        print(fmt_row("Nm=%-4d Nz=256log" % Nm, c1, ref_1h, c2, ref_2h))

    # --- Redshift convergence, LINEAR (Nm fixed high = 256 log) ---
    print("\n[Redshift convergence LINEAR: Nm=256 log fixed]")
    for Nz in [40, 80, 160, 320]:
        c1, c2 = compute(hm, tracer, 256, Nz, "lin")
        print(fmt_row("Nm=256 Nz=%-4dLIN" % Nz, c1, ref_1h, c2, ref_2h))

    # --- Redshift convergence, LOG z (Nm fixed high = 256 log) ---
    print("\n[Redshift convergence LOG z: Nm=256 log fixed]")
    for Nz in [40, 80, 160, 320]:
        c1, c2 = compute(hm, tracer, 256, Nz, "log")
        print(fmt_row("Nm=256 Nz=%-4dLOG" % Nz, c1, ref_1h, c2, ref_2h))

    # --- Redshift convergence, LOG (1+z) (Nm fixed high = 256 log) ---
    print("\n[Redshift convergence LOG(1+z): Nm=256 log fixed]")
    for Nz in [40, 80, 160, 320]:
        c1, c2 = compute(hm, tracer, 256, Nz, "log1p")
        print(fmt_row("Nm=256 Nz=%-4dLOG1p" % Nz, c1, ref_1h, c2, ref_2h))

    # --- The actual current setting: 80 log / 80 LINEAR ---
    print("\n[CURRENT setting: Nm=80 log, Nz=80 LINEAR]")
    c1, c2 = compute(hm, tracer, 80, 80, "lin")
    print(fmt_row("Nm=80 Nz=80 LINEAR", c1, ref_1h, c2, ref_2h))

    # --- Spacing-bias check: converged linear vs converged log ---
    print("\n[Spacing bias: high-res LINEAR vs high-res LOG (should be ~0 if converged)]")
    print(fmt_row("Nz=320LIN vs LOGref", refL_1h, ref_1h, refL_2h, ref_2h))

    # --- Low-z contribution diagnostic ---
    print("\n[Low-z (z<0.1) contribution to cl_1h, computed from high-res log z]")
    m = jnp.asarray(mass_grid(256))
    z_full = jnp.asarray(z_grid(512, "log"))
    l = jnp.asarray(ELL)
    cl_full = np.asarray(hm.cl_1h(tracer, None, l, m, z_full))
    # restrict to z<0.1
    zmask = np.asarray(z_full) < 0.1
    n_lowz = int(zmask.sum())
    print("  Of Nz=512 log nodes, %d lie at z<0.1 (z_min=%.4f)."
          % (n_lowz, float(z_full[0])))
    # how many low-z nodes does each grid get?
    for tag, Nz, sp in [("80 lin", 80, "lin"), ("80 log", 80, "log"),
                        ("160 lin", 160, "lin"), ("160 log", 160, "log")]:
        zz = z_grid(Nz, sp)
        print("    %-8s : %3d nodes at z<0.1, %3d at z<0.05"
              % (tag, int((zz < 0.1).sum()), int((zz < 0.05).sum())))


if __name__ == "__main__":
    main()
