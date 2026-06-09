"""
Generate hmfast reference numbers for ParametricGNFWPressureProfile.u_r.

Builds a minimal hmfast HaloModel with a Planck-like cosmology
(h=0.6766, omega_b=0.02242, omega_cdm=0.1193) and a 500c mass
definition, so the input mass IS M500c (convert_m_delta is identity
because old and new mass definitions coincide).

We then re-derive, with the EXACT same formulas as
ParametricGNFWPressureProfile.u_r, the two internal central-y
quantities y0_param and y0_orig, plus r500c and E(z), and print them.
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, "/scratch/scratch-lxu/agent_dev/auto_research_agent/hmfast/src")

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from hmfast.cosmology import Cosmology
from hmfast.halos.halo_model import HaloModel
from hmfast.halos.mass_definition import MassDefinition, convert_m_delta
from hmfast.halos.concentration import D08Concentration
from hmfast.halos.profiles.pressure import ParametricGNFWPressureProfile
from hmfast.utils import Const

# ----------------------------------------------------------------------
# 1. Build cosmology + halo model
# ----------------------------------------------------------------------
# h = 0.6766 -> H0 = 67.66 ; omega_b h^2 = 0.02242 ; omega_c h^2 = 0.1193
cosmo = Cosmology(
    emulator_set="lcdm:v1",
    H0=67.66,
    omega_b=0.02242,
    omega_cdm=0.1193,
)

mdef_500c = MassDefinition(delta=500, reference="critical")

# convert_masses=True so D08Concentration.c_delta does not raise on the
# 500c key. Since the profile converts 500c -> 500c (identical mass def),
# the concentration value is irrelevant to the result (convert_m_delta is
# the identity in that case).
hm = HaloModel(
    cosmology=cosmo,
    mass_definition=mdef_500c,
    concentration=D08Concentration(),
    convert_masses=True,
)

# Sanity: report Omega_m total
cp = cosmo._cosmo_params()
print("# ===== Cosmology check =====")
print(f"h              = {float(cp['h']):.6f}")
print(f"Omega_b        = {float(cp['Omega_b']):.8f}")
print(f"Omega_cdm      = {float(cp['Omega_cdm']):.8f}")
print(f"Omega0_m (tot) = {float(cp['Omega0_m']):.14f}")
print(f"H0             = {float(cosmo.H0):.4f}")
print()

# ----------------------------------------------------------------------
# 2. Profile (defaults requested, B=1.41)
# ----------------------------------------------------------------------
prof = ParametricGNFWPressureProfile(
    A_SZ=-4.97, alpha_SZ=0.7867,
    P0=8.130, c500=1.156, alpha=1.0620, beta=5.4807, gamma=0.3292,
    B=1.41,
)

# Pull params
H0 = cosmo.H0
h = H0 / 100.0
A_SZ, alpha_SZ = prof.A_SZ, prof.alpha_SZ
P0, B = prof.P0, prof.B

mass_def_500c = MassDefinition(500, "critical")

# Constants exactly as in u_r
sigma_T_cm2 = 6.6524587e-25
m_e_c2_eV = 510998.95
shape_integral = 0.470502095
mpc_to_cm = Const._Mpc_over_m_ * 100.0

masses = [1.0e14, 5.0e14, 1.0e15]
redshifts = [0.16, 0.5, 1.0]

print("# ===== hmfast reference: ParametricGNFWPressureProfile internals =====")
print("# y0_orig = 2 (sigma_T/m_e c^2) P0 P500c_arnaud r500c_cm shape_integral")
print("# y0_param = 10^A_SZ (m500c h/B / (0.7*3e14))^alpha_SZ E^2 (h/0.7)^-0.5")
print("# r500c WITHOUT bias (i.e. mass_def_500c.r_delta of M500c), physical Mpc")
print()

header = (f"{'M500c[Msun]':>12} {'z':>5} {'E(z)':>12} {'r500c[Mpc]':>12} "
          f"{'y0_param':>14} {'y0_orig':>14} {'ratio':>12}")
print(header)
print("-" * len(header))

for M in masses:
    for z in redshifts:
        m = jnp.atleast_1d(jnp.asarray(M, dtype=jnp.float64))
        zz = jnp.atleast_1d(jnp.asarray(z, dtype=jnp.float64))

        # 500c -> 500c conversion (identity); still compute c_old as u_r does
        c_old = hm.concentration.c_delta(hm, m, zz)
        m500c = convert_m_delta(cosmo, m, zz, mass_def_500c, mass_def_500c, c_old=c_old)

        # r500c (physical Mpc), shape (Nm, Nz) -> scalar
        r_500c = mass_def_500c.r_delta(cosmo, m500c, zz)
        r500c_val = float(jnp.squeeze(r_500c))

        # E(z) = H(z)/H0
        H = jnp.atleast_1d(cosmo.hubble_parameter(zz))
        E_z = float(jnp.squeeze(H / H0))

        m500c_tilde = float(jnp.squeeze(m500c)) * h / B  # M500c h / B

        # Arnaud P500c
        P_500c_arnaud = (
            1.65 * (h / 0.7) ** 2 * E_z ** (8.0 / 3.0)
            * (m500c_tilde / (0.7 * 3e14)) ** (2.0 / 3.0 + 0.12)
            * (0.7 / h) ** 1.5
        )

        r_500c_cm = r500c_val * mpc_to_cm

        y0_orig = (
            2.0 * (sigma_T_cm2 / m_e_c2_eV)
            * P0 * P_500c_arnaud * r_500c_cm
            * shape_integral
        )

        y0_param = (
            (10.0 ** A_SZ)
            * (m500c_tilde / (0.7 * 3e14)) ** alpha_SZ
            * E_z ** 2
            * (h / 0.7) ** (-0.5)
        )

        ratio = y0_param / y0_orig

        print(f"{M:12.3e} {z:5.2f} {E_z:12.8f} {r500c_val:12.8f} "
              f"{y0_param:14.6e} {y0_orig:14.6e} {ratio:12.6f}")

print()
print("# ===== Cross-check vs XGPaint =====")
for z, e_xg in [(0.5, 1.2770800279), (1.0, 1.7800005933)]:
    zz = jnp.atleast_1d(jnp.asarray(z, dtype=jnp.float64))
    E_z = float(jnp.squeeze(cosmo.hubble_parameter(zz) / H0))
    print(f"z={z}: hmfast E(z)={E_z:.10f}  XGPaint E(z)={e_xg:.10f}  "
          f"rel diff={(E_z - e_xg)/e_xg:+.3e}")

# r500c WITHOUT bias for M500c=1e15 at z=0.5 (expect ~1.4 Mpc)
zz = jnp.atleast_1d(jnp.asarray(0.5, dtype=jnp.float64))
m = jnp.atleast_1d(jnp.asarray(1.0e15, dtype=jnp.float64))
r = float(jnp.squeeze(mass_def_500c.r_delta(cosmo, m, zz)))
print(f"r500c(M500c=1e15, z=0.5, no bias) = {r:.6f} Mpc  (XGPaint expects ~1.4 Mpc)")

print()
print("# ===== Consistency diagnostic for the XGPaint E(z) targets =====")
# Pure flat-LCDM, no radiation, with the requested Omega_m.
Om_req = 0.30957590896528514
print(f"# Requested Omega_m (= Omega_b + Omega_cdm, NO neutrinos) = {Om_req:.14f}")
print(f"# hmfast Omega0_m (INCLUDES m_ncdm=0.06 eV) = {float(cp['Omega0_m']):.14f}")
print("#")
print("# Analytic flat-LCDM E(z) = sqrt(Om(1+z)^3 + (1-Om)), no radiation:")
for z in (0.5, 1.0):
    E_an = (Om_req * (1 + z) ** 3 + (1 - Om_req)) ** 0.5
    print(f"#   z={z}: E(z)={E_an:.10f}")
print("#")
print("# Omega_m implied by each XGPaint E(z) (flat-LCDM, no radiation):")
for z, e_xg in [(0.5, 1.2770800279), (1.0, 1.7800005933)]:
    Om_imp = (e_xg ** 2 - 1.0) / ((1 + z) ** 3 - 1.0)
    print(f"#   z={z}: XGPaint E={e_xg:.10f} -> implied Omega_m={Om_imp:.6f}")
print("#")
print("# CONCLUSION: hmfast and XGPaint share the SAME Omega_m~0.3096 flat-LCDM.")
print("# The z=1.0 XGPaint E(z) matches hmfast (and the analytic value) to ~0.02%.")
print("# The z=0.5 XGPaint E(z)=1.27708 implies Omega_m~0.266 and is INTERNALLY")
print("# INCONSISTENT with its own z=1.0 value; it is an erroneous reference.")
