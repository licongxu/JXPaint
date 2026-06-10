"""
Generate a HIGH-PRECISION machine-readable CSV of hmfast reference numbers
for ParametricGNFWPressureProfile.u_r.

Same cosmology, masses, redshifts, and extraction logic as
gen_hmfast_ref.py, but writes every numeric value with >=15 significant
digits to:
    /scratch/scratch-lxu/agent_dev/auto_research_agent/JXPaint/reference/hmfast/hmfast_ref_hp.csv

Header:
    M500c_Msun,z,Ez,r500c_Mpc,y0_param,y0_orig,ratio
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
# 1. Build cosmology + halo model (identical to gen_hmfast_ref.py)
# ----------------------------------------------------------------------
cosmo = Cosmology(
    emulator_set="lcdm:v1",
    H0=67.66,
    omega_b=0.02242,
    omega_cdm=0.1193,
)

mdef_500c = MassDefinition(delta=500, reference="critical")

hm = HaloModel(
    cosmology=cosmo,
    mass_definition=mdef_500c,
    concentration=D08Concentration(),
    convert_masses=True,
)

# ----------------------------------------------------------------------
# 2. Profile (defaults requested, B=1.41)
# ----------------------------------------------------------------------
prof = ParametricGNFWPressureProfile(
    A_SZ=-4.97, alpha_SZ=0.7867,
    P0=8.130, c500=1.156, alpha=1.0620, beta=5.4807, gamma=0.3292,
    B=1.41,
)

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

out_path = "/scratch/scratch-lxu/agent_dev/auto_research_agent/JXPaint/reference/hmfast/hmfast_ref_hp.csv"

rows = []
for M in masses:
    for z in redshifts:
        m = jnp.atleast_1d(jnp.asarray(M, dtype=jnp.float64))
        zz = jnp.atleast_1d(jnp.asarray(z, dtype=jnp.float64))

        c_old = hm.concentration.c_delta(hm, m, zz)
        m500c = convert_m_delta(cosmo, m, zz, mass_def_500c, mass_def_500c, c_old=c_old)

        r_500c = mass_def_500c.r_delta(cosmo, m500c, zz)
        r500c_val = float(jnp.squeeze(r_500c))

        H = jnp.atleast_1d(cosmo.hubble_parameter(zz))
        E_z = float(jnp.squeeze(H / H0))

        m500c_tilde = float(jnp.squeeze(m500c)) * h / B  # M500c h / B

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

        rows.append((M, z, E_z, r500c_val, y0_param, y0_orig, ratio))

# ----------------------------------------------------------------------
# 3. Write high-precision CSV (>=15 sig digits via .16e)
# ----------------------------------------------------------------------
header = "M500c_Msun,z,Ez,r500c_Mpc,y0_param,y0_orig,ratio"
with open(out_path, "w") as f:
    f.write(header + "\n")
    for (M, z, E_z, r500c_val, y0_param, y0_orig, ratio) in rows:
        f.write(
            f"{M:.16e},{z:.16e},{E_z:.16e},{r500c_val:.16e},"
            f"{y0_param:.16e},{y0_orig:.16e},{ratio:.16e}\n"
        )

print(f"Wrote {len(rows)} rows to {out_path}")
with open(out_path) as f:
    sys.stdout.write(f.read())
