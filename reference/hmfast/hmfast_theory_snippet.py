import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"          # GPU 1 free; do NOT set JAX_PLATFORMS=cpu
import sys; sys.path.insert(0, "/scratch/scratch-lxu/agent_dev/auto_research_agent/hmfast/src")
import jax; jax.config.update("jax_enable_x64", True)
import numpy as np, jax.numpy as jnp
from hmfast.cosmology import Cosmology
from hmfast.halos.halo_model import HaloModel
from hmfast.halos.mass_definition import MassDefinition
from hmfast.halos.concentration import D08Concentration
from hmfast.halos.profiles.pressure import GNFWPressureProfile     # standard Arnaud -> y0_orig
from hmfast.tracers.tsz import tSZTracer

cosmo = Cosmology(emulator_set="lcdm:v1", H0=67.66, omega_b=0.02242, omega_cdm=0.1193)
hm = HaloModel(cosmology=cosmo, mass_definition=MassDefinition(delta=500, reference="critical"),
               concentration=D08Concentration(), convert_masses=True, hm_consistency=False)
tracer = tSZTracer(profile=GNFWPressureProfile(P0=8.130, c500=1.156, alpha=1.0620,
                                               beta=5.4807, gamma=0.3292, B=1.41))
m = jnp.asarray(np.logspace(14.0, 16.0, 80))                       # M500c [physical Msun]
z = jnp.asarray(np.logspace(np.log10(0.005), np.log10(2.81), 160)) # log-spaced
ell = np.unique(np.logspace(1, np.log10(2500), 60).astype(int))
cl_1h = np.asarray(hm.cl_1h(tracer, None, jnp.asarray(ell.astype(float)), m, z))  # UNbeamed C_l^yy
