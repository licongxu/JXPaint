"""Decisive amplitude test: Arnaud GNFW vs ParametricGNFW cl_1h vs painted-map data."""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# do NOT set JAX_PLATFORMS=cpu

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
from hmfast.halos.profiles.pressure import GNFWPressureProfile, ParametricGNFWPressureProfile
from hmfast.tracers.tsz import tSZTracer

REF = "/scratch/scratch-lxu/agent_dev/auto_research_agent/JXPaint/reference/hmfast/"
print("jax.devices():", jax.devices())

cosmo = Cosmology(emulator_set="lcdm:v1", H0=67.66, omega_b=0.02242, omega_cdm=0.1193)
mdef = MassDefinition(delta=500, reference="critical")
hm = HaloModel(cosmology=cosmo, mass_definition=mdef,
               concentration=D08Concentration(), convert_masses=True,
               hm_consistency=False)

m = jnp.asarray(np.logspace(14.0, 16.0, 80))
z = jnp.asarray(np.logspace(np.log10(0.005), np.log10(2.81), 160))
ell = np.unique(np.logspace(1, np.log10(2500), 60).astype(int))
l = jnp.asarray(ell.astype(float))

prof_A = GNFWPressureProfile(P0=8.130, c500=1.156, alpha=1.0620,
                             beta=5.4807, gamma=0.3292, B=1.41)
prof_B = ParametricGNFWPressureProfile(A_SZ=-4.97, alpha_SZ=0.7867,
                                       P0=8.130, c500=1.156, alpha=1.0620,
                                       beta=5.4807, gamma=0.3292, B=1.41)

cl1h_A = np.asarray(hm.cl_1h(tSZTracer(profile=prof_A), None, l, m, z))
cl2h_A = np.asarray(hm.cl_2h(tSZTracer(profile=prof_A), None, l, m, z))
cl1h_B = np.asarray(hm.cl_1h(tSZTracer(profile=prof_B), None, l, m, z))

# data target (unbeamed) -> log-interp onto our ell grid
d = np.load(REF + "data_deconv_target.npz")
ell_d = d["ell"].astype(float)
cld = d["cl_data_unbeamed"]
mask = (ell_d > 0) & (cld > 0)
def interp_data(L):
    return np.exp(np.interp(np.log(L), np.log(ell_d[mask]), np.log(cld[mask])))

probe = [100, 300, 600, 1000, 1500, 2000]
def cl_at(arr, L):
    return np.exp(np.interp(np.log(L), np.log(ell), np.log(arr)))

print("\n  ell      cl_A(Arnaud)   cl_B(Param)    cl_data        data/A     data/B")
rA, rB = {}, {}
for L in probe:
    a = cl_at(cl1h_A, L); b = cl_at(cl1h_B, L); dd = interp_data(L)
    rA[L] = dd / a; rB[L] = dd / b
    print("%6d  %.4e  %.4e  %.4e  %8.4f  %8.4f" % (L, a, b, dd, rA[L], rB[L]))

print("\nFlatness of data/A (normalised to ell=600):")
for L in probe:
    print("  ell=%5d  rel=%.3f" % (L, rA[L] / rA[600]))
print("Flatness of data/B (normalised to ell=600):")
for L in probe:
    print("  ell=%5d  rel=%.3f" % (L, rB[L] / rB[600]))

hi = [L for L in probe if L >= 300]
print("\nmean data/A (ell>=300) = %.4f  std/mean = %.3f"
      % (np.mean([rA[L] for L in hi]), np.std([rA[L] for L in hi])/np.mean([rA[L] for L in hi])))
print("mean data/B (ell>=300) = %.4f  std/mean = %.3f"
      % (np.mean([rB[L] for L in hi]), np.std([rB[L] for L in hi])/np.mean([rB[L] for L in hi])))
print("y0_orig/y0_param amplitude factor (A/B in C) at ell=600 = %.3f"
      % (cl_at(cl1h_A,600)/cl_at(cl1h_B,600)))

# save the correct (Arnaud) profile
np.savez(REF + "cl_yy_hmfast.npz",
         ell=ell, cl_1h=cl1h_A, cl_2h=cl2h_A, cl_tot=cl1h_A + cl2h_A, unbeamed=True)
print("\nSaved cl_yy_hmfast.npz (Arnaud GNFWPressureProfile, unbeamed)")
