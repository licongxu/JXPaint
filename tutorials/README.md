# JXPaint tutorials

Jupyter notebooks introducing JXPaint. Run them from this `tutorials/` directory
with the project venv:

```bash
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
cd tutorials && jupyter lab        # or: jupyter notebook
```

They ship with executed outputs (figures + numbers), so you can read them without
running anything.

| Notebook | What it covers |
|----------|----------------|
| [`01_quickstart.ipynb`](01_quickstart.ipynb) | Load a catalogue, paint a Compton-$y$ map on the GPU (~0.2 s), validate bit-for-bit vs the XGPaint reference, visualise. |
| [`02_profiles.ipynb`](02_profiles.ipynb) | The CustomGNFW profile: 3D shape, projected $F(x)$, $\theta_{500}$ and the $y_0$ amplitudes vs $(M,z)$, and the beam-convolved 2D shape table. |
| [`03_speed_and_cosmology.ipynb`](03_speed_and_cosmology.ipynb) | Speed vs XGPaint (~600×), and varying cosmology with **0 interpolator rebuilds** (the table is cosmology-independent). |

**Note:** the first GPU paint call in a session JIT-compiles the kernel (a few
seconds); every subsequent call — any catalogue or cosmology of the same halo
count — reuses it and runs in ~0.2 s.
