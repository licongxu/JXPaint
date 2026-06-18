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
| [`01_quickstart.ipynb`](01_quickstart.ipynb) | Paint a Compton-$y$ map on the GPU (~0.2 s), validate bit-for-bit vs the XGPaint reference, and compare the power spectrum to the **hmfast** halo-model theory (imported directly; beam-deconvolved data vs unbeamed Arnaud-GNFW 1-halo). |
| [`02_profiles.ipynb`](02_profiles.ipynb) | The CustomGNFW profile: 3D shape, projected $F(x)$, $\theta_{500}$ and the $y_0$ amplitudes vs $(M,z)$, and the beam-convolved 2D shape table. |
| [`03_speed_and_cosmology.ipynb`](03_speed_and_cosmology.ipynb) | Speed vs XGPaint (~600×), time vs $N_{\rm side}$ scaling, and varying cosmology with **0 interpolator rebuilds** (the table is cosmology-independent). |
| [`04_highres_beam.ipynb`](04_highres_beam.ipynb) | Build a **1.4 arcmin** beam table (`build_beamed_shape_table`), paint at **$N_{\rm side}=4096$**, and compare the beam-deconvolved spectrum to the unbeamed hmfast theory out to $\ell\sim4000$. |
| [`05_painting_patches.py`](05_painting_patches.py) | Paint a rectangular flat-sky catalogue cutout, e.g. **10 x 10 degrees**, save it as `.npz`, and make a quick PNG preview. |
| [`06_arnaud_b1_fullsky_and_patches.py`](06_arnaud_b1_fullsky_and_patches.py) | Demo commands for full-sky and **10 x 10 degree** patch painting with the simple Arnaud gNFW pressure profile and `B=1`. |

**Note:** the first GPU paint call in a session JIT-compiles the kernel (a few
seconds); every subsequent call — any catalogue or cosmology of the same halo
count — reuses it and runs in ~0.2 s.
