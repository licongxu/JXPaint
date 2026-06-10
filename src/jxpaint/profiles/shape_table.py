"""Beam-convolved 2D shape table y_t(log theta, log theta500).

Loads the exact production table dumped from XGPaint's cached `.jld2`
(reference/data/beamed_table_coefs.f64 + _axes.txt) and evaluates it with
**bicubic B-spline** interpolation, matching Interpolations.jl
`extrapolate(scale(interpolate(A, BSpline(Cubic(Line(OnGrid))))), 0.0)`.

The dumped array is the prefiltered coefficient matrix (parent of the
OffsetArray, shape (Ngrid+2, Ngrid+2)).  Native BSpline grid positions run
1..Ngrid; coefficient offset-index c (0..Ngrid+1) maps to numpy row c.
Cubic B-spline weights for the 4 coefs at offsets {-1,0,1,2} of ix=floor(p):
    w[-1]=(1-f)^3/6, w[0]=(3f^3-6f^2+4)/6,
    w[ 1]=(-3f^3+3f^2+3f+1)/6, w[2]=f^3/6.
Zero (FilledExtrapolation) outside [lt_min,lt_max] x [l5_min,l5_max].
"""
import os
import numpy as np
import jax
import jax.numpy as jnp

_DEF_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                        "reference", "data")


def _cubic_weights(f):
    """Cubic B-spline weights for offsets [-1,0,1,2]; f in [0,1). Returns (...,4)."""
    f2 = f * f
    f3 = f2 * f
    wm1 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0     # (1-f)^3/6
    w0 = (3.0 * f3 - 6.0 * f2 + 4.0) / 6.0
    w1 = (-3.0 * f3 + 3.0 * f2 + 3.0 * f + 1.0) / 6.0
    w2 = f3 / 6.0
    return jnp.stack([wm1, w0, w1, w2], axis=-1)


class BeamedShapeTable:
    """Bicubic B-spline interpolator over a regular (log theta, log theta500) grid."""

    def __init__(self, coefs, logtheta_min, logtheta_max, logtheta500_min,
                 logtheta500_max, fillvalue=0.0):
        # coefs: prefiltered parent matrix (Ngrid+2, Ngrid+2).  Kept as numpy so
        # loading does not initialise CUDA (lets the painter fork before any GPU
        # use); a device copy is made lazily on first GPU evaluate.
        self.coefs = np.asarray(coefs, dtype=np.float64)
        self._coefs_dev = None
        self.P0, self.P1 = self.coefs.shape           # Ngrid+2
        self.n0 = self.P0 - 2                          # grid points axis0
        self.n1 = self.P1 - 2
        self.lt_min = float(logtheta_min)
        self.lt_max = float(logtheta_max)
        self.l5_min = float(logtheta500_min)
        self.l5_max = float(logtheta500_max)
        self.dlt = (self.lt_max - self.lt_min) / (self.n0 - 1)
        self.dl5 = (self.l5_max - self.l5_min) / (self.n1 - 1)
        self.fill = float(fillvalue)

    @property
    def logtheta_min(self):
        return self.lt_min

    def coefs_device(self):
        """Device (jnp) copy of the coefficient table; uploaded once, cached."""
        if self._coefs_dev is None:
            self._coefs_dev = jnp.asarray(self.coefs, dtype=jnp.float64)
        return self._coefs_dev

    def evaluate(self, logtheta, logtheta500):
        """Bicubic y_t at (logtheta, logtheta500); zero outside the grid."""
        coefs = self.coefs_device()
        lt = jnp.asarray(logtheta, dtype=jnp.float64)
        l5 = jnp.asarray(logtheta500, dtype=jnp.float64)

        # native 1-based grid positions
        p0 = 1.0 + (lt - self.lt_min) / self.dlt
        p1 = 1.0 + (l5 - self.l5_min) / self.dl5

        inside = ((lt >= self.lt_min) & (lt <= self.lt_max)
                  & (l5 >= self.l5_min) & (l5 <= self.l5_max))

        # clamp so floor stays in [1, n-1] -> neighbor rows in [0, Ngrid+1]
        p0c = jnp.clip(p0, 1.0, float(self.n0))
        p1c = jnp.clip(p1, 1.0, float(self.n1))
        ix0 = jnp.floor(p0c).astype(jnp.int32)
        ix1 = jnp.floor(p1c).astype(jnp.int32)
        ix0 = jnp.clip(ix0, 1, self.n0 - 1)
        ix1 = jnp.clip(ix1, 1, self.n1 - 1)
        f0 = p0c - ix0
        f1 = p1c - ix1

        w0 = _cubic_weights(f0)    # (...,4) offsets -1..2  -> rows ix0-1..ix0+2
        w1 = _cubic_weights(f1)    # (...,4)

        # numpy rows = coef offset index = ix-1 + k, k=0..3
        r0 = ix0[..., None] - 1 + jnp.arange(4)     # (...,4)
        r1 = ix1[..., None] - 1 + jnp.arange(4)
        r0 = jnp.clip(r0, 0, self.P0 - 1)
        r1 = jnp.clip(r1, 0, self.P1 - 1)

        # gather 4x4 block: coefs[r0[...,a], r1[...,b]]
        block = coefs[r0[..., :, None], r1[..., None, :]]   # (...,4,4)
        val = jnp.einsum("...a,...ab,...b->...", w0, block, w1)
        return jnp.where(inside, val, self.fill)


def load_beamed_table(data_dir=_DEF_DIR) -> BeamedShapeTable:
    """Load the dumped production beamed shape table (bicubic coefs)."""
    axes_path = os.path.join(data_dir, "beamed_table_axes.txt")
    bin_path = os.path.join(data_dir, "beamed_table_coefs.f64")
    with open(axes_path) as f:
        lines = [l.strip() for l in f]
    lt_min, lt_max, _n0 = lines[1].split(",")
    l5_min, l5_max, _n1 = lines[3].split(",")
    fill = float(lines[5])
    p0n, p1n = lines[7].split(",")[:2]
    p0n, p1n = int(p0n), int(p1n)
    raw = np.fromfile(bin_path, dtype=np.float64)
    assert raw.size == p0n * p1n, f"table size {raw.size} != {p0n}*{p1n}"
    coefs = raw.reshape((p0n, p1n), order="F")    # Julia column-major parent
    return BeamedShapeTable(coefs, float(lt_min), float(lt_max),
                            float(l5_min), float(l5_max), fill)
