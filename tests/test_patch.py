import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class UnitShapeTable:
    logtheta_min = np.log(1e-9)

    def evaluate(self, logtheta, logtheta500):
        return np.ones_like(logtheta, dtype=np.float64)


def test_patch_paints_centered_halo():
    from jxpaint.painting.patch import paint_catalogue_patch
    from jxpaint import constants as C

    image, meta = paint_catalogue_patch(
        z=np.array([0.2]),
        M_1e14=np.array([5.0]),
        lon=np.array([0.0]),
        lat=np.array([np.pi / 2.0]),
        y0_true=np.array([2.0]),
        shape_table=UnitShapeTable(),
        center_ra_deg=0.0,
        center_dec_deg=0.0,
        width_deg=2.0,
        height_deg=2.0,
        pixel_size_arcmin=10.0,
    )

    assert image.shape == (12, 12)
    assert meta["candidate_halos"] == 1
    assert image.max() == 2.0 / C.B_BIAS ** (1.0 / 3.0)
    assert image.sum() > image.max()


def test_patch_ignores_distant_halo():
    from jxpaint.painting.patch import paint_catalogue_patch

    image, meta = paint_catalogue_patch(
        z=np.array([0.2]),
        M_1e14=np.array([5.0]),
        lon=np.array([np.deg2rad(90.0)]),
        lat=np.array([np.pi / 2.0]),
        y0_true=np.array([2.0]),
        shape_table=UnitShapeTable(),
        center_ra_deg=0.0,
        center_dec_deg=0.0,
        width_deg=2.0,
        height_deg=2.0,
        pixel_size_arcmin=10.0,
    )

    assert meta["candidate_halos"] == 0
    assert np.count_nonzero(image) == 0
