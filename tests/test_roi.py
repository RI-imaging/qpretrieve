import numpy as np

import qpretrieve


def test_run_pipeline_rois_identity(hologram):
    """ROI path should match full pipeline when ROI covers whole frame."""
    holo_full = qpretrieve.OffAxisHologram(hologram)
    ref = holo_full.run_pipeline()

    mask_all = np.ones_like(hologram, dtype=bool)
    holo_roi = qpretrieve.OffAxisHologram(hologram)
    roi_result = holo_roi.run_pipeline_rois(roi_mask=mask_all)

    assert np.allclose(ref, roi_result)


def test_run_pipeline_rois_background_fill(hologram):
    """Background outside ROI should be filled as requested."""
    mask = np.zeros_like(hologram, dtype=bool)
    mask[16:48, 16:48] = True

    holo = qpretrieve.OffAxisHologram(hologram)
    res = holo.run_pipeline_rois(
        roi_mask=mask,
        background_fill="zero",
    )

    outside = ~mask
    assert np.all(res[0][outside] == 0)
    assert np.any(res[0][mask] != 0)


def test_run_pipeline_rois_no_stitch(hologram):
    """stitch=False should return only the ROI field."""
    box = (16, 48, 16, 48)

    # Reference: run pipeline directly on cropped data
    direct = qpretrieve.OffAxisHologram(hologram[box[0]:box[1], box[2]:box[3]])
    ref_field = direct.run_pipeline()

    # ROI path without stitching
    holo = qpretrieve.OffAxisHologram(hologram)
    roi_field = holo.run_pipeline_rois(roi_boxes=[box], stitch=False)

    assert roi_field.shape == ref_field.shape
    assert np.allclose(ref_field, roi_field)
