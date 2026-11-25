"""Benchmarks for background correction over sample-specific distance scans.

If you need to do background correction, it is often faster to
just correct for the background and then propagate (the typical way).
If you do not need background correction, use the direct FFT method.

Run this script with
`pytest tests/test_bm/test_bm_background_correction.py`
to see the differences for the example dataset.

"""
from pathlib import Path

import pytest
import qpretrieve
from qpretrieve.fourier.fourier_field_data import finalize_fourier_field

pytest.importorskip("nrefocus")

import nrefocus  # noqa: E402
import numpy as np  # noqa: E402

_DATA_PATH = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "data"
        / "hologram_cell.npz"
)
_DISTANCE_STEP_UM = 0.1
_SCAN_RANGES_UM = (
    pytest.param((1, 5), id="sample-1-range"),
    pytest.param((3, 7), id="sample-2-range"),
)
_PROPAGATION_KWARGS = dict(
    nm=1.533, res=8.25, method="fresnel", padding=False
)


def _distances(distance_range_um):
    start_um, stop_um = distance_range_um
    count = int(round((stop_um - start_um) / _DISTANCE_STEP_UM)) + 1
    return np.linspace(start_um, stop_um, count)


def _refocus(field, distance_um):
    # nrefocus takes distance in pixels; here one pixel corresponds to 1 µm.
    return nrefocus.refocus(
        field=field,
        d=distance_um,
        output_domain="spatial",
        **_PROPAGATION_KWARGS,
    )


def _finalize_uncached(fourier_field):
    """Perform qpretrieve's iFFT/crop without reusing artifact.field cache."""
    return finalize_fourier_field(
        fft_in=fourier_field.fft_used,
        ifft_fn=fourier_field.ifft_fn,
        input_shape=fourier_field.input_shape,
        fft_shape=fourier_field.fft_shape,
        padding=fourier_field.padding,
        scale_to_filter=fourier_field.scale_to_filter,
        crop_radius=fourier_field.crop_radius,
    )


@pytest.fixture(scope="module")
def background_scan_fields():
    """Prepare the shared qpretrieve Fourier fields outside timed scans."""
    with np.load(_DATA_PATH) as data:
        sample_data = data["data"]
        background_data = data["bg_data"]

    sample_holo = qpretrieve.OffAxisHologram(sample_data)
    sample_fourier = sample_holo.run_pipeline(
        filter_name="smooth disk",
        filter_size=1 / 2,
        output_domain="fourier",
    )
    background_holo = qpretrieve.OffAxisHologram(background_data)
    background_holo.process_like(sample_holo)
    background_fourier = background_holo._fourier_field_data

    # The direct path can reuse the reconstructed background field for every
    # sample, while the propagated path needs a background field per distance.
    background_field = _finalize_uncached(background_fourier)
    return sample_fourier, background_fourier, background_field


def _scan_correct_then_propagate(
        sample_fourier, background_field, distance_range_um
):
    """iFFT and divide sample/background once, then propagate each distance."""
    sample_field = _finalize_uncached(sample_fourier)
    corrected_field = sample_field / background_field
    height, width = corrected_field.shape[-2:]
    padded = np.zeros(
        (corrected_field.shape[0], *sample_fourier.fft_shape),
        dtype=corrected_field.dtype,
    )
    padded[:, :height, :width] = corrected_field

    outputs = []
    for distance_um in _distances(distance_range_um):
        propagated = _refocus(padded, distance_um)
        outputs.append(propagated[0, :height, :width])
    return outputs


def _scan_propagate_then_correct(
        sample_fourier, background_fourier, distance_range_um
):
    """Propagate both Fourier fields at each distance, then divide."""
    outputs = []
    background_cache = {}
    for distance_um in _distances(distance_range_um):
        key = float(distance_um)
        if key not in background_cache:
            background_cache[key] = _refocus(background_fourier, distance_um)
        sample_propagated = _refocus(sample_fourier, distance_um)
        outputs.append(
            np.divide(sample_propagated, background_cache[key])[0]
        )
    return outputs


def _assert_scan_outputs(outputs, input_shape, distance_range_um):
    assert len(outputs) == len(_distances(distance_range_um))
    assert all(output.shape == input_shape for output in outputs)


@pytest.mark.parametrize("distance_range_um", _SCAN_RANGES_UM)
def test_bm_background_correct_then_propagate(
        benchmark, background_scan_fields, distance_range_um
):
    """Benchmark the spatial correction workflow for a complete fine scan."""
    sample_fourier, _, background_field = background_scan_fields
    result = benchmark(
        _scan_correct_then_propagate,
        sample_fourier,
        background_field,
        distance_range_um,
    )
    _assert_scan_outputs(
        result, sample_fourier.input_shape, distance_range_um
    )


@pytest.mark.parametrize("distance_range_um", _SCAN_RANGES_UM)
def test_bm_background_propagate_then_correct(
        benchmark, background_scan_fields, distance_range_um
):
    """Benchmark propagating both fields at every fine-scan distance."""
    sample_fourier, background_fourier, _ = background_scan_fields
    result = benchmark(
        _scan_propagate_then_correct,
        sample_fourier,
        background_fourier,
        distance_range_um,
    )
    _assert_scan_outputs(
        result, sample_fourier.input_shape, distance_range_um
    )
