import numpy as np
import pytest

import qpretrieve
from qpretrieve import fourier
from qpretrieve.fourier import FourierFieldData


def test_fftfilter_filter_can_return_fourier_domain():
    x = np.linspace(-100, 100, 100)
    xx, yy = np.meshgrid(x, -x, indexing="ij")
    gauss = np.exp(-(xx ** 2 + yy ** 2) / 625)

    ft = fourier.FFTFilterNumpy(gauss, subtract_mean=False)
    field = ft.filter(
        filter_name="disk",
        filter_size=0.25,
        freq_pos=(0, 0),
        scale_to_filter=False,
        output_domain="fourier",
    )

    assert isinstance(field, FourierFieldData)
    assert field.output_domain == "spatial"
    assert ft.fft_used is not None


def test_oah_run_pipeline_can_return_fourier_domain(hologram):
    holo = qpretrieve.OffAxisHologram(hologram)

    field = holo.run_pipeline(output_domain="fourier")

    assert isinstance(field, FourierFieldData)
    assert field.output_domain == "spatial"
    assert holo._field is None
    assert holo._fourier_field_data is field
    assert holo._phase is None
    assert holo._amplitude is None


def test_oah_compute_field_reuses_cached_intermediates(hologram):
    holo = qpretrieve.OffAxisHologram(hologram)

    artifact = holo.run_pipeline(output_domain="fourier")
    field = holo.compute_field()

    assert field is holo._field
    assert field.shape == (1, hologram.shape[0], hologram.shape[1])
    assert holo._phase is None
    assert holo._amplitude is None
    assert artifact.field is field


def test_oah_compute_field_matches_output_domain_spatial(hologram):
    holo_spatial = qpretrieve.OffAxisHologram(hologram)
    field_spatial = holo_spatial.run_pipeline(output_domain="spatial")

    holo_fourier = qpretrieve.OffAxisHologram(hologram)
    artifact_fourier = holo_fourier.run_pipeline(output_domain="fourier")
    field_fourier = artifact_fourier.finalize()

    assert np.allclose(field_spatial, field_fourier)


def test_oah_compute_field_accepts_propagated_fft(hologram):
    holo = qpretrieve.OffAxisHologram(hologram)
    artifact = holo.run_pipeline(output_domain="fourier")

    # here the propagated_fft could be a fft from nrefocus
    field = holo.compute_field(propagated_fft=artifact.fft_used)

    assert field is holo._field
    assert holo.phase.shape == field.shape
    assert holo.amplitude.shape == field.shape


def test_ffd_subtract_background(hologram):
    """subtract_background() in Fourier space equals spatial
    complex-field subtraction."""
    rng = np.random.default_rng(0)
    bg_image = hologram + rng.normal(scale=0.5, size=hologram.shape)

    oah = qpretrieve.OffAxisHologram(hologram)
    oah_bg = qpretrieve.OffAxisHologram(bg_image)

    # Run the sample pipeline in Fourier domain to establish filter settings
    ffd_sample = oah.run_pipeline(output_domain="fourier")
    # Apply identical filter/sideband settings to the background
    oah_bg.process_like(oah)  # pipeline_kws includes output_domain="fourier"
    ffd_bg = oah_bg._fourier_field_data

    assert isinstance(ffd_sample, FourierFieldData)
    assert isinstance(ffd_bg, FourierFieldData)

    # Subtract in Fourier space then finalize
    ffd_corrected = ffd_sample.subtract_background(ffd_bg)
    assert isinstance(ffd_corrected, FourierFieldData)
    field_corrected = ffd_corrected.finalize()

    # Spatial-domain equivalent: iFFT is linear so
    # ifft(A-B) == ifft(A) - ifft(B)
    field_sample = ffd_sample.finalize()
    field_bg = ffd_bg.finalize()

    assert np.allclose(field_corrected, field_sample - field_bg)


def test_ffd_subtract_background_shape_mismatch(hologram):
    """subtract_background() raises ValueError when fft_used shapes differ."""
    oah = qpretrieve.OffAxisHologram(hologram)
    ffd_sample = oah.run_pipeline(output_domain="fourier")

    # Build a background hologram of a different spatial size
    bg_image = hologram[:hologram.shape[0] // 2, :hologram.shape[1] // 2]
    oah_bg = qpretrieve.OffAxisHologram(bg_image)
    ffd_bg = oah_bg.run_pipeline(output_domain="fourier")

    with pytest.raises(ValueError, match="Shape mismatch"):
        ffd_sample.subtract_background(ffd_bg)
