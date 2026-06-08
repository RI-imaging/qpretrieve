import numpy as np

import qpretrieve
from qpretrieve import fourier
from qpretrieve.fourier import FourierFieldArtifact


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

    assert isinstance(field, FourierFieldArtifact)
    assert field.output_domain == "spatial"
    assert ft.fft_used is not None


def test_oah_run_pipeline_can_return_fourier_domain(hologram):
    holo = qpretrieve.OffAxisHologram(hologram)

    field = holo.run_pipeline(output_domain="fourier")

    assert isinstance(field, FourierFieldArtifact)
    assert field.output_domain == "spatial"
    assert holo._field is None
    assert holo._field_artifact is field
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
