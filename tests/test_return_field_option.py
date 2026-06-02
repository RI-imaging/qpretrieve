import numpy as np

import qpretrieve
from qpretrieve import fourier


def test_fftfilter_filter_can_skip_returning_field():
    x = np.linspace(-100, 100, 100)
    xx, yy = np.meshgrid(x, -x, indexing="ij")
    gauss = np.exp(-(xx ** 2 + yy ** 2) / 625)

    ft = fourier.FFTFilterNumpy(gauss, subtract_mean=False)
    field = ft.filter(
        filter_name="disk",
        filter_size=0.25,
        freq_pos=(0, 0),
        scale_to_filter=False,
        return_field=False,
    )

    assert field is None
    assert ft.fft_used is not None


def test_oah_run_pipeline_can_skip_returning_field(hologram):
    holo = qpretrieve.OffAxisHologram(hologram)

    field = holo.run_pipeline(return_field=False)

    assert field is None
    assert holo._field is None
    assert holo._phase is None
    assert holo._amplitude is None


def test_oah_compute_field_reuses_cached_intermediates(hologram):
    holo = qpretrieve.OffAxisHologram(hologram)

    holo.run_pipeline(return_field=False)
    field = holo.compute_field()

    assert field is holo._field
    assert field.shape == (1, hologram.shape[0], hologram.shape[1])
    assert holo._phase is None
    assert holo._amplitude is None
