import numpy as np

import qpretrieve
from qpretrieve.fourier import FFTFilterCupy
from qpretrieve._ndarray_backend import xp, NDArrayBackendWarning

import pytest

from ..helper_methods import skip_if_missing


@skip_if_missing("cupy")
def test_cupy3d_backend_swap(hologram, set_ndarray_backend_to_numpy):

    # use FFTFilterCupy with numpy backend
    assert xp.is_numpy()

    data1 = hologram
    data_rp = np.array([data1, data1, data1, data1, data1])

    with pytest.warns(NDArrayBackendWarning):
        holo1 = qpretrieve.OffAxisHologram(data_rp,
                                           fft_interface=FFTFilterCupy,
                                           padding=False)
        kwargs = dict(filter_name="disk", filter_size=1 / 3)
        res1 = holo1.run_pipeline(**kwargs)
        assert res1.shape == (5, 64, 64)

    # use FFTFilterCupy with cupy backend
    qpretrieve.set_ndarray_backend('cupy')
    assert xp.is_cupy()

    holo1 = qpretrieve.OffAxisHologram(data_rp,
                                       fft_interface=FFTFilterCupy,
                                       padding=False)
    kwargs = dict(filter_name="disk", filter_size=1 / 3)
    res2 = holo1.run_pipeline(**kwargs)
    assert res2.shape == (5, 64, 64)

    assert np.allclose(res1, res2, rtol=0, atol=1e-13)
