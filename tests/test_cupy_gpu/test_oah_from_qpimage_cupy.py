"""These are tests from qpimage for Cupy imported `FFTFilter`s"""
import pytest
import numpy as np

import qpretrieve
from qpretrieve.fourier import FFTFilterNumpy, FFTFilterCupy, FFTFilterPyFFTW
from qpretrieve._ndarray_backend import xp, NDArrayBackendWarning

from ..helper_methods import skip_if_missing


@skip_if_missing("cupy")
def test_get_field_compare_cupy2d(hologram, set_ndarray_backend_to_cupy):
    assert xp.is_cupy()
    holo1 = qpretrieve.OffAxisHologram(hologram,
                                       fft_interface=FFTFilterCupy,
                                       padding=False)
    kwargs = dict(filter_name="disk", filter_size=1 / 3)
    res1 = holo1.run_pipeline(**kwargs)

    qpretrieve.set_ndarray_backend("numpy")
    assert xp.is_numpy()
    holo2 = qpretrieve.OffAxisHologram(hologram,
                                       fft_interface=FFTFilterNumpy,
                                       padding=False)
    kwargs = dict(filter_name="disk", filter_size=1 / 3)
    res2 = holo2.run_pipeline(**kwargs)

    assert res1.shape == (1, 64, 64)
    assert res2.shape == (1, 64, 64)

    assert np.allclose(res1[0], res2[0], rtol=0, atol=1e-12)


@skip_if_missing("cupy")
def test_get_field_compare_cupy3d(hologram, set_ndarray_backend_to_cupy):
    data_rp = np.array([hologram, hologram, hologram, hologram, hologram])

    assert xp.is_cupy()
    holo1 = qpretrieve.OffAxisHologram(data_rp,
                                       fft_interface=FFTFilterCupy,
                                       padding=False)
    kwargs = dict(filter_name="disk", filter_size=1 / 3)
    res1 = holo1.run_pipeline(**kwargs)
    assert res1.shape == (5, 64, 64)

    qpretrieve.set_ndarray_backend("numpy")
    assert xp.is_numpy()
    holo1 = qpretrieve.OffAxisHologram(data_rp,
                                       fft_interface=FFTFilterNumpy,
                                       padding=False)
    kwargs = dict(filter_name="disk", filter_size=1 / 3)
    res2 = holo1.run_pipeline(**kwargs)
    assert res2.shape == (5, 64, 64)

    assert np.allclose(res1, res2, rtol=0, atol=1e-12)


@skip_if_missing("cupy")
def test_get_field_cupy3d_scale_to_filter(hologram,
                                          set_ndarray_backend_to_cupy):
    data_rp = np.array([hologram, hologram, hologram, hologram, hologram])

    assert xp.is_cupy()
    holo1 = qpretrieve.OffAxisHologram(data_rp,
                                       fft_interface=FFTFilterCupy,
                                       padding=True)
    kwargs = dict(filter_name="disk", filter_size=1 / 3,
                  scale_to_filter=True)
    res1 = holo1.run_pipeline(**kwargs)

    qpretrieve.set_ndarray_backend("numpy")
    assert xp.is_numpy()
    holo2 = qpretrieve.OffAxisHologram(data_rp,
                                       fft_interface=FFTFilterNumpy,
                                       padding=True)
    kwargs = dict(filter_name="disk", filter_size=1 / 3,
                  scale_to_filter=True)
    res2 = holo2.run_pipeline(**kwargs)

    assert res1.shape == (5, 18, 18)
    assert res2.shape == (5, 18, 18)
    assert np.allclose(res1[0], res2[0], rtol=0, atol=1e-13)


@skip_if_missing("cupy")
def test_get_field_backend_mixup_fail(hologram, set_ndarray_backend_to_cupy):
    kwargs = dict(filter_name="disk", filter_size=1 / 3)

    assert xp.is_cupy()
    holo1 = qpretrieve.OffAxisHologram(hologram,
                                       fft_interface=FFTFilterCupy,
                                       padding=False)
    holo1.run_pipeline(**kwargs)

    qpretrieve.set_ndarray_backend("numpy")
    assert xp.is_numpy()
    with pytest.raises(TypeError,
                       match=r"Implicit conversion to a NumPy array is not "
                             r"allowed. Please use `.get\(\)` to construct a "
                             r"NumPy array explicitly"):
        # user runs the CuPy pipeline after changing to numpy backend
        holo1.run_pipeline(**kwargs)


@skip_if_missing("cupy")
def test_fftfilter_backend_mismatch(hologram):
    """Shows how a FFTFilter and ndarray backend mismatch creates a warning"""
    # this works but provides a user warning
    wrong_backend = "numpy"
    expected_backend = "cupy"
    fft_interface = FFTFilterCupy
    qpretrieve.set_ndarray_backend(wrong_backend)
    with pytest.warns(
            NDArrayBackendWarning,
            match=rf"You are using `{fft_interface.__name__}` "
                  rf"with the '{wrong_backend}' ndarray backend. This might "
                  rf"limit the FFT speed. To set the correct ndarray backend, "
                  rf"use `qpretrieve.set_ndarray_backend\('{expected_backend}'\)`"
    ):
        _ = qpretrieve.OffAxisHologram(hologram,
                                       fft_interface=fft_interface,
                                       padding=False)

    # this works but provides a user warning
    expected_backend = "numpy"
    wrong_backend = "cupy"
    fft_interface = FFTFilterNumpy
    qpretrieve.set_ndarray_backend(wrong_backend)
    with pytest.warns(
            NDArrayBackendWarning,
            match=rf"You are using `{fft_interface.__name__}` "
                  rf"with the '{wrong_backend}' ndarray backend. This might "
                  rf"limit the FFT speed. To set the correct ndarray backend, "
                  rf"use `qpretrieve.set_ndarray_backend\('{expected_backend}'\)`"
    ):
        _ = qpretrieve.OffAxisHologram(hologram,
                                       fft_interface=fft_interface,
                                       padding=False)

    # this fails because pyfftw arrays don't work with cupy
    wrong_backend = "cupy"
    fft_interface = FFTFilterPyFFTW
    qpretrieve.set_ndarray_backend(wrong_backend)
    with pytest.raises(
            TypeError,
            match=r"Implicit conversion to a NumPy array is not "
                  r"allowed. Please use `.get\(\)` to construct a "
                  r"NumPy array explicitly"
    ):
        _ = qpretrieve.OffAxisHologram(hologram,
                                       fft_interface=fft_interface,
                                       padding=False)
