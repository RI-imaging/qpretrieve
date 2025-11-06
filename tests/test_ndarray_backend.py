import qpretrieve

import pytest


def test_ndarray_backend_numpy_default():
    """should return numpy"""

    assert qpretrieve._ndarray_backend._is_numpy()
    qpretrieve.set_ndarray_backend('numpy')
    assert qpretrieve._ndarray_backend._is_numpy()


def test_ndarray_backend_cupy():
    """should return cupy"""
    assert qpretrieve._ndarray_backend._is_numpy()
    qpretrieve.set_ndarray_backend('cupy')
    assert qpretrieve._ndarray_backend._is_cupy()


def test_ndarray_backend_swap():
    """should return the correct set backend"""
    qpretrieve.set_ndarray_backend('cupy')
    assert qpretrieve._ndarray_backend._is_cupy()
    qpretrieve.set_ndarray_backend('numpy')
    assert qpretrieve._ndarray_backend._is_numpy()
    qpretrieve.set_ndarray_backend('cupy')
    assert qpretrieve._ndarray_backend._is_cupy()


def test_ndarray_backend_bad():
    """should raise an ImportError"""
    bad_backend = "funpy"
    match_err_str = (f"The backend '{bad_backend}' is not installed. "
                     f"Either install it or use the default backend: numpy")

    with pytest.raises(ImportError, match=match_err_str):
        qpretrieve.set_ndarray_backend(bad_backend)
