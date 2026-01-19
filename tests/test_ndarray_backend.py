import pytest

import qpretrieve
from qpretrieve._ndarray_backend import xp

from .helper_methods import skip_if_missing


def test_ndarray_backend_numpy_default():
    """should return numpy"""

    assert xp.is_numpy()
    qpretrieve.set_ndarray_backend('numpy')
    assert xp.is_numpy()


def test_ndarray_backend_bad():
    """should raise an ImportError"""
    bad_backend = "funpy"
    match_err_str = (f"The backend '{bad_backend}' is not installed. "
                     f"Either install it or use the default backend: 'numpy'.")

    with pytest.raises(ImportError, match=match_err_str):
        qpretrieve.set_ndarray_backend(bad_backend)


@skip_if_missing("cupy")
def test_ndarray_backend_cupy():
    """should return cupy"""
    assert xp.is_numpy()
    qpretrieve.set_ndarray_backend('cupy')
    assert xp.is_cupy()


@skip_if_missing("cupy")
def test_ndarray_backend_swap():
    """should return the correct set backend"""
    qpretrieve.set_ndarray_backend('cupy')
    assert xp.is_cupy()
    qpretrieve.set_ndarray_backend('numpy')
    assert xp.is_numpy()
    qpretrieve.set_ndarray_backend('cupy')
    assert xp.is_cupy()
