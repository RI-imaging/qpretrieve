


def test_ndarray_backend_numpy_default():
    """should return the numpy module"""
    from qpretrieve.ndarray_backend import get_ndarray_backend

    xp = get_ndarray_backend()
    assert xp.__name__ == 'numpy'

    xp = get_ndarray_backend(requested_backend='numpy')
    assert xp.__name__ == 'numpy'



def test_ndarray_backend_cupy():
    """should return the numpy module"""
    from qpretrieve.ndarray_backend import get_ndarray_backend
    xp = get_ndarray_backend(requested_backend='cupy')

    assert xp.__name__ == 'cupy'
