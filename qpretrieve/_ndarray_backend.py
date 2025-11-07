import importlib

_default_backend = "numpy"
_xp = importlib.import_module(_default_backend)


def get_ndarray_backend():
    return _xp


def set_ndarray_backend(backend_name: str = "numpy"):
    """Return requested ndarray backend

    Parameters
    ----------
    backend_name
        Options are 'numpy' and 'cupy'

    """
    global _xp
    try:
        if _xp.__name__ != backend_name:
            # we are actually swapping, so cache should be cleared
            import qpretrieve
            qpretrieve.filter.get_filter_array.cache_clear()

        # run the backend swap regardless
        _xp = importlib.import_module(backend_name)

    except ModuleNotFoundError as err:
        raise ImportError(f"The backend '{backend_name}' is not installed. "
                          f"Either install it or use the default backend: "
                          f"{_default_backend}") from err


def __getattr__(name):
    """Expose this module as a proxy for numpy or cupy"""
    return getattr(_xp, name)


def _is_numpy():
    return _xp.__name__.startswith("numpy")


def _is_cupy():
    return _xp.__name__.startswith("cupy")


def _assert_is_numpy():
    assert _is_numpy(), (
        "ndarray_backend is not 'numpy', to use "
        "'FFTFilterNumpy', run `qpretrieve.set_ndarray_backend('numpy')`.")


def _assert_is_cupy():
    assert _is_cupy(), (
        "ndarray_backend is not 'cupy', to use "
        "'FFTFilterCupy', run `qpretrieve.set_ndarray_backend('cupy')`.")


class NDArrayBackendWarning(UserWarning):
    pass
