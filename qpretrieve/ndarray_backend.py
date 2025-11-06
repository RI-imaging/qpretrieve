import warnings
import numpy as _numpy

try:
    import cupy as _cupy
except ImportError:
    _cupy = None
    warnings.warn("ndarray backend 'CuPy' unavailable!")


def get_ndarray_backend(requested_backend:str=None):
    """Return requested ndarray backend

    Parameters
    ----------
    requested_backend
        Options are 'numpy' and 'cupy'

    """
    if requested_backend is None or requested_backend == 'numpy':
        return _numpy

    if _cupy is None:
        return _numpy
    else:
        return _cupy
