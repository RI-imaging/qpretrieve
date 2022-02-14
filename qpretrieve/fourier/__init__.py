# flake8: noqa: F401
from .ff_numpy import FFTFilterNumpy

try:
    from .ff_pyfftw import FFTFilterPyFFTW
except ImportError:
    FFTFilterPyFFTW = None


def get_best_interface():
    """Return the fastest refocusing interface available

    If `pyfftw` is installed, :class:`.FFTFilterPyFFTW`
    is returned. The fallback is :class:`.FFTFilterNumpy`.
    """
    ordered_candidates = [
        FFTFilterPyFFTW,
        FFTFilterNumpy,
    ]
    for cand in ordered_candidates:
        if cand is not None and cand.is_available:
            return cand
