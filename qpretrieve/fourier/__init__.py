# flake8: noqa: F401
import warnings

from .ff_numpy import FFTFilterNumpy
from .ff_scipy import FFTFilterScipy

try:
    from .ff_pyfftw import FFTFilterPyFFTW
except ImportError:
    FFTFilterPyFFTW = None

try:
    from .ff_cupy import FFTFilterCupy
except ImportError:
    FFTFilterCupy = None

PREFERRED_INTERFACE = None


def get_available_interfaces():
    """Return a list of available FFT algorithms"""
    interfaces = [
        FFTFilterPyFFTW,
        FFTFilterNumpy,
        FFTFilterScipy,
        FFTFilterCupy,
    ]
    interfaces_available = []
    for interface in interfaces:
        if interface is not None and interface.is_available:
            interfaces_available.append(interface)
    return interfaces_available


def get_best_interface():
    """Return the fastest refocusing interface available

    If `pyfftw` is installed, :class:`.FFTFilterPyFFTW`
    is returned. The fallback is :class:`.FFTFilterNumpy`.
    """
    ordered_candidates = [
        FFTFilterPyFFTW,
        FFTFilterNumpy,
    ]
    if PREFERRED_INTERFACE is not None:
        for cand in ordered_candidates:
            if (cand is not None
                    and cand.is_available
                    and cand.__name__ == PREFERRED_INTERFACE):
                return cand
        else:
            warnings.warn(
                f"Preferred interface '{PREFERRED_INTERFACE}' unavailable!")

    for cand in ordered_candidates:
        if cand is not None and cand.is_available:
            return cand
