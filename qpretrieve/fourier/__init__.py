# flake8: noqa: F401
import warnings
from typing import Type

from .base import FFTFilter
from .ff_numpy import FFTFilterNumpy

try:
    from .ff_pyfftw import FFTFilterPyFFTW
except ImportError:
    FFTFilterPyFFTW = None
    warnings.warn("Interface 'FFTFilterPyFFTW' unavailable!")

try:
    from .ff_cupy import FFTFilterCupy
except ImportError:
    FFTFilterCupy = None
    warnings.warn("Interface 'FFTFilterCupy' unavailable!")

PREFERRED_INTERFACE = None


def get_available_interfaces() -> list[Type[FFTFilter]]:
    """Return a list of available FFT algorithms"""
    interfaces = [
        FFTFilterPyFFTW,
        FFTFilterNumpy,
        FFTFilterCupy,
    ]
    interfaces_available = []
    for interface in interfaces:
        if interface is not None and interface.is_available:
            interfaces_available.append(interface)
    return interfaces_available


def get_best_interface() -> Type[FFTFilter]:
    """Return the fastest refocusing interface available

    If `pyfftw` is installed, :class:`.FFTFilterPyFFTW`
    is returned. The fallback is :class:`.FFTFilterNumpy`.
    """
    ordered_candidates = [
        FFTFilterPyFFTW,
        FFTFilterNumpy,
        FFTFilterCupy,
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
