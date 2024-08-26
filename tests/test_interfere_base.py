import pathlib
import numpy as np
import pytest

import qpretrieve

data_path = pathlib.Path(__file__).parent / "data"


def test_interfere_base_best_interface():
    edata = np.load(data_path / "hologram_cell.npz")

    holo = qpretrieve.OffAxisHologram(data=edata["data"])
    assert holo.ff_iface.is_available
    assert issubclass(holo.ff_iface,
                      qpretrieve.fourier.base.FFTFilter)
    assert issubclass(holo.ff_iface,
                      qpretrieve.fourier.ff_numpy.FFTFilterNumpy)


def test_interfere_base_choose_interface():
    edata = np.load(data_path / "hologram_cell.npz")

    holo = qpretrieve.OffAxisHologram(
        data=edata["data"],
        fft_interface=qpretrieve.fourier.FFTFilterScipy)
    assert holo.ff_iface.is_available
    assert issubclass(holo.ff_iface,
                      qpretrieve.fourier.base.FFTFilter)
    assert issubclass(holo.ff_iface,
                      qpretrieve.fourier.ff_scipy.FFTFilterScipy)


def test_interfere_base_bad_interface():
    edata = np.load(data_path / "hologram_cell.npz")

    with pytest.raises(ValueError):
        _ = qpretrieve.OffAxisHologram(
            data=edata["data"],
            fft_interface="MyReallyCoolFFTInterface")
