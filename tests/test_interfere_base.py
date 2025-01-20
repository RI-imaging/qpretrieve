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
        fft_interface=qpretrieve.fourier.FFTFilterNumpy)
    assert holo.ff_iface.is_available
    assert issubclass(holo.ff_iface,
                      qpretrieve.fourier.base.FFTFilter)
    assert issubclass(holo.ff_iface,
                      qpretrieve.fourier.ff_numpy.FFTFilterNumpy)


def test_interfere_base_bad_interface():
    edata = np.load(data_path / "hologram_cell.npz")

    with pytest.raises(ValueError):
        _ = qpretrieve.OffAxisHologram(
            data=edata["data"],
            fft_interface="MyReallyCoolFFTInterface")


def test_interfere_base_orig_data_fmt():
    edata = np.load(data_path / "hologram_cell.npz")

    holo = qpretrieve.OffAxisHologram(data=edata["data"])
    assert holo.orig_data_fmt is not None
    assert holo.orig_data_fmt == "2d"


def test_interfere_base_orig_data_fmt_get_original_format():
    edata = np.load(data_path / "hologram_cell.npz")
    orig_shape = (200, 210)
    assert edata["data"].shape == orig_shape

    holo = qpretrieve.OffAxisHologram(data=edata["data"])
    assert holo.field.shape == (1, 200, 210)

    field_orig = holo.get_orig_orig_data_fmt(data_attr=holo.field)

    assert field_orig.shape == orig_shape
