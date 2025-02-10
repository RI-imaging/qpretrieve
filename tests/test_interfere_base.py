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

    for InterferCls in [qpretrieve.OffAxisHologram,
                        qpretrieve.QLSInterferogram]:
        interfer_inst = InterferCls(
            data=edata["data"],
            fft_interface=qpretrieve.fourier.FFTFilterNumpy)
        assert interfer_inst.ff_iface.is_available
        assert issubclass(interfer_inst.ff_iface,
                          qpretrieve.fourier.base.FFTFilter)
        assert issubclass(interfer_inst.ff_iface,
                          qpretrieve.fourier.ff_numpy.FFTFilterNumpy)


def test_interfere_base_bad_interface():
    edata = np.load(data_path / "hologram_cell.npz")
    bad_name = "MyReallyCoolFFTInterface"

    with pytest.raises(
            qpretrieve.interfere.BadFFTFilterError,
            match=f"User-chosen FFT Interface '{bad_name}' is not available."):
        _ = qpretrieve.OffAxisHologram(
            data=edata["data"],
            fft_interface=bad_name)


def test_interfere_base_orig_array_layout():
    edata = np.load(data_path / "hologram_cell.npz")

    holo = qpretrieve.OffAxisHologram(data=edata["data"])
    assert holo.orig_array_layout is not None
    assert holo.orig_array_layout == "2d"


def test_interfere_base_get_data_with_input_layout():
    edata = np.load(data_path / "hologram_cell.npz")
    orig_shape = (200, 210)
    assert edata["data"].shape == orig_shape

    holo = qpretrieve.OffAxisHologram(data=edata["data"])
    assert holo.field.shape == (1, 200, 210)

    field_orig1 = holo.get_data_with_input_layout(data=holo.field)
    field_orig2 = holo.get_data_with_input_layout(data="field")

    assert field_orig1.shape == field_orig2.shape == orig_shape


def test_interfere_base_get_data_with_input_layout_fft_warning():
    edata = np.load(data_path / "hologram_cell.npz")
    orig_shape = (200, 210)
    assert edata["data"].shape == orig_shape

    holo = qpretrieve.OffAxisHologram(data=edata["data"])
    assert holo.field.shape == (1, 200, 210)

    fft_orig1 = holo.get_data_with_input_layout(data="fft_filtered")
    fft_orig2 = holo.get_data_with_input_layout(data="fft")
    assert fft_orig1.shape == fft_orig2.shape


def test_get_data_with_input_layout_2d(hologram):
    """The original data format should be returned correctly"""
    data_2d = hologram
    expected_output_shape = (1, data_2d.shape[-2], data_2d.shape[-1])

    # 2d data format
    oah = qpretrieve.OffAxisHologram(data_2d, padding=False,
                                     subtract_mean=False)
    res = oah.run_pipeline()
    assert res.shape == expected_output_shape

    data_attrs = [oah.field, oah.fft_origin, oah.fft_filtered,
                  oah.amplitude, oah.phase,
                  "field", "fft_origin", "fft_filtered",
                  "amplitude", "phase"]
    for data_attr in data_attrs:
        if not isinstance(data_attr, str):
            assert data_attr.shape == expected_output_shape
        # original shape was 2d
        orig_data = oah.get_data_with_input_layout(data_attr)
        assert orig_data.shape == data_2d.shape


def test_get_data_with_input_layout_rgb(hologram):
    """The original data format should be returned correctly"""
    data_rgb = np.stack([hologram, hologram, hologram], axis=-1)
    expected_output_shape = (1, hologram.shape[-2], hologram.shape[-1])

    # 2d data format
    oah = qpretrieve.OffAxisHologram(data_rgb, padding=False,
                                     subtract_mean=False)
    _ = oah.run_pipeline()

    data_attrs = [oah.field, oah.fft_origin, oah.fft_filtered,
                  oah.amplitude, oah.phase,
                  "field", "fft_origin", "fft_filtered",
                  "amplitude", "phase"]
    for data_attr in data_attrs:
        if not isinstance(data_attr, str):
            assert data_attr.shape == expected_output_shape
        # original shape was 2d
        assert oah.get_data_with_input_layout(
            data_attr).shape == data_rgb.shape


def test_get_data_with_input_layout_rgba(hologram):
    """The original data format should be returned correctly"""
    data_rgba = np.stack([hologram, hologram, hologram,
                          np.zeros_like(hologram)], axis=-1)
    expected_output_shape = (1, hologram.shape[-2], hologram.shape[-1])

    # 2d data format
    oah = qpretrieve.OffAxisHologram(data_rgba, padding=False,
                                     subtract_mean=False)
    _ = oah.run_pipeline()

    data_attrs = [oah.field, oah.fft_origin, oah.fft_filtered,
                  oah.amplitude, oah.phase,
                  "field", "fft_origin", "fft_filtered",
                  "amplitude", "phase"]
    for data_attr in data_attrs:
        if not isinstance(data_attr, str):
            assert data_attr.shape == expected_output_shape
        # original shape was 2d
        assert oah.get_data_with_input_layout(
            data_attr).shape == data_rgba.shape
