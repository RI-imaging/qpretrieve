import numpy as np
import qpretrieve


def test_field_format_consistency_2d(hologram):
    """The original data format should be returned correctly"""
    data_2d = hologram
    expected_output_shape = (1, data_2d.shape[-2], data_2d.shape[-1])

    # 2d data format
    oah = qpretrieve.OffAxisHologram(data_2d, padding=False,
                                     subtract_mean=False)
    res = oah.run_pipeline()
    assert res.shape == expected_output_shape

    data_attrs = [oah.field, oah.fft_origin, oah.fft_filtered,
                  oah.amplitude, oah.phase]
    for data_attr in data_attrs:
        assert data_attr.shape == expected_output_shape
        # original shape was 2d
        assert oah.get_orig_data_fmt(data_attr).shape == data_2d.shape


def test_field_format_consistency_rgb(hologram):
    """The original data format should be returned correctly"""
    data_rgb = np.stack([hologram, hologram, hologram], axis=-1)
    expected_output_shape = (1, hologram.shape[-2], hologram.shape[-1])

    # 2d data format
    oah = qpretrieve.OffAxisHologram(data_rgb, padding=False,
                                     subtract_mean=False)
    _ = oah.run_pipeline()

    data_attrs = [oah.field, oah.fft_origin, oah.fft_filtered,
                  oah.amplitude, oah.phase]
    for data_attr in data_attrs:
        assert data_attr.shape == expected_output_shape
        # original shape was 2d
        assert oah.get_orig_data_fmt(data_attr).shape == data_rgb.shape


def test_field_format_consistency_rgba(hologram):
    """The original data format should be returned correctly"""
    data_rgba = np.stack([hologram, hologram, hologram,
                          np.zeros_like(hologram)], axis=-1)
    expected_output_shape = (1, hologram.shape[-2], hologram.shape[-1])

    # 2d data format
    oah = qpretrieve.OffAxisHologram(data_rgba, padding=False,
                                     subtract_mean=False)
    _ = oah.run_pipeline()

    data_attrs = [oah.field, oah.fft_origin, oah.fft_filtered,
                  oah.amplitude, oah.phase]
    for data_attr in data_attrs:
        assert data_attr.shape == expected_output_shape
        # original shape was 2d
        assert oah.get_orig_data_fmt(data_attr).shape == data_rgba.shape
