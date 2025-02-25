import pathlib

import h5py
import numpy as np
from skimage.restoration import unwrap_phase

import qpretrieve
from qpretrieve.data_array_layout import (
    convert_data_to_3d_array_layout
)

data_path = pathlib.Path(__file__).parent / "data"


def test_qlsi_phase():
    with h5py.File(data_path / "qlsi_paa_bead.h5") as h5:
        qlsi = qpretrieve.QLSInterferogram(
            data=h5["0"][:],
            reference=h5["reference"][:],
            filter_name="tukey",
            filter_size=180,
            filter_size_interpretation="frequency index",
            wavelength=h5["0"].attrs["wavelength"],
            qlsi_pitch_term=h5["0"].attrs["qlsi_pitch_term"],
        )
        qlsi.run_pipeline()
        assert qlsi.pipeline_kws["wavelength"] == 550e-9
        assert qlsi.pipeline_kws["qlsi_pitch_term"] == 1.87711e-08
        assert qlsi.wavefront.argmax() == 242294
        assert np.allclose(qlsi.wavefront.max(), 8.179288852406586e-08,
                           atol=0, rtol=1e-12)
        assert qlsi.phase.argmax() == 242294
        assert np.allclose(qlsi.phase.max(), 0.9343997734657971,
                           atol=0, rtol=1e-12)


def test_qlsi_phase_3d():
    with h5py.File(data_path / "qlsi_paa_bead.h5") as h5:
        data_3d, _ = convert_data_to_3d_array_layout(h5["0"][:])
        data_3d = np.repeat(data_3d, repeats=10, axis=0)
        assert data_3d.shape == (10, 720, 720)
        qlsi = qpretrieve.QLSInterferogram(
            data=data_3d,
            reference=h5["reference"][:],
            filter_name="tukey",
            filter_size=180,
            filter_size_interpretation="frequency index",
            wavelength=h5["0"].attrs["wavelength"],
            qlsi_pitch_term=h5["0"].attrs["qlsi_pitch_term"],
        )
        qlsi.run_pipeline()
        assert qlsi.pipeline_kws["wavelength"] == 550e-9
        assert qlsi.pipeline_kws["qlsi_pitch_term"] == 1.87711e-08
        for wavefront in qlsi.wavefront:
            assert wavefront.argmax() == 242294
            assert np.allclose(wavefront.max(), 8.179288852406586e-08,
                               atol=0, rtol=1e-12)

        for phase in qlsi.phase:
            assert phase.argmax() == 242294
            assert np.allclose(qlsi.phase.max(), 0.9343997734657971,
                               atol=0, rtol=1e-12)


def test_qlsi_fftfreq_reshape_2d_3d(hologram):
    data_2d = hologram
    data_3d, _ = qpretrieve.data_array_layout._convert_2d_to_3d(data_2d)

    fx_2d = np.fft.fftfreq(data_2d.shape[-1]).reshape(-1, 1)
    fx_3d = np.fft.fftfreq(data_3d.shape[-1]).reshape(data_3d.shape[0], -1, 1)

    fy_2d = np.fft.fftfreq(data_2d.shape[-2]).reshape(1, -1)
    fy_3d = np.fft.fftfreq(data_3d.shape[-2]).reshape(data_3d.shape[0], 1, -1)

    assert np.array_equal(fx_2d, fx_3d[0])
    assert np.array_equal(fy_2d, fy_3d[0])


def test_qlsi_unwrap_phase_2d_3d():
    """
    Check whether `skimage.restoration.unwrap_phase` unwraps 2d
    images along the z axis when given a 3d array input.
    Answer is no. `unwrap_phase` is designed for to unwrap data
    on all axes at once.
    """
    with h5py.File(data_path / "qlsi_paa_bead.h5") as h5:
        image = h5["0"][:]

    # Standard analysis pipeline
    pipeline_kws = {
        'wavelength': 550e-9,
        'qlsi_pitch_term': 1.87711e-08,
        'filter_name': "disk",
        'filter_size': 180,
        'filter_size_interpretation': "frequency index",
        'scale_to_filter': False,
        'invert_phase': False
    }

    data_2d = image
    data_3d, _ = qpretrieve.data_array_layout._convert_2d_to_3d(data_2d)

    ft_2d = qpretrieve.fourier.FFTFilterNumpy(data_2d, subtract_mean=False)
    ft_3d = qpretrieve.fourier.FFTFilterNumpy(data_3d, subtract_mean=False)

    pipeline_kws["sideband_freq"] = qpretrieve.interfere. \
        if_qlsi.find_peaks_qlsi(ft_2d.fft_origin[0])

    hx_2d = ft_2d.filter(filter_name=pipeline_kws["filter_name"],
                         filter_size=pipeline_kws["filter_size"],
                         scale_to_filter=pipeline_kws["scale_to_filter"],
                         freq_pos=pipeline_kws["sideband_freq"])
    hx_3d = ft_3d.filter(filter_name=pipeline_kws["filter_name"],
                         filter_size=pipeline_kws["filter_size"],
                         scale_to_filter=pipeline_kws["scale_to_filter"],
                         freq_pos=pipeline_kws["sideband_freq"])

    assert np.array_equal(hx_2d, hx_3d)

    px_2d = unwrap_phase(np.angle(hx_2d[0]))

    px_3d_loop = np.zeros_like(hx_3d)
    for i, _hx in enumerate(hx_3d):
        px_3d_loop[i] = unwrap_phase(np.angle(_hx))

    assert np.array_equal(px_2d, px_3d_loop[0])  # this passes

    px_3d = unwrap_phase(np.angle(hx_3d))  # this is not equivalent
    assert not np.array_equal(px_2d, px_3d[0])


def test_qlsi_rotate_2d_3d(hologram):
    """
    Ensure the old 2d and new 3d rotation is identical.
    Note that the hologram is used only as an example input image,
    and it is not the correct data type for QLSI.
    """
    data_2d = hologram
    data_3d, _ = qpretrieve.data_array_layout._convert_2d_to_3d(data_2d)

    rot_2d = qpretrieve.interfere.if_qlsi.rotate_noreshape(
        data_2d,
        angle=2,
        axes=(1, 0),  # this was the default used before
        reshape=False,
    )
    rot_3d = qpretrieve.interfere.if_qlsi.rotate_noreshape(
        data_3d,
        angle=2,
        axes=(-1, -2),  # the y and x axes
        reshape=False,
    )
    rot_3d_2 = qpretrieve.interfere.if_qlsi.rotate_noreshape(
        data_3d,
        angle=2,
        axes=(-2, -1),  # the y and x axes
        reshape=False,
    )

    assert rot_2d.dtype == rot_3d.dtype
    assert np.array_equal(rot_2d, rot_3d[0])
    assert np.array_equal(rot_2d, rot_3d_2[0])


def test_qlsi_pad_2d_3d(hologram):
    """
    Ensure the old 2d and new 3d padding is identical.
    Note that the hologram is used only as an example input image,
    and it is not the correct data type for QLSI.
    """
    data_2d = hologram
    data_3d, _ = qpretrieve.data_array_layout._convert_2d_to_3d(data_2d)

    sx, sy = data_2d.shape[-2:]
    gradpad_2d = np.pad(
        data_2d, ((sx // 2, sx // 2), (sy // 2, sy // 2)),
        mode="constant", constant_values=0)
    gradpad_3d = np.pad(
        data_3d, ((0, 0), (sx // 2, sx // 2), (sy // 2, sy // 2)),
        mode="constant", constant_values=0)

    assert gradpad_2d.dtype == gradpad_3d.dtype
    assert np.array_equal(gradpad_2d, gradpad_3d[0])


def test_fxy_complex_mul(hologram):
    """
    Ensure the old 2d and new 3d complex multiplication is identical.
    Note that the hologram is used only as an example input image,
    and it is not the correct data type for QLSI.
    """
    data_2d = hologram
    data_3d, _ = qpretrieve.data_array_layout._convert_2d_to_3d(data_2d)

    assert np.array_equal(data_2d, data_3d[0])

    # 2d
    fx_2d = np.fft.fftfreq(data_2d.shape[0]).reshape(-1, 1)
    fy_2d = np.fft.fftfreq(data_2d.shape[1]).reshape(1, -1)
    fxy_2d = -2 * np.pi * 1j * (fx_2d + 1j * fy_2d)
    assert fxy_2d.shape == (64, 64)
    fxy_2d[0, 0] = 1

    # 3d
    fx_3d = np.fft.fftfreq(data_3d.shape[-2]).reshape(-1, 1)
    fy_3d = np.fft.fftfreq(data_3d.shape[-1]).reshape(1, -1)
    fxy_3d = -2 * np.pi * 1j * (fx_3d + 1j * fy_3d)
    fxy_3d = np.repeat(fxy_3d[np.newaxis, :, :],
                       repeats=data_3d.shape[0], axis=0)
    assert fxy_3d.shape == (1, 64, 64)
    fxy_3d[:, 0, 0] = 1

    assert np.array_equal(fx_2d, fx_3d)
    assert np.array_equal(fxy_2d, fxy_3d[0])
