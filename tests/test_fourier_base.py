import copy
import pathlib

import h5py
import numpy as np
from skimage.restoration import unwrap_phase
from scipy import signal

from qpretrieve import fourier, interfere

data_path = pathlib.Path(__file__).parent / "data"


def test_scale_sanity_check():
    """Scale an image and compare with skimage"""
    # create a 2D gaussian test image
    x = np.linspace(-100, 100, 100)
    xx, yy = np.meshgrid(x, -x, indexing="ij")
    gauss = np.exp(-(xx**2 + yy**2) / 625)

    ft = fourier.FFTFilterNumpy(gauss, subtract_mean=False)

    # That's basically a sanity check. The IFT(FT(gauss)) == gauss
    if1 = ft.filter(filter_name="disk",
                    filter_size=1,
                    freq_pos=(0, 0),
                    scale_to_filter=False)
    assert np.allclose(gauss, if1, rtol=0, atol=1e-15)

    # Now we reduce the filter radius to 0.25. This should still yield
    # good results, because the gaussian is small. If you look at the
    # difference image, you will see rings now.
    if2 = ft.filter(filter_name="disk",
                    filter_size=0.25,
                    freq_pos=(0, 0),
                    scale_to_filter=False)
    assert np.allclose(gauss, if2, rtol=0, atol=4e-4)

    # Now scale to the filter size:
    if3 = ft.filter(filter_name="disk",
                    filter_size=0.25,
                    freq_pos=(0, 0),
                    scale_to_filter=True)
    # (and scale the input image as well)
    gauss3 = signal.resample(signal.resample(gauss, 50, axis=0),
                             50, axis=1)
    assert np.allclose(gauss3, if3, rtol=0, atol=1e-7)


def test_scale_sanity_check_other_filter():
    """Scale an image and compare with skimage"""
    # create a 2D gaussian test image
    x = np.linspace(-100, 100, 100)
    xx, yy = np.meshgrid(x, -x, indexing="ij")
    gauss = np.exp(-(xx ** 2 + yy ** 2) / 625)

    ft = fourier.FFTFilterNumpy(gauss, subtract_mean=False)

    # That's basically a sanity check. The IFT(FT(gauss)) == gauss
    if1 = ft.filter(filter_name="disk",
                    filter_size=1,
                    freq_pos=(0, 0),
                    scale_to_filter=False)
    assert np.allclose(gauss, if1, rtol=0, atol=1e-15)

    # Now we reduce the filter radius to 0.25. This should still yield
    # good results, because the gaussian is small. If you look at the
    # difference image, you will see rings now.
    if2 = ft.filter(filter_name="disk",
                    filter_size=0.125,
                    freq_pos=(0, 0),
                    scale_to_filter=False)
    assert np.allclose(gauss, if2, rtol=0, atol=4e-4)

    # Now scale to the filter size:
    if3 = ft.filter(filter_name="disk",
                    filter_size=0.125,
                    freq_pos=(0, 0),
                    scale_to_filter=True)
    # (and scale the input image as well)
    gauss3 = signal.resample(
        signal.resample(gauss, 25, axis=0),
        25, axis=1)
    assert np.allclose(gauss3, if3, rtol=0, atol=1e-7)


def test_scale_to_filter_oah():
    data = np.load(data_path / "hologram_cell.npz")
    image = data["data"]
    refer = data["bg_data"]

    # Standard analysis pipeline
    pipeline_kws = {
        'filter_name': 'disk',
        'filter_size': 0.3333333333333333,
        'filter_size_interpretation': 'sideband distance',
        'scale_to_filter': False,
        'sideband_freq': (-0.203125, -0.12109375),
        'invert_phase': False
    }

    ifh = interfere.OffAxisHologram(image, **pipeline_kws)
    ifh.run_pipeline()

    ifr = interfere.OffAxisHologram(refer, **pipeline_kws)
    ifr.run_pipeline()

    phase = unwrap_phase(ifh.phase - ifr.phase)
    assert phase.shape == (200, 210)
    assert np.allclose(phase.mean(), 1.0840394954441188, atol=1e-5)

    # Rescaled pipeline
    pipeline_kws_scale = copy.copy(pipeline_kws)
    pipeline_kws_scale["scale_to_filter"] = True
    ifh.run_pipeline(**pipeline_kws_scale)
    ifr.run_pipeline(**pipeline_kws_scale)
    phase_scaled = unwrap_phase(ifh.phase - ifr.phase)
    assert phase_scaled.shape == (33, 34)
    assert np.allclose(phase_scaled.mean(), 1.0469570087033453, atol=1e-5)


def test_scale_to_filter_qlsi():
    with h5py.File(data_path / "qlsi_paa_bead.h5") as h5:
        image = h5["0"][:]
        refer = h5["reference"][:]

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

    ifh = interfere.QLSInterferogram(image, **pipeline_kws)
    ifh.run_pipeline()

    ifr = interfere.QLSInterferogram(refer, **pipeline_kws)
    ifr.run_pipeline()

    phase = unwrap_phase(ifh.phase - ifr.phase)
    assert phase.shape == (720, 720)
    assert np.allclose(phase.mean(), 0.12434563269684816, atol=1e-6)

    # Rescaled pipeline
    pipeline_kws_scale = copy.copy(pipeline_kws)
    pipeline_kws_scale["scale_to_filter"] = True
    ifh.run_pipeline(**pipeline_kws_scale)
    ifr.run_pipeline(**pipeline_kws_scale)
    phase_scaled = unwrap_phase(ifh.phase - ifr.phase)

    assert phase_scaled.shape == (126, 126)

    assert np.allclose(phase_scaled.mean(), 0.1257080793074251, atol=1e-6)
