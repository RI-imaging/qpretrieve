import pathlib
import numpy as np

import qpretrieve

data_path = pathlib.Path(__file__).parent / "data"


def test_interfere_base_best_interface():
    edata = np.load(data_path / "hologram_cell.npz")

    holo = qpretrieve.OffAxisHologram(data=edata["data"])
    assert holo.ff_iface.is_available
    assert issubclass(holo.ff_iface, qpretrieve.fourier.base.FFTFilter)
    assert issubclass(holo.ff_iface, qpretrieve.fourier.ff_numpy.FFTFilterNumpy)
