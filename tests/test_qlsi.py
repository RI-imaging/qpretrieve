import pathlib

import h5py
import numpy as np
import qpretrieve


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
        assert qlsi.wavefront.argmax() == 241575
        assert np.allclose(qlsi.wavefront.max(), 6.270214993392721e-08,
                           atol=0, rtol=1e-12)

        assert qlsi.phase.argmax() == 241575
        assert np.allclose(qlsi.phase.max(), 0.7163076858062234,
                           atol=0, rtol=1e-12)
