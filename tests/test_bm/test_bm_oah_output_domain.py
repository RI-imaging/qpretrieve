import pytest

pytest.importorskip("pytest_benchmark")

import qpretrieve


def _make_holo(hologram):
    return qpretrieve.OffAxisHologram(hologram)


def test_bm_oah_output_domain_spatial(benchmark, hologram):
    """Original legacy pipeline with iFFT"""
    field = benchmark(
        lambda: _make_holo(hologram).run_pipeline(output_domain="spatial"))

    assert field.shape == (1, hologram.shape[0], hologram.shape[1])


def test_bm_oah_output_domain_fourier(benchmark, hologram):
    """Pipeline without iFFT, output in Fourier domain"""
    result = benchmark(
        lambda: _make_holo(hologram).run_pipeline(output_domain="fourier"))

    assert result is not None


def test_bm_oah_output_domain_fourier_then_spatial(benchmark, hologram):
    """Pipeline initially without iFFT, then iFFT (for use with nrefocus)"""

    def run():
        holo = _make_holo(hologram)
        artifact = holo.run_pipeline(output_domain="fourier")
        return artifact.finalize()  # or holo.compute_field()

    field = benchmark(run)

    assert field.shape == (1, hologram.shape[0], hologram.shape[1])
