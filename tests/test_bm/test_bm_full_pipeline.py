import pytest

pytest.importorskip("pytest_benchmark")
pytest.importorskip("nrefocus")

import nrefocus
import qpretrieve

_PROPAGATION_KWARGS = dict(d=1.5, nm=1.533, res=8.25, method="fresnel", padding=False)


def _run_spatial(hologram):
    """Original legacy pipeline with all Fourier transforms"""
    holo = qpretrieve.OffAxisHologram(hologram, padding=False)
    field = holo.run_pipeline(output_domain="spatial")
    return nrefocus.refocus(field=field, **_PROPAGATION_KWARGS)


def _run_fourier(hologram):
    """Don't do the qpretrieve iFFT, nrefocus knows what to do with FFT"""
    holo = qpretrieve.OffAxisHologram(hologram, padding=False)
    artifact = holo.run_pipeline(output_domain="fourier")
    return nrefocus.refocus(field=artifact, **_PROPAGATION_KWARGS)


def _run_fourier_no_finalize(hologram):
    """Output the data in Fourier domain"""
    holo = qpretrieve.OffAxisHologram(hologram, padding=False)
    artifact = holo.run_pipeline(output_domain="fourier")
    return nrefocus.refocus(field=artifact, output_domain="fourier",
                            **_PROPAGATION_KWARGS)


def _run_fourier_then_finalize(hologram):
    """Output the data in Fourier domain, then do iFFT with qpretrieve"""
    holo = qpretrieve.OffAxisHologram(hologram, padding=False)
    artifact = holo.run_pipeline(output_domain="fourier")
    field_refoc = nrefocus.refocus(field=artifact, output_domain="fourier",
                                   **_PROPAGATION_KWARGS)
    return holo.compute_field(propagated_fft=field_refoc)


def test_bm_full_pipeline_spatial(benchmark, hologram):
    """Original legacy pipeline with all Fourier transforms"""
    result = benchmark(_run_spatial, hologram)
    assert result is not None


def test_bm_full_pipeline_fourier(benchmark, hologram):
    """Don't do the qpretrieve iFFT, nrefocus knows what to do with FFT"""
    result = benchmark(_run_fourier, hologram)
    assert result is not None


def test_bm_full_pipeline_fourier_no_finalize(benchmark, hologram):
    """Output the data in Fourier domain"""
    result = benchmark(_run_fourier_no_finalize, hologram)
    assert result is not None


def test_bm_full_pipeline_fourier_then_finalize(benchmark, hologram):
    """Output the data in Fourier domain, then do iFFT with qpretrieve"""
    result = benchmark(_run_fourier_then_finalize, hologram)
    assert result is not None
