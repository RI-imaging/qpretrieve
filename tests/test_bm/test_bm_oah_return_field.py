import pytest

pytest.importorskip("pytest_benchmark")

import qpretrieve


def _make_holo(hologram):
    return qpretrieve.OffAxisHologram(hologram)


def test_bm_oah_return_field_true(benchmark, hologram):
    field = benchmark(
        lambda: _make_holo(hologram).run_pipeline(return_field=True))

    assert field.shape == (1, hologram.shape[0], hologram.shape[1])


def test_bm_oah_return_field_false(benchmark, hologram):
    result = benchmark(
        lambda: _make_holo(hologram).run_pipeline(return_field=False))

    assert result is None


def test_bm_oah_return_field_false_then_true(benchmark, hologram):
    def run():
        holo = _make_holo(hologram)
        holo.run_pipeline(return_field=False)
        return holo.compute_field()

    field = benchmark(run)

    assert field.shape == (1, hologram.shape[0], hologram.shape[1])
