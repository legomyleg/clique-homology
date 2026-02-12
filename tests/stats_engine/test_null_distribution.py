import numpy as np
import pytest

import clique_homology.stats_engine.null_distribution as null_distribution_module
from clique_homology.stats_engine.null_distribution import null_distribution


def test_null_distribution_returns_one_result_per_iteration(monkeypatch: pytest.MonkeyPatch) -> None:
    graph = object()
    coloring = ["red", "blue"]

    def fake_random_coloring(_colors: list[str], allowed_colors: list[str] | None = None) -> list[str]:
        return ["blue", "red"]

    def fake_betti_numbers(_graph: object, new_coloring: list[str], allowed_colors: list[str] | None = None) -> np.ndarray:
        assert new_coloring == ["blue", "red"]
        return np.array([1, 0], dtype=int)

    monkeypatch.setattr(null_distribution_module, "random_coloring", fake_random_coloring)
    monkeypatch.setattr(null_distribution_module, "betti_numbers", fake_betti_numbers)

    observed = null_distribution(graph, coloring, iterations=4)

    assert len(observed) == 4
    assert all(np.array_equal(value, np.array([1, 0])) for value in observed)


def test_null_distribution_zero_iterations_returns_empty_list() -> None:
    observed = null_distribution(object(), ["red"], iterations=0)
    assert observed == []


def test_null_distribution_rejects_negative_iterations() -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        null_distribution(object(), ["red"], iterations=-1)


def test_null_distribution_passes_allowed_palette_through(monkeypatch: pytest.MonkeyPatch) -> None:
    graph = object()
    calls: list[tuple[str, tuple[str, ...] | None]] = []

    def fake_random_coloring(colors: list[str], allowed_colors: list[str] | None = None) -> list[str]:
        calls.append(("random_coloring", tuple(allowed_colors) if allowed_colors is not None else None))
        return colors

    def fake_betti_numbers(_graph: object, colors: list[str], allowed_colors: list[str] | None = None) -> np.ndarray:
        calls.append(("betti_numbers", tuple(allowed_colors) if allowed_colors is not None else None))
        return np.array([len(colors)], dtype=int)

    monkeypatch.setattr(null_distribution_module, "random_coloring", fake_random_coloring)
    monkeypatch.setattr(null_distribution_module, "betti_numbers", fake_betti_numbers)

    palette = ["red", "blue"]
    null_distribution(graph, ["red", "blue"], iterations=2, allowed_colors=palette)

    assert calls == [
        ("random_coloring", ("red", "blue")),
        ("betti_numbers", ("red", "blue")),
        ("random_coloring", ("red", "blue")),
        ("betti_numbers", ("red", "blue")),
    ]


def test_null_distribution_propagates_downstream_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_random_coloring(_colors: list[str], allowed_colors: list[str] | None = None) -> list[str]:
        return ["red"]

    def fake_betti_numbers(_graph: object, _colors: list[str], allowed_colors: list[str] | None = None) -> np.ndarray:
        raise RuntimeError("boom")

    monkeypatch.setattr(null_distribution_module, "random_coloring", fake_random_coloring)
    monkeypatch.setattr(null_distribution_module, "betti_numbers", fake_betti_numbers)

    with pytest.raises(RuntimeError, match="boom"):
        null_distribution(object(), ["red"], iterations=1)
