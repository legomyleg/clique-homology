import pytest

import clique_homology.stats_engine.random_coloring as random_coloring_module
from clique_homology.stats_engine.random_coloring import random_coloring


def test_random_coloring_rejects_non_string_inputs() -> None:
    with pytest.raises(TypeError, match="must be strings"):
        random_coloring(["red", 2, "blue"])  # type: ignore[list-item]


def test_random_coloring_rejects_non_string_allowed_palette() -> None:
    with pytest.raises(TypeError, match="must be strings"):
        random_coloring(["red", "blue"], allowed_colors=["red", 1])  # type: ignore[list-item]


def test_random_coloring_rejects_empty_palette_for_nonempty_input() -> None:
    with pytest.raises(ValueError, match="palette is empty"):
        random_coloring(["red"], allowed_colors=[])


def test_random_coloring_preserves_length() -> None:
    observed = random_coloring(["red", "red", "blue", "green"])
    assert len(observed) == 4


def test_random_coloring_with_allowed_palette_is_subset() -> None:
    observed = random_coloring(["red", "blue", "green"], allowed_colors=["red", "blue"])
    assert set(observed).issubset({"red", "blue"})


def test_random_coloring_empty_input_returns_empty() -> None:
    assert random_coloring([]) == []


def test_random_coloring_non_proportional_uses_unique_colors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_options: list[list[str]] = []

    def fake_choice(options: list[str]) -> str:
        seen_options.append(list(options))
        return options[0]

    monkeypatch.setattr(random_coloring_module, "choice", fake_choice)
    random_coloring(["red", "red", "blue"], proportional=False)

    assert len(seen_options) == 3
    for options in seen_options:
        assert set(options) == {"red", "blue"}
        assert len(options) == 2


def test_random_coloring_proportional_uses_original_multiset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_options: list[list[str]] = []

    def fake_choice(options: list[str]) -> str:
        seen_options.append(list(options))
        return options[0]

    monkeypatch.setattr(random_coloring_module, "choice", fake_choice)
    random_coloring(["red", "red", "blue"], proportional=True)

    assert len(seen_options) == 3
    for options in seen_options:
        assert options == ["red", "red", "blue"]
