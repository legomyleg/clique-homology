import importlib
import sys

import numpy as np
import pytest


def test_p_values_import_has_no_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    sys.modules.pop("clique_homology.stats_engine.p_values", None)
    importlib.import_module("clique_homology.stats_engine.p_values")
    captured = capsys.readouterr()
    assert captured.out == ""


def test_get_mahalanobis_matches_manual_value() -> None:
    p_values = importlib.import_module("clique_homology.stats_engine.p_values")
    vector = np.array([2.0, 3.0])
    mean = np.array([1.0, 1.0])
    inv_cov = np.array([[1.0, 0.0], [0.0, 2.0]])

    observed = p_values.get_mahalanobis(vector, mean, inv_cov)
    assert observed == pytest.approx(9.0)


def test_calculate_p_vector_basic_properties() -> None:
    p_values = importlib.import_module("clique_homology.stats_engine.p_values")
    obs = np.array([1.0, 2.0])
    null = np.array([[1.0, 2.0], [1.5, 2.5], [0.5, 1.5], [1.2, 2.2]])

    p_val, d2_obs, d2_null = p_values.calculate_p_vector(obs, null)

    assert 0.0 <= p_val <= 1.0
    assert isinstance(d2_obs, float)
    assert isinstance(d2_null, np.ndarray)
    assert d2_null.shape == (4,)


def test_calculate_p_vector_handles_singular_covariance() -> None:
    p_values = importlib.import_module("clique_homology.stats_engine.p_values")
    obs = np.array([2.0, 2.0])
    null = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

    p_val, d2_obs, d2_null = p_values.calculate_p_vector(obs, null)

    assert 0.0 <= p_val <= 1.0
    assert d2_obs >= 0.0
    assert np.all(d2_null >= 0.0)


def test_calculate_p_vector_supports_1d_observations() -> None:
    p_values = importlib.import_module("clique_homology.stats_engine.p_values")
    obs = np.array([2.0])
    null = np.array([[1.0], [2.0], [3.0], [4.0]])

    p_val, d2_obs, d2_null = p_values.calculate_p_vector(obs, null)

    assert 0.0 <= p_val <= 1.0
    assert isinstance(d2_obs, float)
    assert d2_null.shape == (4,)


def test_calculate_p_vector_rejects_shape_mismatch() -> None:
    p_values = importlib.import_module("clique_homology.stats_engine.p_values")
    with pytest.raises(ValueError, match="shape"):
        p_values.calculate_p_vector(np.array([1.0, 2.0]), np.array([[1.0], [2.0]]))
