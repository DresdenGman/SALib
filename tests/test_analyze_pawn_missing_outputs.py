import numpy as np
import pytest

from SALib.analyze import pawn


def problem():
    return {"num_vars": 1, "names": ["x"], "bounds": [[0, 11]]}


def test_missing_outputs_do_not_invalidate_all_pawn_distances():
    X = np.arange(12.0)[:, None]
    Y = np.array([0, 1, np.nan, 3, 4, 5, 6, 7, np.nan, 9, 10, 11])
    original = Y.copy()

    result = pawn.analyze(problem(), X, Y, S=3)

    # The three conditional empirical CDFs have KS distances 0.7, 0.3, 0.7
    # from the unconditional CDF of the ten observed outputs.
    np.testing.assert_allclose(result["minimum"], [0.3])
    np.testing.assert_allclose(result["mean"], [17 / 30])
    np.testing.assert_allclose(result["median"], [0.7])
    np.testing.assert_allclose(result["maximum"], [0.7])
    np.testing.assert_allclose(result["stdev"], [np.std([0.7, 0.3, 0.7])])
    np.testing.assert_allclose(result["CV"], [np.std([0.7, 0.3, 0.7]) / (17 / 30)])
    np.testing.assert_array_equal(Y, original)


def test_conditioning_interval_with_only_missing_outputs_is_skipped():
    X = np.arange(12.0)[:, None]
    Y = np.arange(12.0)
    Y[:4] = np.nan

    result = pawn.analyze(problem(), X, Y, S=3)

    for name in ["minimum", "mean", "median", "maximum"]:
        np.testing.assert_allclose(result[name], [0.5])
    np.testing.assert_allclose(result["stdev"], [0])
    np.testing.assert_allclose(result["CV"], [0])


def test_all_missing_outputs_raise_an_actionable_error():
    X = np.arange(12.0)[:, None]
    Y = np.full(12, np.nan)

    with pytest.raises(ValueError, match="non-NaN output"):
        pawn.analyze(problem(), X, Y, S=3)
