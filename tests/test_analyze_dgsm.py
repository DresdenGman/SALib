import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import norm

from SALib.analyze import dgsm
from SALib.sample import finite_diff
from SALib.test_functions import Ishigami


@pytest.mark.parametrize("scale", [0.1, -10.0, 10.0])
def test_dgsm_confidence_is_invariant_to_output_units(scale):
    problem = {
        "num_vars": 3,
        "names": ["x1", "x2", "x3"],
        "bounds": [[-np.pi, np.pi]] * 3,
    }
    X = finite_diff.sample(problem, 256, seed=123)
    Y = Ishigami.evaluate(X)

    original = dgsm.analyze(problem, X, Y, seed=123)
    scaled = dgsm.analyze(problem, X, scale * Y, seed=123)

    assert_allclose(scaled["dgsm"], original["dgsm"])
    assert_allclose(scaled["dgsm_conf"], original["dgsm_conf"])
    assert_allclose(scaled["vi"], scale**2 * original["vi"])


def test_linear_derivative_still_has_variance_uncertainty(monkeypatch):
    # The squared derivative is exactly one in every resample. The normalized
    # measure still varies because the output variance is estimated from data.
    base = np.array([0.0, 1.0, 3.0, 6.0])
    draws = np.array([[0, 0, 1, 2], [0, 1, 2, 3]])
    monkeypatch.setattr(np.random, "randint", lambda *args, **kwargs: draws)

    estimate, confidence = dgsm.calc_dgsm(
        base, base + 1.0, np.ones(4), [0.0, 1.0], 2, 0.95
    )

    # These two samples have population variances 1.5 and 5.25, respectively.
    expected_draws = np.array([1.0 / 1.5, 1.0 / 5.25]) / np.pi**2
    assert_allclose(estimate, 1.0 / (5.25 * np.pi**2))
    assert_allclose(confidence, norm.ppf(0.975) * expected_draws.std(ddof=1))
    assert confidence > 0
