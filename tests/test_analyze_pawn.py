import numpy as np
import pytest
from numpy.testing import assert_allclose

from SALib import ProblemSpec
from SALib.analyze import pawn
from SALib.test_functions import Ishigami
from SALib.util import read_param_file


def test_analyze_pawn():
    param_file = "src/SALib/test_functions/params/Ishigami_groups.txt"
    problem = read_param_file(param_file)
    sp = ProblemSpec(problem)
    sp.sample_sobol(512).evaluate(Ishigami.evaluate).analyze_pawn()


def test_pawn_retains_maximum_input_in_final_interval():
    problem = {"num_vars": 1, "names": ["x"], "bounds": [[0, 5]]}
    X = np.arange(6.0)[:, None]
    # Only the maximum input produces a different output. Omitting it from
    # every conditional distribution hides the tail's influence.
    Y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    result = pawn.analyze(problem, X, Y, S=3)

    # KS distances of the three two-row intervals are 1/6, 1/6 and 1/3.
    assert_allclose(result["minimum"], [1 / 6])
    assert_allclose(result["mean"], [2 / 9])
    assert_allclose(result["maximum"], [1 / 3])


@pytest.mark.parametrize("slides", [6, 14, 24, 49])
def test_pawn_quantile_grid_stays_within_unit_interval(slides):
    X = np.linspace(0.0, 1.0, 100)[:, None]
    problem = {"num_vars": 1, "names": ["x"], "bounds": [[0, 1]]}
    result = pawn.analyze(problem, X, X[:, 0], S=slides)

    assert np.isfinite(result["mean"]).all()


def test_pawn_single_interval_matches_unconditional_distribution():
    X = np.arange(6.0)[:, None]
    problem = {"num_vars": 1, "names": ["x"], "bounds": [[0, 5]]}
    result = pawn.analyze(problem, X, X[:, 0], S=1)

    for key in ["minimum", "mean", "median", "maximum", "stdev"]:
        assert_allclose(result[key], [0.0])
    assert np.isnan(result["CV"]).all()


def test_pawn_includes_repeated_maxima_without_double_counting():
    X = np.repeat(np.arange(4.0), 2)[:, None]
    problem = {"num_vars": 1, "names": ["x"], "bounds": [[0, 3]]}
    result = pawn.analyze(problem, X, X[:, 0], S=4)

    # The four distinct input values each condition on exactly two rows.
    assert_allclose(result["mean"], [0.625])
    assert_allclose(result["minimum"], [0.5])
    assert_allclose(result["maximum"], [0.75])
