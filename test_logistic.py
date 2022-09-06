from logistic import iterate_f, get_logistic_map_step
from conftest import random_state
import numpy as np
import pytest
from math import isclose

cases_s = [( 0.1, 2.2),
(0.2, 3.4),
(0.75, 1.7)]

cases = [(1, 0.1, 2.2),
(4, 0.2, 3.4),
(2, 0.75, 1.7)]

@pytest.mark.parametrize("x, r", cases_s)
def test_logistic_map_step(x, r):
    step_result = get_logistic_map_step(x, r)
    x_expected = r * x * (1 - x)
    assert isclose(step_result, x_expected)

@pytest.mark.parametrize("it, x, r", cases)
def test_iterate_f(it, x, r):
    it_results_expected = []
    it_results = iterate_f(it, x, r)
    x_it = x
    for step in range(it):
        x_it = r * x_it * (1 - x_it)
        it_results_expected.append(x_it)
    it_results_expected = np.array(it_results_expected)
    assert np.all(np.isclose(it_results_expected, it_results))

def test_convergence(random_state):
    x = random_state.rand()
    r = 1.5
    x_prev = x - 1
    while not isclose(x, x_prev):
        x_prev = x
        x = get_logistic_map_step(x, r)
    assert isclose(x, 1.0/3)

def test_chaotic(random_state):
    x = random_state.rand()
    r = 3.8
    n_it = 100000
    x_results = iterate_f(n_it, x, r)
    assert np.all(np.logical_and(x > 0.0, x < 1.0))
    assert len(x_results[-1000:]) == len(np.unique(x_results[-1000:]))

