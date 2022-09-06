import numpy as np
import pytest

SEED = np.arange(0, 1e6, dtype=int)

@pytest.fixture
def random_state():
    random_state = np.random.RandomState(SEED)
    return random_state

