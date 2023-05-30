import random

import pytest
from data.models import Dataset, Sample
import numpy as np
from data.processors import DatasetProcessor


@pytest.fixture
def features() -> np.ndarray:
    yield np.array([
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        [3, 4, 5, 6, 7],
        [4, 5, 6, 7, 8],
        [5, 6, 7, 8, 9],
    ])


@pytest.fixture
async def dataset() -> Dataset:
    return await DatasetProcessor.create_dataset(100, 100)


@pytest.fixture()
def correct_shape_sample() -> Sample:
    return Sample(features=[random.random() for _ in range(100)])


@pytest.fixture()
def incorrect_shape_sample() -> Sample:
    return Sample(features=[random.random() for _ in range(50)])
