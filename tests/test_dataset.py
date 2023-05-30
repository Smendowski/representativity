import random

import pytest

from data.models import Dataset, Sample
from pydantic.error_wrappers import ValidationError


def test_dataset_creation_with_unequal_sample_lengths() -> None:
    first_sample: Sample = Sample(features=[random.random() for _ in range(10)])
    second_sample: Sample = Sample(features=[random.random() for _ in range(20)])
    with pytest.raises(ValidationError):
        Dataset(samples=[first_sample, second_sample])


def test_dataset_creation_with_equal_sample_lengths() -> None:
    first_sample: Sample = Sample(features=[random.random() for _ in range(10)])
    second_sample: Sample = Sample(features=[random.random() for _ in range(10)])
    dataset: Dataset = Dataset(samples=[first_sample, second_sample])
    assert len(dataset.samples) == 2
    assert len(dataset.get_feature_representation()) == 2
    assert len(dataset.get_target_representation()) == 2
