import random

import pytest
from pydantic.error_wrappers import ValidationError

from data.models import Sample


@pytest.fixture
def sample() -> Sample:
    return Sample(features=[random.random() for _ in range(100)])


def test_sample_validation_with_heterogeneous_features() -> None:
    with pytest.raises(ValidationError):
        Sample(features=[1, "value"])


def test_sample_validation_with_homogenous_features(sample: Sample) -> None:
    assert len(sample.features) == 100
    assert sample.representativeness is None


def test_representativeness_to_sample_assignment(sample: Sample) -> None:
    with pytest.raises(ValidationError):
        sample.representativeness = "value"
    assert sample.representativeness is None
    sample.representativeness = random.random()
    assert sample.representativeness is not None
