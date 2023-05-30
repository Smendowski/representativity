from typing import Coroutine

import numpy as np
import pytest
from pytest_mock.plugin import MockerFixture

from data.extractors import NearestNeighborsBasedRepresentativenessExtractor
from data.models import Dataset
from data.processors import DatasetProcessor


@pytest.mark.asyncio
async def test_create_dataset() -> None:
    dataset = await DatasetProcessor.create_dataset(100, 100)
    features = dataset.get_feature_representation()
    targets = dataset.get_target_representation()

    assert len(dataset.samples) == 100
    assert features.shape == (100, 100)
    assert targets.shape == (100,)


@pytest.mark.asyncio
async def test_run_labeling(
        mocker: MockerFixture, dataset: Coroutine[None, None, Dataset]
) -> None:
    _dataset: Dataset = await dataset
    features: np.ndarray = _dataset.get_feature_representation()
    targets: np.ndarray = _dataset.get_target_representation()

    mocked_extractor = mocker.Mock(spec=NearestNeighborsBasedRepresentativenessExtractor)
    mocked_extractor.extract.return_value = np.ones(features.shape[0])

    assert np.array_equal(targets, np.array([None for _ in range(len(targets))]))

    supervised_dataset: Dataset = DatasetProcessor.run_labeling(_dataset, mocked_extractor)
    assert np.array_equal(supervised_dataset.get_feature_representation(), features)
    assert np.array_equal(supervised_dataset.get_target_representation(), np.ones(features.shape[0]))
    mocked_extractor.extract.assert_called_once()


@pytest.mark.asyncio
async def test_to_supervised(
        mocker: MockerFixture, dataset: Coroutine[None, None, Dataset]
) -> None:
    _dataset: Dataset = await dataset
    features: np.ndarray = _dataset.get_feature_representation()

    mocked_extractor = mocker.Mock(spec=NearestNeighborsBasedRepresentativenessExtractor)
    mocked_extractor.extract.return_value = np.ones(features.shape[0])
    labeled_chunks = await DatasetProcessor.to_supervised(_dataset, 3, mocked_extractor)

    assert len(labeled_chunks) == 3
    for chunk in labeled_chunks:
        assert isinstance(chunk, Dataset)
