from typing import Coroutine

import numpy as np
import pytest

from data.models import Dataset, Sample
from exceptions import (
    EnsembleModelFitWithoutComponentRegressorsRegisteredError,
    InferenceSampleHasUnexpectedShapeError
)

from ml.helpers import ExperimentTracker, TrainingStatus
from ml.models import (
    EnsembleRandomForestBasedRegressor,
    RandomForestBasedRegressor,
    Regressor
)


@pytest.mark.parametrize("number_of_estimators", [0, 10])
def test_register_deregister_regressors(number_of_estimators: int) -> None:
    ensemble_regressor = EnsembleRandomForestBasedRegressor()
    for _ in range(number_of_estimators):
        ensemble_regressor.register_regressor(RandomForestBasedRegressor())

    assert len(ensemble_regressor.get_regressors()) == number_of_estimators
    ensemble_regressor.deregister_regressors()
    assert len(ensemble_regressor.get_regressors()) == 0


@pytest.mark.asyncio
async def test_fit_random_forest_based_regressor(
        dataset: Coroutine[None, None, Dataset]
) -> None:
    _dataset = await dataset
    regressor: Regressor = RandomForestBasedRegressor()
    regressor.fit(_dataset)

    assert regressor.model is not None


@pytest.mark.asyncio
async def test_fit_ensemble_random_forest_based_regressor(
        dataset: Coroutine[None, None, Dataset]
) -> None:
    _dataset = await dataset
    regressor: EnsembleRandomForestBasedRegressor = EnsembleRandomForestBasedRegressor()

    assert regressor.status == TrainingStatus.NOT_STARTED
    with pytest.raises(EnsembleModelFitWithoutComponentRegressorsRegisteredError):
        await regressor.fit(_dataset)
    assert regressor.status == TrainingStatus.ERROR
    regressor.reset_status()


@pytest.mark.asyncio
async def test_regressor_statuses_with_reset_status():
    regressor: Regressor = RandomForestBasedRegressor()
    assert regressor.status == TrainingStatus.NOT_STARTED

    ExperimentTracker.handle_training_started(regressor)
    assert regressor.status == TrainingStatus.DURING_TRAINING
    verbose_status = regressor.get_verbose_status()
    assert "status" in verbose_status
    assert verbose_status["status"] == TrainingStatus.DURING_TRAINING.value

    ExperimentTracker.handle_training_failed(regressor)
    assert regressor.status == TrainingStatus.ERROR
    verbose_status = regressor.get_verbose_status()
    assert "status" in verbose_status
    assert verbose_status["status"] == TrainingStatus.ERROR.value

    ExperimentTracker.handle_training_finished(regressor)
    assert regressor.status == TrainingStatus.FINISHED
    verbose_status = regressor.get_verbose_status()
    assert "status" in verbose_status
    assert verbose_status["status"] == TrainingStatus.FINISHED.value

    regressor.status = TrainingStatus.AWAIT
    regressor.reset_status()
    assert regressor.status == TrainingStatus.NOT_STARTED


@pytest.mark.asyncio
async def test_ensemble_random_forest_based_regressor_status_consistency():
    regressor = RandomForestBasedRegressor()
    ensemble_regressor = EnsembleRandomForestBasedRegressor()
    ensemble_regressor.register_regressor(regressor)

    ensemble_regressor.status = TrainingStatus.ERROR
    assert regressor.status == TrainingStatus.ERROR

    ensemble_regressor.reset_status()
    assert regressor.status == TrainingStatus.NOT_STARTED

    ensemble_regressor.deregister_regressors()


@pytest.mark.asyncio
async def test_predict_random_forest_based_regressor_on_unlabeled_dataset_with_correct_sample(
        dataset: Coroutine[None, None, Dataset], correct_shape_sample: Sample
) -> None:
    _dataset = await dataset

    regressor: Regressor = RandomForestBasedRegressor()
    regressor.fit(_dataset)
    prediction = regressor.predict(correct_shape_sample)
    assert np.isnan(prediction)


@pytest.mark.asyncio
async def test_predict_random_forest_based_regressor_on_unlabeled_dataset_with_incorrect_sample(
        dataset: Coroutine[None, None, Dataset], incorrect_shape_sample: Sample
) -> None:
    _dataset = await dataset

    regressor: Regressor = RandomForestBasedRegressor()
    regressor.fit(_dataset)

    with pytest.raises(InferenceSampleHasUnexpectedShapeError):
        prediction = regressor.predict(incorrect_shape_sample)
        assert np.isnan(prediction)


@pytest.mark.asyncio
async def test_predict_random_forest_based_regressor_on_labeled_dataset_with_correct_sample(
        dataset: Coroutine[None, None, Dataset], correct_shape_sample: Sample
) -> None:
    _dataset = await dataset
    for sample in _dataset.samples:
        sample.representativeness = 1.0

    regressor: Regressor = RandomForestBasedRegressor()
    regressor.fit(_dataset)
    prediction = regressor.predict(correct_shape_sample)
    assert prediction is not None and not np.isnan(prediction)


@pytest.mark.asyncio
async def test_predict_random_forest_based_regressor_on_labeled_dataset_with_incorrect_sample(
        dataset: Coroutine[None, None, Dataset], incorrect_shape_sample: Sample
) -> None:
    _dataset = await dataset
    for sample in _dataset.samples:
        sample.representativeness = 1.0

    regressor: Regressor = RandomForestBasedRegressor()
    regressor.fit(_dataset)

    with pytest.raises(InferenceSampleHasUnexpectedShapeError):
        prediction = regressor.predict(incorrect_shape_sample)
        assert np.isnan(prediction)


@pytest.mark.asyncio
async def test_predict_ensemble_random_forest_based_regressor_on_unlabeled_dataset_with_correct_sample(
        dataset: Coroutine[None, None, Dataset], correct_shape_sample: Sample
) -> None:
    _dataset = await dataset
    ensemble_regressor = EnsembleRandomForestBasedRegressor()
    ensemble_regressor.register_regressor(RandomForestBasedRegressor())

    await ensemble_regressor.fit([_dataset])
    prediction = await ensemble_regressor.predict(correct_shape_sample)
    assert np.isnan(prediction)

    ensemble_regressor.deregister_regressors()


@pytest.mark.asyncio
async def test_predict_ensemble_random_forest_based_regressor_on_unlabeled_dataset_with_incorrect_sample(
        dataset: Coroutine[None, None, Dataset], incorrect_shape_sample: Sample
) -> None:
    _dataset = await dataset
    ensemble_regressor = EnsembleRandomForestBasedRegressor()
    ensemble_regressor.register_regressor(RandomForestBasedRegressor())
    await ensemble_regressor.fit([_dataset])

    with pytest.raises(InferenceSampleHasUnexpectedShapeError):
        prediction = ensemble_regressor.predict(incorrect_shape_sample)
        assert np.isnan(prediction)

    ensemble_regressor.deregister_regressors()


@pytest.mark.asyncio
async def test_predict_ensemble_random_forest_based_regressor_on_labeled_dataset_with_correct_sample(
        dataset: Coroutine[None, None, Dataset], correct_shape_sample: Sample
) -> None:
    _dataset = await dataset
    ensemble_regressor = EnsembleRandomForestBasedRegressor()
    ensemble_regressor.register_regressor(RandomForestBasedRegressor())

    for sample in _dataset.samples:
        sample.representativeness = 1.0

    await ensemble_regressor.fit([_dataset])
    prediction = await ensemble_regressor.predict(correct_shape_sample)
    assert prediction is not None and not np.isnan(prediction)

    ensemble_regressor.deregister_regressors()


@pytest.mark.asyncio
async def test_predict_ensemble_random_forest_based_regressor_on_labeled_dataset_with_incorrect_sample(
        dataset: Coroutine[None, None, Dataset], incorrect_shape_sample: Sample
) -> None:
    _dataset = await dataset
    ensemble_regressor = EnsembleRandomForestBasedRegressor()
    ensemble_regressor.register_regressor(RandomForestBasedRegressor())

    for sample in _dataset.samples:
        sample.representativeness = 1.0

    await ensemble_regressor.fit([_dataset])
    with pytest.raises(InferenceSampleHasUnexpectedShapeError):
        prediction = ensemble_regressor.predict(incorrect_shape_sample)
        assert np.isnan(prediction)

    ensemble_regressor.deregister_regressors()
