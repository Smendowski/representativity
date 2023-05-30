from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from exceptions import (
    EnsembleModelFitWithoutComponentRegressorsRegisteredError
)

from ml.helpers import ExperimentTracker, TrainingStatus
from ml.models import (
    EnsembleRandomForestBasedRegressor,
    RandomForestBasedRegressor,
    Regressor
)


@contextmanager
def mock_current_datetime_representation():
    with patch("ml.helpers.ExperimentTracker.get_current_datetime_representation") as mocked_datetime:
        mocked_datetime.return_value = "2023-05-30 10:00:00"
        yield mocked_datetime


@pytest.fixture
def regressor():
    return RandomForestBasedRegressor()


def test_experiment_tracker_handle_training_started(
        regressor: Regressor
) -> None:
    with mock_current_datetime_representation() as mocked_datetime:
        ExperimentTracker.handle_training_started(regressor)
        assert regressor.status == TrainingStatus.DURING_TRAINING
        assert regressor.start_training_time == mocked_datetime.return_value
        assert regressor.stop_training_time is None
        assert regressor.error_training_time is None


def test_experiment_tracker_handle_training_finished(
        regressor: Regressor
) -> None:
    with mock_current_datetime_representation() as mocked_datetime:
        ExperimentTracker.handle_training_started(regressor)
        ExperimentTracker.handle_training_finished(regressor)
        assert regressor.status == TrainingStatus.FINISHED
        assert regressor.start_training_time == mocked_datetime.return_value
        assert regressor.error_training_time is None
        assert regressor.stop_training_time == mocked_datetime.return_value


def test_experiment_tracker_handle_training_failed(
        regressor: Regressor
) -> None:
    with mock_current_datetime_representation() as mocked_datetime:
        ExperimentTracker.handle_training_started(regressor)
        ExperimentTracker.handle_training_failed(regressor)
        assert regressor.status == TrainingStatus.ERROR
        assert regressor.start_training_time == mocked_datetime.return_value
        assert regressor.error_training_time == mocked_datetime.return_value
        assert regressor.stop_training_time is None


def test_experiment_tracker_from_started_to_reset_transition(
        regressor: Regressor
) -> None:
    ExperimentTracker.handle_training_started(regressor)
    regressor.reset_status()
    assert regressor.status == TrainingStatus.NOT_STARTED
    assert regressor.start_training_time is None
    assert regressor.stop_training_time is None
    assert regressor.error_training_time is None


def test_experiment_tracker_from_finished_to_reset_transition(
        regressor: Regressor
) -> None:
    ExperimentTracker.handle_training_finished(regressor)
    regressor.reset_status()
    assert regressor.status == TrainingStatus.NOT_STARTED
    assert regressor.start_training_time is None
    assert regressor.stop_training_time is None
    assert regressor.error_training_time is None


def test_experiment_tracker_from_failed_to_reset_transition(
        regressor: Regressor
) -> None:
    ExperimentTracker.handle_training_failed(regressor)
    regressor.reset_status()
    assert regressor.status == TrainingStatus.NOT_STARTED
    assert regressor.start_training_time is None
    assert regressor.stop_training_time is None
    assert regressor.error_training_time is None


@pytest.mark.asyncio
async def test_track_ensemble_model_experiment_with_component_regressors_registered(
        dataset
) -> None:
    _dataset = await dataset
    datasets = [_dataset, _dataset]
    ensemble_regressor = EnsembleRandomForestBasedRegressor()

    regressor_mock = MagicMock()
    regressor_mock.get_regressors.return_value = [RandomForestBasedRegressor(), RandomForestBasedRegressor()]

    assert ensemble_regressor.status == TrainingStatus.NOT_STARTED

    with patch.object(ensemble_regressor, "get_regressors", regressor_mock.get_regressors):
        await ensemble_regressor.fit(datasets=datasets)

    assert ensemble_regressor.status == TrainingStatus.FINISHED


@pytest.mark.asyncio
async def test_track_ensemble_model_experiment_without_component_regressors_registered(
        dataset
) -> None:
    _dataset = await dataset
    datasets = [_dataset, _dataset]
    ensemble_regressor = EnsembleRandomForestBasedRegressor()

    regressor_mock = MagicMock()
    regressor_mock.get_regressors.return_value = []

    assert ensemble_regressor.status == TrainingStatus.NOT_STARTED

    with patch.object(ensemble_regressor, "get_regressors", regressor_mock.get_regressors):
        with pytest.raises(EnsembleModelFitWithoutComponentRegressorsRegisteredError):
            await ensemble_regressor.fit(datasets=datasets)

    assert ensemble_regressor.status == TrainingStatus.ERROR
