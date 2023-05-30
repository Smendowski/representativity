from enum import Enum
from datetime import datetime
from exceptions import (
    ModelNotFittedError,
    EnsembleModelFitWithoutComponentRegressorsRegisteredError,
    InferenceSampleHasUnexpectedShapeError
)
from functools import wraps
from typing import Callable


class TrainingStatus(Enum):
    NOT_STARTED = "Training has not started yet"
    AWAIT = "Awaiting data to be preprocessed"
    DURING_TRAINING = "Training in progress"
    ERROR = "An error has occurred during training"
    FINISHED = "Training has finished"


class ExperimentTracker:
    @staticmethod
    def get_current_datetime_representation() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def handle_training_started(regressor) -> None:
        regressor.status = TrainingStatus.DURING_TRAINING
        regressor.start_training_time = ExperimentTracker.get_current_datetime_representation()

    @staticmethod
    def handle_training_finished(regressor) -> None:
        regressor.status = TrainingStatus.FINISHED
        regressor.stop_training_time = ExperimentTracker.get_current_datetime_representation()

    @staticmethod
    def handle_training_failed(regressor) -> None:
        regressor.status = TrainingStatus.ERROR
        regressor.error_training_time = ExperimentTracker.get_current_datetime_representation()


def ensure_fitted(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if hasattr(self, "_regressors"):
            regressors = self.get_regressors()
            if len(regressors) > 0 and self.status == TrainingStatus.FINISHED:
                expected_sample_shape = (regressors[0].model.estimators_[0].max_features_,)
                inference_sample_shape = (len(args[0].features),)
                if expected_sample_shape != inference_sample_shape:
                    raise InferenceSampleHasUnexpectedShapeError(
                        expected_sample_shape=expected_sample_shape,
                        inference_sample_shape=inference_sample_shape
                    )
            elif len(regressors) == 0 or self.status != TrainingStatus.FINISHED:
                raise ModelNotFittedError
        else:
            expected_sample_shape = (self.model.estimators_[0].max_features_,)
            inference_sample_shape = (len(args[0].features),)
            if expected_sample_shape != inference_sample_shape:
                raise InferenceSampleHasUnexpectedShapeError(
                    expected_sample_shape=expected_sample_shape,
                    inference_sample_shape=inference_sample_shape
                )
        return func(self, *args, **kwargs)
    return wrapper


def track_experiment(func: Callable) -> Callable:
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        ExperimentTracker.handle_training_started(self)
        if len(self.get_regressors()) == 0:
            ExperimentTracker.handle_training_failed(self)
            raise EnsembleModelFitWithoutComponentRegressorsRegisteredError()
        else:
            try:
                await func(self, *args, **kwargs)
            except Exception:
                ExperimentTracker.handle_training_failed(self)
        ExperimentTracker.handle_training_finished(self)

    return wrapper
