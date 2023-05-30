from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import BaseEnsemble, RandomForestRegressor

from data.models import Dataset, Sample
from ml.helpers import TrainingStatus, ensure_fitted, track_experiment


class Regressor(ABC):
    @abstractmethod
    def __init__(self) -> None:
        self._status: TrainingStatus = TrainingStatus.NOT_STARTED
        self._start_training_time: str | None = None
        self._stop_training_time: str | None = None
        self._error_training_time: str | None = None

    @property
    @abstractmethod
    def model(self) -> Regressor | BaseEnsemble | BaseEstimator:
        ...

    @property
    @abstractmethod
    def status(self) -> TrainingStatus:
        ...

    @status.setter
    @abstractmethod
    def status(self, status: TrainingStatus) -> None:
        ...

    @abstractmethod
    def reset_status(self) -> None:
        ...

    def get_verbose_status(self) -> dict[str, str]:
        verbose_statuses: dict[TrainingStatus, dict[str, str]] = {
            TrainingStatus.NOT_STARTED: {
                "status": self._status.value
            },
            TrainingStatus.AWAIT: {
                "status": self._status.value,
            },
            TrainingStatus.DURING_TRAINING: {
                "status": self._status.value,
                "start_time": self.start_training_time,
            },
            TrainingStatus.ERROR: {
                "status": self._status.value,
                "start_time": self.start_training_time,
                "error_time": self.error_training_time,
            },
            TrainingStatus.FINISHED: {
                "status": self._status.value,
                "start_time": self.start_training_time,
                "finish_time": self.stop_training_time,
            }
        }

        return verbose_statuses[self._status]

    @property
    def start_training_time(self) -> str:
        return self._start_training_time

    @property
    def stop_training_time(self) -> str:
        return self._stop_training_time

    @property
    def error_training_time(self) -> str:
        return self._error_training_time

    @start_training_time.setter
    def start_training_time(self, datetime_representation: str):
        self._start_training_time = datetime_representation

    @stop_training_time.setter
    def stop_training_time(self, datetime_representation: str):
        self._stop_training_time = datetime_representation

    @error_training_time.setter
    def error_training_time(self, datetime_representation: str):
        self._error_training_time = datetime_representation

    @abstractmethod
    def fit(self, dataset: Dataset) -> None:
        ...

    @ensure_fitted
    @abstractmethod
    def predict(self, sample: Sample) -> float:
        ...


class RandomForestBasedRegressor(Regressor):
    def __init__(self) -> None:
        super().__init__()
        self._model: RandomForestRegressor = RandomForestRegressor()

    @property
    def model(self) -> RandomForestRegressor:
        return self._model

    @property
    def status(self) -> TrainingStatus:
        return self._status

    @status.setter
    def status(self, status: TrainingStatus) -> None:
        if isinstance(status, TrainingStatus):
            self._status = status

    def reset_status(self) -> None:
        self.status = TrainingStatus.NOT_STARTED
        self.start_training_time = None
        self.stop_training_time = None
        self.error_training_time = None

    def fit(self, dataset: Dataset) -> None:
        features = dataset.get_feature_representation()
        targets = dataset.get_target_representation()
        self._model.fit(X=features, y=targets)

    @ensure_fitted
    def predict(self, sample: Sample) -> float:
        features = np.array(sample.features).reshape(1, -1)
        return self._model.predict(features)[0]


class EnsembleRandomForestBasedRegressor(Regressor):
    def __init__(self):
        super().__init__()
        self._regressors: list[Regressor] = []

    @property
    def model(self) -> list[Regressor]:
        return self.get_regressors()

    @property
    def status(self) -> TrainingStatus:
        return self._status

    @status.setter
    def status(self, status: TrainingStatus) -> None:
        if isinstance(status, TrainingStatus):
            self._status = status
            for regressor in self._regressors:
                regressor.status = status

    def reset_status(self) -> None:
        self.status = TrainingStatus.NOT_STARTED
        self.start_training_time = None
        self.stop_training_time = None
        self.error_training_time = None
        self._regressors = []

    def register_regressor(self, regressor: Regressor):
        self._regressors.append(regressor)

    def deregister_regressors(self) -> None:
        self._regressors = []

    def get_regressors(self) -> list[Regressor | None]:
        if not self._regressors:
            return []
        return self._regressors

    @track_experiment
    async def fit(self, datasets: list[Dataset]) -> None:
        with ThreadPoolExecutor() as executor:
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(executor, regressor.fit, dataset_chunk)
                for regressor, dataset_chunk in zip(self.get_regressors(), datasets)
            ]

            await asyncio.gather(*tasks)

    @ensure_fitted
    async def predict(self, sample: Sample) -> float:
        with ThreadPoolExecutor() as executor:
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(executor, regressor.predict, sample)
                for regressor in self.get_regressors()
            ]

            predictions = await asyncio.gather(*tasks)

        return round(np.mean(predictions), 5)
