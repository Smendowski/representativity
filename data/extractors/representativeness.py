import os
from abc import ABC, abstractmethod

import numpy as np
from pydantic.types import PositiveFloat
from sklearn.neighbors import NearestNeighbors

from exceptions import InvalidNNeighborsError


class RepresentativenessExtractor(ABC):
    @staticmethod
    @abstractmethod
    def extract(features: np.ndarray) -> np.ndarray:
        ...

    @staticmethod
    @abstractmethod
    def _calculate_representativeness(mean_distance: PositiveFloat) -> PositiveFloat:
        ...


class NearestNeighborsBasedRepresentativenessExtractor(RepresentativenessExtractor):
    @staticmethod
    def extract(features: np.ndarray) -> np.ndarray:

        n_neighbors = int(os.environ.get("N_NEIGHBORS", 5))
        if not n_neighbors or not n_neighbors > 0 or n_neighbors > len(features):
            raise InvalidNNeighborsError(n_neighbors=n_neighbors)

        neighbors = NearestNeighbors(n_neighbors=n_neighbors).fit(features)
        distances, _ = neighbors.kneighbors(features)
        mean_distances = np.mean(distances[:, 1:], axis=1)

        return np.array(
            list(map(NearestNeighborsBasedRepresentativenessExtractor._calculate_representativeness, mean_distances))
        )

    @staticmethod
    def _calculate_representativeness(mean_distance: PositiveFloat) -> PositiveFloat:
        return 1 / (1 + mean_distance)
