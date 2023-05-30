import numpy as np
import pytest
from pytest import MonkeyPatch

from data.extractors import NearestNeighborsBasedRepresentativenessExtractor
from exceptions import InvalidNNeighborsError


@pytest.mark.parametrize("n_neighbors", [0, -1, 100])
def test_nearest_neighbors_based_extractor_with_incorrect_n_neighbors_specified(
        features: np.ndarray, monkeypatch: MonkeyPatch, n_neighbors: np.ndarray
) -> None:
    with monkeypatch.context() as context:
        context.setenv("N_NEIGHBORS", str(n_neighbors))
        extractor = NearestNeighborsBasedRepresentativenessExtractor()
        with pytest.raises(InvalidNNeighborsError):
            representativeness = extractor.extract(features)
            assert representativeness is None


@pytest.mark.parametrize(
    "n_neighbors, expected_representativeness",
    [
        (1, np.array([np.nan, np.nan, np.nan, np.nan, np.nan])),
        (3, np.array([0.22966848, 0.30901699, 0.30901699, 0.30901699, 0.22966848])),
        (5, np.array([0.15174116, 0.2035367, 0.22966848, 0.2035367, 0.15174116]))
    ]
)
def test_nearest_neighbors_based_extractor_with_correct_n_neighbors_specified(
        features: np.ndarray, monkeypatch: MonkeyPatch, n_neighbors: int, expected_representativeness: np.ndarray
) -> None:
    with monkeypatch.context() as context:
        context.setenv("N_NEIGHBORS", str(n_neighbors))
        extractor = NearestNeighborsBasedRepresentativenessExtractor()
        representativeness = extractor.extract(features)
        assert representativeness.shape == expected_representativeness.shape == (features.shape[0],)
        assert np.array_equal(representativeness, expected_representativeness) or \
            np.all(np.isnan(representativeness) == np.isnan(expected_representativeness))
