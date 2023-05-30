import numpy as np
from pydantic import BaseModel, validator

from exceptions import IncorrectSamplesShapeInDatasetError

from .sample import Sample


class Dataset(BaseModel):
    samples: list[Sample]

    @validator("samples")
    def validate_samples(cls, samples: list[Sample]) -> list[Sample]:
        samples_representation = np.array([sample.features for sample in samples])
        if not np.all(samples_representation.shape[1] == samples_representation[0].shape[0]):
            raise IncorrectSamplesShapeInDatasetError()

        return samples

    def __len__(self):
        return len(self.samples)

    def get_feature_representation(self) -> np.ndarray:
        return np.array([sample.features for sample in self.samples])

    def get_target_representation(self) -> np.ndarray:
        return np.array([sample.representativeness for sample in self.samples])
