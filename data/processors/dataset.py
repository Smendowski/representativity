import asyncio
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from data.extractors import RepresentativenessExtractor
from data.models import Dataset, Sample
from logs import Logger

logger = Logger(__name__)
# TODO: logowanie


class DatasetProcessor:
    @staticmethod
    async def create_dataset(samples: int, features: int) -> Dataset:
        def _create(_samples: int, _features: int) -> Dataset:
            return Dataset(samples=[
                Sample(features=[random.random() for _ in range(_features)]) for _ in range(_samples)
            ])
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            dataset = await loop.run_in_executor(executor, _create, samples, features)
        return dataset

    @staticmethod
    def run_labeling(dataset: Dataset, extractor: RepresentativenessExtractor) -> Dataset:
        _dataset = dataset.copy()
        features = _dataset.get_feature_representation()

        representativeness: np.ndarray[float] = extractor.extract(features)
        for sample, value in zip(_dataset.samples, representativeness):
            sample.representativeness = value

        return _dataset

    @staticmethod
    async def to_supervised(dataset: Dataset, splits: int, extractor: RepresentativenessExtractor) -> list[Dataset]:
        _dataset = dataset.copy()
        np.random.shuffle(_dataset.samples)

        def _to_dataset(_chunk: np.ndarray) -> Dataset:
            return Dataset(samples=_chunk.tolist())

        with ThreadPoolExecutor() as executor:

            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(executor, _to_dataset, chunk)
                for chunk in np.array_split(_dataset.samples, splits)
            ]
            chunks = await asyncio.gather(*tasks)

            tasks = [
                loop.run_in_executor(executor, DatasetProcessor.run_labeling, chunk, extractor)
                for chunk in chunks
            ]

            labeled_chunks = await asyncio.gather(*tasks)

        return list(labeled_chunks)
