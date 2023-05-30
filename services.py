import os

from data.models import Dataset, Sample
from data.processors import DatasetProcessor
from data.extractors import NearestNeighborsBasedRepresentativenessExtractor

from ml.models import RandomForestBasedRegressor, ensemble_random_forest_based_regressor

from dotenv import load_dotenv
from os.path import join, dirname

load_dotenv(join(dirname(__file__), ".env"))
NUMBER_OF_ENSEMBLE_MODELS: int = int(os.environ.get("NUMBER_OF_ENSEMBLE_MODELS", 5))


async def prepare_dataset(dataset: Dataset) -> list[Dataset]:
    supervised_dataset_chunked: list[Dataset] = await DatasetProcessor.to_supervised(
        dataset=dataset,
        splits=NUMBER_OF_ENSEMBLE_MODELS,
        extractor=NearestNeighborsBasedRepresentativenessExtractor()
    )

    return supervised_dataset_chunked


async def train_model(dataset: Dataset) -> None:
    ensemble_random_forest_based_regressor.reset_status()
    supervised_dataset_chunked = await prepare_dataset(dataset)

    for _ in range(NUMBER_OF_ENSEMBLE_MODELS):
        ensemble_random_forest_based_regressor.register_regressor(RandomForestBasedRegressor())

    await ensemble_random_forest_based_regressor.fit(supervised_dataset_chunked)


async def get_model_prediction(sample: Sample) -> float:
    return await ensemble_random_forest_based_regressor.predict(sample)


async def get_model_status() -> dict[str, str]:
    return ensemble_random_forest_based_regressor.get_verbose_status()
