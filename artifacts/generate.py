import random

from data.models import Dataset, Sample
from data.processors import DatasetProcessor
import asyncio
import json


async def main() -> None:
    dataset: Dataset = await DatasetProcessor.create_dataset(1_000, 5)
    with open("artifacts/dataset_1_000_samples_5_features.json", "w") as file:
        file.write(json.dumps(dataset.dict()))

    dataset: Dataset = await DatasetProcessor.create_dataset(10_000, 10)
    with open("artifacts/dataset_10_000_samples_10_features.json", "w") as file:
        file.write(json.dumps(dataset.dict()))

    dataset: Dataset = await DatasetProcessor.create_dataset(100_000, 10)
    with open("artifacts/dataset_100_000_samples_10_features.json", "w") as file:
        file.write(json.dumps(dataset.dict()))

    samples = [Sample(features=[random.random() for _ in range(5)]).dict() for _ in range(5)]
    with open("artifacts/samples_5_features.json", "w") as file:
        file.write(json.dumps(samples))

    samples = [Sample(features=[random.random() for _ in range(10)]).dict() for _ in range(5)]
    with open("artifacts/samples_10_features.json", "w") as file:
        file.write(json.dumps(samples))

if __name__ == "__main__":
    asyncio.run(main())
