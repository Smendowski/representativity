import random

import pytest
from fastapi.testclient import TestClient

from data.models import Dataset, Sample
from main import app


@pytest.fixture(scope="function")
def client():
    return TestClient(app)


@pytest.fixture
def correct_dataset_small() -> Dataset:
    return Dataset(samples=[Sample(features=[random.random() for _ in range(10)]) for _ in range(100)])


@pytest.fixture
def correct_shape_samples():
    return [Sample(features=[random.random() for _ in range(10)]).dict() for _ in range(2)]


@pytest.fixture
def incorrect_shape_samples():
    return [Sample(features=[random.random() for _ in range(5)]).dict() for _ in range(2)]


def test_status_endpoint_when_train_not_invoked(client) -> None:
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {
        "status": "Training has not started yet"
    }


def test_predict_endpoint_with_correct_shape_samples_when_train_not_invoked(client, correct_shape_samples) -> None:
    response = client.post("/predict", json=correct_shape_samples)
    assert response.status_code == 202
    assert response.json() == {
        "detail": "Prediction cannot be made. Regressor is not fitted yet"
    }

    # Sample with a correct shape but without a representativeness field defined
    sample_without_representativeness = {'features': [0.3 for _ in range(10)]}
    response = client.post("/predict", json=[sample_without_representativeness])
    assert response.status_code == 202
    assert response.json() == {
        "detail": "Prediction cannot be made. Regressor is not fitted yet"
    }


def test_predict_endpoint_with_incorrect_shape_samples_when_train_not_invoked(client, incorrect_shape_samples) -> None:
    response = client.post("/predict", json=incorrect_shape_samples)
    assert response.status_code == 202
    assert response.json() == {
        "detail": "Prediction cannot be made. Regressor is not fitted yet"
    }

    # Sample with an incorrect shape but without a representativeness field defined
    sample_without_representativeness = {'features': [0.3 for _ in range(5)]}
    response = client.post("/predict", json=[sample_without_representativeness])
    assert response.status_code == 202
    assert response.json() == {
        "detail": "Prediction cannot be made. Regressor is not fitted yet"
    }


def test_train_model_endpoint_only_accepts_post(client) -> None:
    responses = []
    responses.extend([client.get("/train"), client.put("/train"), client.delete("/train"), client.patch("/train")])
    for response in responses:
        assert response.status_code == 405


def test_complete_scenario_with_correct_dataset_structure(
        client, correct_dataset_small, correct_shape_samples
) -> None:
    response = client.post("/train", json=correct_dataset_small.dict())
    assert response.status_code == 202
    assert response.json() == {
        "detail": "Job has been submitted"
    }

    status_response = client.get("/status")
    assert status_response.status_code == 200
    status_json_response = status_response.json()
    assert set(list(status_json_response.keys())) == set(["status", "start_time", "finish_time"])
    assert status_json_response["status"] == "Training has finished"
    assert status_json_response["start_time"] is not None
    assert status_json_response["finish_time"] is not None

    predict_response = client.post("/predict", json=correct_shape_samples)
    assert predict_response.status_code == 200
    assert len(predict_response.json()["representativeness"]) == len(correct_shape_samples)


def test_train_model_endpoint_with_post_and_incorrect_dataset_structure(client) -> None:
    response = client.post("/train", json={
        "samples": {
            "sample_1": {
                "features": [1, 2, 3, 4, 5]
            },
            "sample_2": {
                "features": [1, 2, 3, 4, 5]
            },
            "sample_3": {
                "features": [1, 2, 3, 4, 5]
            }
        }
    })
    assert response.status_code == 422
