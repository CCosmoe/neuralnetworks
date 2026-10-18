import numpy as np
from fastapi.testclient import TestClient

from api.main import app, model
from Network import generate_spiral_data

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json()["layer_sizes"] == [2, 64, 64, 3]


def test_predict_returns_valid_probabilities():
    response = client.post("/predict", json={"points": [[0.3, -0.5], [0.0, 0.0]]})
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 2
    for prediction in predictions:
        assert len(prediction["probabilities"]) == 3
        assert abs(sum(prediction["probabilities"]) - 1) < 1e-4
        assert prediction["predicted_class"] == int(np.argmax(prediction["probabilities"]))


def test_saved_model_is_accurate_on_fresh_data():
    np.random.seed(0)
    X, y = generate_spiral_data(100, 3)
    accuracy = np.mean(np.argmax(model.predict_proba(X), axis=1) == np.argmax(y, axis=1))
    assert accuracy > 0.9


def test_rejects_wrong_point_shape():
    response = client.post("/predict", json={"points": [[1.0, 2.0, 3.0]]})
    assert response.status_code == 422


def test_rejects_empty_request():
    response = client.post("/predict", json={"points": []})
    assert response.status_code == 422


def test_rejects_too_many_points():
    response = client.post("/predict", json={"points": [[0.0, 0.0]] * 1001})
    assert response.status_code == 422
