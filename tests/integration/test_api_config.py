"""API route tests for inference and judge config."""

from fastapi.testclient import TestClient

from musicai_api.main import app

client = TestClient(app)


def test_inference_config_endpoint():
    response = client.get("/v1/inference/config")
    assert response.status_code == 200
    data = response.json()
    assert data["runtime"] in {"local", "cloud"}
    assert "demucs_model" in data


def test_inference_status_endpoint():
    response = client.get("/v1/inference/status")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "adapters" in data


def test_judge_config_endpoint():
    response = client.get("/v1/judge/config")
    assert response.status_code == 200
    data = response.json()
    assert data["use_music21"] is True
    assert data["max_simultaneous_notes"] == 4
