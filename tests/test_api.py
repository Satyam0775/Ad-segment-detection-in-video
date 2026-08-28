from unittest.mock import patch

from fastapi.testclient import TestClient

from src.acquisition.validators import InvalidURLError
from src.main import app
from src.pipeline import PipelineResult
from src.schemas.response import DetectionResponse, Kind, Platform, SourceInfo, Stats

client = TestClient(app)


def test_detect_rejects_empty_url():
    resp = client.post("/api/v1/detect", json={"url": ""})
    assert resp.status_code == 422  # pydantic request validation


def test_detect_rejects_unsupported_url():
    with patch("src.api.routes.run_pipeline", side_effect=InvalidURLError("not a valid http(s) URL")):
        resp = client.post("/api/v1/detect", json={"url": "not a url"})
    assert resp.status_code == 400
    assert "not a valid" in resp.json()["detail"]


def test_detect_returns_valid_contract_shape_on_success():
    fake_response = DetectionResponse(
        source=SourceInfo(url="https://youtube.com/watch?v=abc", platform=Platform.youtube, kind=Kind.vod, duration_s=100.0),
        segments=[],
        stats=Stats(wall_clock_s=1.0, estimated_cost_usd=0.0, frames_sampled=10, model_calls=0),
    )
    fake_result = PipelineResult(response=fake_response, work_dir=__import__("pathlib").Path("."))

    with patch("src.api.routes.run_pipeline", return_value=fake_result):
        resp = client.post("/api/v1/detect", json={"url": "https://youtube.com/watch?v=abc"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["source"]["platform"] == "youtube"
    assert "stats" in body
    assert body["segments"] == []


def test_detect_does_not_leak_raw_traceback_on_unexpected_error():
    with patch("src.api.routes.run_pipeline", side_effect=RuntimeError("some internal detail with a secret path")):
        resp = client.post("/api/v1/detect", json={"url": "https://youtube.com/watch?v=abc"})
    assert resp.status_code == 500
    assert "secret path" not in resp.json()["detail"]
