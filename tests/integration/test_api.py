from fastapi.testclient import TestClient


def _client():
    from credit_scoring.serving.app import create_app

    return TestClient(create_app())


def test_health_reports_loaded_model(model_env):
    with _client() as client:
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["model_version"] == "test-0001"


def test_predict_returns_a_decision(model_env, example):
    with _client() as client:
        response = client.post("/predict", json=example)
        assert response.status_code == 200
        body = response.json()
        assert 0.0 <= body["default_probability"] <= 1.0
        assert body["decision"] in ("approve", "decline")
        assert body["model_version"] == "test-0001"


def test_invalid_input_returns_422(model_env, example):
    example["age"] = 5
    with _client() as client:
        assert client.post("/predict", json=example).status_code == 422


def test_model_info_exposes_metadata(model_env):
    with _client() as client:
        info = client.get("/model/info").json()
        assert info["algorithm"] == "gradient_boosting"
        assert info["version"] == "test-0001"


def test_batch_and_metrics(model_env, example):
    with _client() as client:
        batch = client.post("/predict/batch", json={"applications": [example, example]})
        assert batch.status_code == 200
        body = batch.json()
        assert len(body["predictions"]) == 2
        assert "drift_warning" in body

        metrics = client.get("/metrics").text
        assert "credit_predictions_total" in metrics


def test_dashboard_assets_are_served(model_env):
    with _client() as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "text/html" in page.headers["content-type"]
        assert 'id="score-form"' in page.text
        for asset in ("style.css", "app.js", "fonts/manrope-regular.ttf"):
            assert client.get(f"/assets/{asset}").status_code == 200


def test_dashboard_reports_model_and_scoring_activity(model_env, example):
    with _client() as client:
        before = client.get("/dashboard/data").json()
        assert before["model"]["version"] == "test-0001"
        prediction = client.post("/predict", json=example).json()
        after = client.get("/dashboard/data").json()
        decision = prediction["decision"]
        assert after["predictions"][decision] == before["predictions"].get(decision, 0) + 1
        client.post("/predict/batch", json={"applications": [example, example]})
        batch = client.get("/dashboard/data").json()
        assert "age" in batch["drift"]
        assert batch["drift_threshold"] > 0


def test_dashboard_remains_available_without_a_model(model_env):
    from credit_scoring.serving import service

    with _client() as client:
        service._service = service.ScoringService()
        assert client.get("/").status_code == 200
        assert client.get("/dashboard/data").json()["model"] is None
        assert client.get("/model/info").status_code == 503
