import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

import text_classifier.service as service
from text_classifier.api import app
from text_classifier.classification import Classifier
from text_classifier.summarization import Summarizer


@pytest.fixture
def client(monkeypatch, fakes):
    sent, zs, summ = fakes
    monkeypatch.setattr(service, "Classifier", lambda: Classifier(sent, zs))
    monkeypatch.setattr(service, "Summarizer", lambda: Summarizer(summ, 30))
    return TestClient(app)


def test_health_is_instant(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["loaded"] == 0


def test_analyze_text(client, article):
    r = client.post("/analyze", json={"text": article})
    assert r.status_code == 200
    assert r.json()["classification"]["sentiment"]["label"] == "positive"


def test_analyze_file(client, article):
    r = client.post("/analyze/file", files={"file": ("a.txt", article.encode(), "text/plain")})
    assert r.status_code == 200 and r.json()["summary"]


def test_bad_inputs(client):
    assert client.post("/analyze", json={}).status_code == 422
    r = client.post("/analyze/file", files={"file": ("a.png", b"x", "image/png")})
    assert r.status_code == 422 and "Unsupported" in r.json()["detail"]
