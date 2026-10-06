import pytest

import server


@pytest.fixture()
def client():
    return server.app.test_client()


def test_health(client):
    assert client.get("/health").json == {"status": "ok"}


def test_analyze_ok(client):
    r = client.post("/analyze", json={"text": "I love this"})
    assert r.status_code == 200
    assert r.json["primary_emotion"] == "joy"
    assert r.headers["Access-Control-Allow-Origin"] == "*"


def test_analyze_bad_input(client):
    assert client.post("/analyze", json={"text": ""}).status_code == 400
    assert client.post("/analyze", data="not json").status_code == 400
    assert client.post("/analyze", json={"text": 123}).status_code == 400


def test_analyze_query_string_fallback(client):
    assert client.post("/analyze?text=I%20am%20happy").json["ok"] is True


def test_legacy_route(client):
    r = client.get("/emotionDetector", query_string={"textToAnalyse": "μου λείπεις"})
    assert r.status_code == 200 and r.json["primary_emotion"] == "sadness"


def test_batch(client):
    r = client.post("/analyze/batch", json={"texts": ["so happy", "so sad", ""]})
    assert r.status_code == 200
    assert [x["ok"] for x in r.json["results"]] == [True, True, False]


def test_batch_validation(client):
    assert client.post("/analyze/batch", json={}).status_code == 400
    assert client.post("/analyze/batch", json={"texts": ["a"] * 51}).status_code == 400


def test_meta_lists_languages(client):
    codes = {l["code"] for l in client.get("/api/meta").json["lexicon_languages"]}
    assert {"en", "el", "es", "ja"} <= codes


def test_options_preflight(client):
    r = client.options("/analyze")
    assert r.status_code in (200, 204)
    assert "POST" in r.headers["Access-Control-Allow-Methods"]


def test_oversized_body_rejected(client):
    r = client.post("/analyze", data="x" * (600 * 1024), content_type="application/json")
    assert r.status_code == 413


def test_removed_chat_endpoints_are_gone(client):
    # the paid LLM chat companion was removed on purpose
    assert client.post("/chat", json={"message": "hi"}).status_code == 404
    assert client.post("/chat/end", json={}).status_code == 404


def test_index_renders(client):
    r = client.get("/")
    assert r.status_code == 200 and b"Emotion" in r.data
    assert b"ANTHROPIC" not in r.data
