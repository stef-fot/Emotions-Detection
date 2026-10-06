"""
Flask server for the Emotion Detector dashboard and JSON API.

Endpoints
  GET  /                         dashboard UI
  POST /analyze                  {"text": "..."}              -> analysis
  POST /analyze/batch            {"texts": ["...", ...]}      -> list of analyses
  GET  /emotionDetector?textToAnalyse=...   legacy GET wrapper (original lab route)
  GET  /api/meta                 model + language coverage
  GET  /health                   liveness probe

Everything runs in-process: no API keys, no external services.
"""

from __future__ import annotations

import json
import os

from flask import Flask, jsonify, render_template, request

from EmotionDetection.emotion_detection import (LABELS, MAX_INPUT_CHARS,
                                                MODEL_NAME, MODEL_VERSION,
                                                LEXICON_LANGS, emotion_detector)
from EmotionDetection.lexicon import EMOJI_EMOTIONS, LANG_NAMES

MAX_BATCH = 50


def _emoji_groups() -> dict:
    """{emotion: [emoji, ...]} straight from the lexicon, so the UI's emoji
    picker can never drift from what the analyzer actually recognises."""
    groups: dict = {}
    for emoji, emotion in EMOJI_EMOTIONS.items():
        groups.setdefault(emotion, []).append(emoji)
    return groups


app = Flask(__name__)
# Reject oversized bodies early (a 4000-char text is ~16 KB even in UTF-8).
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024
app.json.ensure_ascii = False


@app.after_request
def _headers(response):
    # The API is public and stateless, so any origin may call it.
    response.headers.setdefault("Access-Control-Allow-Origin", "*")
    response.headers.setdefault("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    response.headers.setdefault("Access-Control-Allow-Headers", "Content-Type")
    response.headers.setdefault("Access-Control-Max-Age", "3600")
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    if request.path.startswith(("/analyze", "/api", "/emotionDetector")):
        response.headers.setdefault("Cache-Control", "no-store")
    return response


@app.errorhandler(413)
def _too_large(_):
    return jsonify({"ok": False, "message": "Request is too large."}), 413


@app.errorhandler(404)
def _not_found(_):
    return jsonify({"ok": False, "message": "Not found."}), 404


@app.errorhandler(405)
def _bad_method(_):
    return jsonify({"ok": False, "message": "Method not allowed."}), 405


@app.route("/")
def render_index_page():
    """Renders the dashboard UI."""
    return render_template("index.html",
                           model_version=MODEL_VERSION,
                           languages=len(LEXICON_LANGS),
                           max_chars=MAX_INPUT_CHARS,
                           emoji_groups=_emoji_groups())


def _text_from_request() -> str:
    payload = request.get_json(silent=True)
    if isinstance(payload, dict):
        value = payload.get("text")
        if isinstance(value, str) and value:
            return value
    return request.args.get("textToAnalyse", "") or request.args.get("text", "")


@app.route("/analyze", methods=["POST"])
def analyze():
    """Analyse one text. Also accepts ``?text=`` / ``?textToAnalyse=``."""
    result = emotion_detector(_text_from_request())
    return jsonify(result), (200 if result.get("ok") else 400)


@app.route("/analyze/batch", methods=["POST"])
def analyze_batch():
    """Analyse up to 50 texts in one call: {"texts": ["...", ...]}."""
    payload = request.get_json(silent=True)
    texts = payload.get("texts") if isinstance(payload, dict) else None
    if not isinstance(texts, list) or not texts:
        return jsonify({"ok": False, "message": 'Send {"texts": ["...", ...]}.'}), 400
    if len(texts) > MAX_BATCH:
        return jsonify({"ok": False,
                        "message": f"At most {MAX_BATCH} texts per request."}), 400
    results = [emotion_detector(t if isinstance(t, str) else "") for t in texts]
    return jsonify({"ok": True, "count": len(results), "results": results})


# Backwards-compatible with the original `/emotionDetector?textToAnalyse=...`
# lab route; returns the rich JSON.
@app.route("/emotionDetector")
def sent_detector():
    result = emotion_detector(request.args.get("textToAnalyse", ""))
    return app.response_class(
        response=json.dumps(result, ensure_ascii=False),
        status=200 if result.get("ok") else 400,
        mimetype="application/json",
    )


@app.route("/api/meta")
def meta():
    return jsonify({
        "model": {"name": MODEL_NAME, "version": MODEL_VERSION},
        "labels": list(LABELS),
        "emoji": _emoji_groups(),
        "max_input_chars": MAX_INPUT_CHARS,
        "max_batch": MAX_BATCH,
        "lexicon_languages": sorted(
            ({"code": c, "name": LANG_NAMES.get(c, c)} for c in LEXICON_LANGS),
            key=lambda d: d["name"]),
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"),
            port=int(os.environ.get("PORT", "5000")),
            debug=os.environ.get("FLASK_DEBUG") == "1")
