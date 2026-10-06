"""
Serverless entry points (Google Cloud Functions / AWS Lambda) for the
Emotion Detector. Same analyzer as the Flask app; deploy from the
repository root so the ``EmotionDetection`` package is included
(the root ``main.py`` re-exports ``analyze``).
"""

import json
import os
import sys
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from EmotionDetection.emotion_detection import emotion_detector  # noqa: E402

_CORS = {"Access-Control-Allow-Origin": "*"}


@lru_cache(maxsize=2048)
def _cached(text: str) -> str:
    """Per warm instance: repeated phrases skip all work."""
    return json.dumps(emotion_detector(text), ensure_ascii=False)


def analyze(request):
    """HTTP entry point. Body {"text": "..."} (or ?text=...)."""
    if request.method == "OPTIONS":
        return ("", 204, {**_CORS, "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
                          "Access-Control-Allow-Headers": "Content-Type",
                          "Access-Control-Max-Age": "3600"})
    payload = request.get_json(silent=True)
    text = payload.get("text") if isinstance(payload, dict) else None
    if not isinstance(text, str) or not text:
        text = request.args.get("text", "")
    body = _cached(text.strip())
    status = 200 if json.loads(body).get("ok") else 400
    return (body, status, {**_CORS, "Content-Type": "application/json"})


def lambda_handler(event, context):  # pragma: no cover
    """AWS Lambda adapter (API Gateway proxy event or {"text": "..."})."""
    import base64
    body = event.get("body") or ""
    if event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode("utf-8")
    text = ""
    try:
        text = (json.loads(body) or {}).get("text", "") if body else ""
    except (ValueError, AttributeError):
        pass
    text = text or event.get("text") or (event.get("queryStringParameters") or {}).get("text", "")
    result = emotion_detector(text)
    return {"statusCode": 200 if result.get("ok") else 400,
            "headers": {"Content-Type": "application/json", **_CORS},
            "body": json.dumps(result, ensure_ascii=False)}
