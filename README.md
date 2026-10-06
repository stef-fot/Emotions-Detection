<div align="center">

# 🎭 Emotion Detector

**Explainable multilingual emotion & sentiment analysis. Offline, no API keys, ~1 ms.**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3-000000?logo=flask&logoColor=white)
![Tests](https://img.shields.io/badge/tests-73_passing-27e0a4)
![Runs offline](https://img.shields.io/badge/runs-100%25_offline-19e3c1)

</div>

Paste text in almost any language. The app returns a probability distribution over **joy, sadness, anger, fear, disgust, surprise and neutral**, a sentiment score, a calibrated confidence, a **sentence-by-sentence emotional arc**, and the **exact words that drove the decision** (highlighted, with negation and intensifier flags).

It started as a Watson NLP course lab. It is now a self-contained, tested NLP project with a measured evaluation.

## What makes it different

| | |
| --- | --- |
| 🔍 **Explainable** | Every match comes back as `evidence` with its character span, weight, and flags (`negated`, `intensified`, `emoji`, `english-fallback`). The UI highlights them. |
| 🌍 **54 lexicon languages** | Script + function-word language ID, accent folding (`χαρουμενος` = `χαρούμενος`), light stemming for inflected languages (Greek, Slavic) and agglutinative ones (Turkish, Korean). Other scripts are detected but honestly flagged as "no lexicon". |
| 🧠 **Context aware** | Negation (`not happy` becomes sadness, clause-bounded), intensifiers, elongation (`sooo`), CAPS, `!`, emoji, idioms (`could not be happier`). |
| 📈 **Emotional arc** | Per-sentence sentiment timeline, so mixed or turning moods are visible. |
| 📏 **Evaluated** | `python -m evaluation.evaluate` reports accuracy, per-class P/R/F1 and a confusion matrix on a hand-labelled multilingual set. |
| 🛡️ **Responsible** | Self-harm phrases (Greek, English, Spanish and more) trigger helpline resources, deterministically. Low-evidence and unsupported-language results lower the confidence and show warnings. |
| ⚡ **Zero cost** | No LLM, no API key, no network. The earlier paid chat companion was removed on purpose. |

## Quick start

```bash
pip install -r requirements.txt
python server.py            # http://127.0.0.1:5000
```

```bash
python -m EmotionDetection "I am NOT happy, this is DISGUSTING"     # CLI
pip install -r requirements-dev.txt && pytest                       # 73 tests
python -m evaluation.evaluate --errors                              # model report
```

Docker: `docker build -t emotion-detector . && docker run -p 8080:8080 emotion-detector`

## How it works

1. **Clean**: strip control characters and URLs, normalise whitespace, cap at 4000 chars.
2. **Identify language**: dominant Unicode script, then diagnostic letters (`ї` Ukrainian, kana means Japanese, `پ` Persian) and, for Latin script, function words + lexicon hits + diacritics.
3. **Segment** into sentences.
4. **Match** the per-language lexicon (words and phrases) on case- and accent-folded tokens, with English fallback at 0.4 weight for code-switching. Hits are modified by negation, intensifiers, elongation, caps and `!`; emoji add a strong signal.
5. **Distribute**: `p(class) ∝ (evidence + 0.05)^1.3`, with a neutral pseudo-count of 0.8, so no evidence means neutral and evidence shifts mass smoothly.
6. **Calibrate**: `confidence = 0.99 · sqrt(1 − e^(−evidence/2.5)) · min(1, 0.4 + 1.2·margin)`, discounted for language uncertainty and capped at 0.25 when the language has no lexicon.
7. **Explain**: return evidence spans, timeline, warnings, and crisis resources if needed.

It is a transparent lexicon model, not a neural network. That is a deliberate trade: fast, offline, deterministic and fully inspectable, at the cost of recall on slang, sarcasm and unseen vocabulary.

## Evaluation (honest numbers)

`evaluation/dataset.csv` has 279 short hand-written sentences in 13+ languages. The splits matter:

| Split | n | Accuracy | Macro-F1 | How to read it |
| --- | ---: | ---: | ---: | --- |
| `test1` (first run, before error analysis) | 70 | **87.1%** | 0.883 | Written before looking at any output. |
| `holdout` (first run, before error analysis) | 68 | **88.2%** | 0.904 | Fresh sentences, again unseen. |
| `blind` (first run) | 49 | **93.9%** | 0.888 | Written after the first fixes. 98.0% after one more general fix (negation must not cross a comma). |
| `dev` | 92 | 100% | 1.000 | Used while building the lexicon: optimistic. |

After each first run I did error analysis and fixed general gaps (English suffixes, Korean/Turkish stems, clause-bounded negation, more vocabulary). Those splits are now **regression tests** (`tests/test_evaluation.py` enforces floors) and no longer measure generalisation. The majority-class baseline is about 25%. The realistic expectation on new text is **roughly 85 to 95% on clear, short, emotional sentences**, and much lower on sarcasm, long texts, slang and low-resource languages. The set is small: treat it as a sanity check, not a benchmark. Greek, English, Spanish, French, German, Russian and Japanese are best covered.

## API

`POST /analyze` with `{"text": "..."}`

```json
{
  "ok": true, "language": "English", "language_code": "en", "language_confidence": 0.81,
  "primary_emotion": "disgust",
  "distribution_percent": {"joy": 0.3, "sadness": 29.6, "anger": 0.3, "fear": 0.3, "disgust": 52.0, "surprise": 0.3, "neutral": 17.2},
  "sentiment": {"score": -0.62, "positivity": 19.0, "label": "negative"},
  "confidence": 0.54,
  "summary": "Mainly disgust (52%), mixed with sadness (30%). Overall tone: negative.",
  "evidence": [{"start": 5, "end": 10, "text": "happy", "emotion": "sadness", "weight": 1.2, "negated": true, "intensified": false, "source": "lexicon"}],
  "timeline": [{"text": "...", "primary_emotion": "disgust", "positivity": 13.7}],
  "warnings": [], "crisis": false, "crisis_resources": null, "elapsed_ms": 0.4
}
```

Other routes: `POST /analyze/batch` (`{"texts": [...]}`, max 50), `GET /api/meta`, `GET /health`, and the legacy `GET /emotionDetector?textToAnalyse=...`. Errors return `400` with a `message`.

## Project layout

```
server.py                   Flask app + JSON API
EmotionDetection/
  emotion_detection.py      pipeline: match, distribute, calibrate, explain
  language.py               script + language identification, tokenizer
  lexicon.py                all language data (add words here)
  wellbeing.py              crisis-phrase detection + helplines
templates/index.html        dashboard (vanilla JS, no build step, no CDN)
evaluation/                 dataset.csv + evaluate.py
tests/                      pytest suite (engine, language, API, eval floors)
cloud_function/, main.py    Cloud Functions / Lambda handlers
Dockerfile, app.yaml        deployment
```

## Extending

Add words to `LEXICON[emotion][lang]` in `EmotionDetection/lexicon.py` (and a `STOPWORDS` entry to make a new Latin-script language detectable), add labelled rows to `evaluation/dataset.csv`, run `pytest`.

## Known limitations

- Lexicon coverage is uneven; sarcasm, irony, slang and long documents are weak spots.
- Negation is not handled for Japanese, Chinese, Cantonese and Thai (no word boundaries).
- Emoji can be ambiguous (😭 may be joy); they are treated as fixed signals.
- Crisis detection is phrase matching, not clinical assessment. The helpline numbers should be re-verified before production use. This app is not a diagnostic tool.
- A natural next step is an optional transformer backend (for example a multilingual emotion model) blended with the lexicon. It is not included because it could not be tested offline here.

## License

Apache-2.0 per the original lab starter (add a `LICENSE` file before publishing widely).
