"""
Multilingual, explainable emotion & sentiment analyzer.

Pipeline (everything runs in-process, no network, typically ~1 ms):

  1. clean         strip control chars / URLs, collapse absurd repetition
  2. language      script + function-word language identification
  3. segment       split into sentences (kept for the emotional-arc timeline)
  4. match         lexicon lookup per sentence with
                     - phrase and single-word matching on accent-folded tokens
                     - light stemming for inflected / agglutinative languages
                     - negation ("not happy") and intensifiers ("very happy")
                     - emoji, ELONGATED words, ALL-CAPS and "!" emphasis
  5. distribute    evidence counts -> probability distribution over
                   joy, sadness, anger, fear, disgust, surprise + neutral
  6. explain       every match is returned as ``evidence`` with its exact
                   character span, so the UI can show *why* a label was chosen
  7. calibrate     confidence from evidence strength and top-1 / top-2 margin;
                   honest warnings when a language has no lexicon

This is a transparent lexicon model, not a neural network: it is fast,
offline, fully inspectable and deterministic. See ``eval/`` for its measured
accuracy on a small hand-labelled multilingual set.
"""

from __future__ import annotations

import math
import re
import time
import unicodedata
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

from .language import detect_language, script_of, fold, tokenize
from .lexicon import (AGGLUTINATIVE_LANGS, EMOJI_EMOTIONS, GREETINGS,
                      INFLECTED_LANGS, INTENSIFIERS, LEXICON, NEGATIONS,
                      NO_SPACE_LANGS)
from .wellbeing import crisis_resources_for, detect_crisis

__all__ = ["emotion_detector", "format_legacy", "EMOTIONS", "LABELS",
           "MODEL_NAME", "MODEL_VERSION"]

MODEL_NAME = "multilingual-lexicon"
MODEL_VERSION = "3.0"

MAX_INPUT_CHARS = 4000
MAX_TIMELINE = 60
MAX_EVIDENCE = 120

EMOTIONS: Tuple[str, ...] = ("joy", "sadness", "anger", "fear", "disgust", "surprise")
LABELS: Tuple[str, ...] = EMOTIONS + ("neutral",)

# How positive each label is on a -1..+1 scale (used for the sentiment gauge).
SENTIMENT_WEIGHTS: Dict[str, float] = {
    "joy": 1.00, "surprise": 0.10, "neutral": 0.00,
    "fear": -0.30, "sadness": -0.70, "anger": -0.85, "disgust": -0.80,
}

# --- model constants (documented in the README, tuned on eval/dataset.csv) ---
HIT_WEIGHT = 1.5          # one lexicon match
EMOJI_WEIGHT = 2.0        # emoji are an unambiguous, strong signal
INTENSIFIER_BOOST = 1.6   # "very happy"
ELONGATION_BOOST = 1.3    # "sooo happy"
CAPS_BOOST = 1.25         # "HAPPY"
BANG_BOOST = 1.15         # sentence ends with "!"
FALLBACK_FACTOR = 0.4     # English words inside non-English text count less
NEGATION_REDIRECT = 0.8   # "not happy" -> sadness at 80% weight
NEGATION_WINDOW = 3       # tokens looked back for a negator / intensifier
NEUTRAL_PRIOR = 0.8       # pseudo-count of "nothing emotional here"
CLASS_PRIOR = 0.05
SHARPEN = 1.3

# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
_CONTROL = re.compile(
    "[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f​-‏ -‮⁠﻿]")
_URL = re.compile(r"(?:https?://|www\.)\S+|\b[\w.+-]+@[\w-]+\.[\w.-]+", re.IGNORECASE)
_WS = re.compile(r"[^\S\n]+")
_REPEATED = re.compile(r"(.)\1{6,}", re.DOTALL)
_ELONGATED = re.compile(
    r"([A-Za-zÀ-ɏͰ-ϿЀ-ӿ])\1{2,}")
_VS16 = "️"


def _clean_text(text) -> str:
    if text is None:
        return ""
    t = unicodedata.normalize("NFC", str(text))
    t = _CONTROL.sub(" ", t)
    t = _URL.sub(" ", t)
    t = _WS.sub(" ", t)
    t = re.sub(r" ?\n ?", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return _REPEATED.sub(lambda m: m.group(1) * 3, t)


# ---------------------------------------------------------------------------
# Compiled lexicon indexes (built once at import)
# ---------------------------------------------------------------------------
_FOLDABLE_SCRIPTS = {"Latin", "Greek", "Cyrillic"}


@lru_cache(maxsize=65536)
def _key(token: str) -> str:
    """Canonical matching form of a token: no apostrophes, case/accent-folded
    for Latin/Greek/Cyrillic (so "Χαρούμενος", "χαρουμενος" and "ΧΑΡΟΥΜΕΝΟΣ"
    are the same), merely case-folded for every other script."""
    for a in ("'", "’", "ʼ"):
        token = token.replace(a, "")
    if not token:
        return token
    if script_of(token[0]) in _FOLDABLE_SCRIPTS:
        return fold(token)
    return token.casefold()


_EMOJI = {k.replace(_VS16, ""): v for k, v in EMOJI_EMOTIONS.items()}

# per language: single words, phrases (first key -> [(keys, emotions)]),
# substrings (no-space scripts), light-stem tables
_WORDS: Dict[str, Dict[str, List[str]]] = {}
_PHRASES: Dict[str, Dict[str, List[Tuple[Tuple[str, ...], List[str]]]]] = {}
_SUBSTR: Dict[str, List[Tuple[str, List[str]]]] = {}
_STEMS: Dict[str, Dict[str, List[str]]] = {}
_NEG: Dict[str, set] = {}
_INT: Dict[str, set] = {}


def _add(table: Dict[str, List[str]], key: str, emotion: str) -> None:
    bucket = table.setdefault(key, [])
    if emotion not in bucket:
        bucket.append(emotion)


def _build_indexes() -> None:
    substr_tmp: Dict[str, Dict[str, List[str]]] = {}
    phr_tmp: Dict[str, Dict[Tuple[str, ...], List[str]]] = {}
    for emotion, by_lang in LEXICON.items():
        for lang, words in by_lang.items():
            for w in words:
                if lang in NO_SPACE_LANGS:
                    s = unicodedata.normalize("NFC", w).casefold().replace(" ", "")
                    if s:
                        _add(substr_tmp.setdefault(lang, {}), s, emotion)
                    continue
                keys = tuple(_key(t) for _, _, t in tokenize(w))
                if not keys:
                    continue
                if len(keys) == 1:
                    _add(_WORDS.setdefault(lang, {}), keys[0], emotion)
                else:
                    _add(phr_tmp.setdefault(lang, {}), keys, emotion)
    for lang, table in substr_tmp.items():
        _SUBSTR[lang] = sorted(table.items(), key=lambda kv: -len(kv[0]))
    for lang, table in phr_tmp.items():
        idx: Dict[str, List[Tuple[Tuple[str, ...], List[str]]]] = {}
        for keys, emos in table.items():
            idx.setdefault(keys[0], []).append((keys, emos))
        for lst in idx.values():
            lst.sort(key=lambda kv: -len(kv[0]))
        _PHRASES[lang] = idx
    for lang in INFLECTED_LANGS:
        stems: Dict[str, List[str]] = {}
        for word, emos in _WORDS.get(lang, {}).items():
            if len(word) >= 6:
                for e in emos:
                    _add(stems, word[:-2], e)
        _STEMS[lang] = stems
    for lang, words in NEGATIONS.items():
        _NEG[lang] = {_key(w) for w in words if " " not in w}
    for lang, words in INTENSIFIERS.items():
        _INT[lang] = {_key(w) for w in words if " " not in w}


_build_indexes()
LEXICON_LANGS = frozenset(_WORDS) | frozenset(_SUBSTR) | frozenset(_PHRASES)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
_EN_SUFFIXES = (("ing", ("", "e")), ("ed", ("", "e")), ("es", ("",)),
                ("s", ("",)), ("ly", ("",)))


def _single_variants(tok_key: str, lang: str) -> List[str]:
    """Candidate lookup keys for one token: the key itself, de-elongated
    forms ("goooood" -> "good", "god"), and for English a light suffix strip
    ("disgusting" -> "disgust", "loving" -> "love", "hates" -> "hate")."""
    out = [tok_key]
    for repl in (r"\1\1", r"\1"):
        v = re.sub(r"(\w)\1{2,}", repl, tok_key)
        if v not in out:
            out.append(v)
    if lang == "en":
        for suffix, restore in _EN_SUFFIXES:
            if tok_key.endswith(suffix) and len(tok_key) - len(suffix) >= 3:
                stem = tok_key[:-len(suffix)]
                for add in restore:
                    for cand in (stem + add, stem[:-1] if len(stem) > 3 and stem[-1] == stem[-2] else None):
                        if cand and cand not in out:
                            out.append(cand)
    return out


def _lookup(lang: str, keys: Sequence[str], i: int,
            variants: Sequence[str]) -> Optional[Tuple[int, List[str]]]:
    """Longest lexicon match starting at token ``i``: (n_tokens, emotions)."""
    first = keys[i]
    # Exact phrase match, then exact / de-elongated / suffix-stripped word.
    for keys_tuple, emos in _PHRASES.get(lang, {}).get(first, ()):
        n = len(keys_tuple)
        if tuple(keys[i:i + n]) == keys_tuple:
            return (n, emos)
    words = _WORDS.get(lang)
    if not words:
        return None
    for v in variants:
        emos = words.get(v)
        if emos:
            return (1, emos)
    # An intensifier ("απίστευτα" = incredibly) must not match an emotion
    # word of the same stem ("απίστευτο" = unbelievable) by inflection.
    if first in _INT.get(lang, ()):
        return None
    stems = _STEMS.get(lang)
    if stems and len(first) >= 6:
        for cut in range(0, 4):
            k = len(first) - cut
            if k < 4:
                break
            emos = stems.get(first[:k])
            if emos:
                return (1, emos)
    min_stem = AGGLUTINATIVE_LANGS.get(lang)
    if min_stem and len(first) > min_stem:
        for k in range(len(first) - 1, min_stem - 1, -1):
            emos = words.get(first[:k])
            if emos:
                return (1, emos)
    return None


_CLAUSE_BREAK = re.compile(r"[,;:()\[\]\u2014\u2013\u2026\u3001\uff0c]")


def _context_window(seg: str, toks, keys: Sequence[str], i: int) -> List[str]:
    """Up to NEGATION_WINDOW tokens before token ``i``, stopping at a clause
    boundary: in "I had no idea, that's astonishing" the "no" must not
    negate "astonishing"."""
    window: List[str] = []
    for j in range(i - 1, max(-1, i - 1 - NEGATION_WINDOW), -1):
        if _CLAUSE_BREAK.search(seg, toks[j][1], toks[j + 1][0]):
            break
        window.append(keys[j])
    return window


def _new_counts() -> Dict[str, float]:
    return {e: 0.0 for e in EMOTIONS}


def _score_segment(seg: str, base: int, lang: str, is_all_caps: bool
                   ) -> Tuple[Dict[str, float], List[dict]]:
    counts = _new_counts()
    evidence: List[dict] = []
    bang = seg.rstrip().endswith(("!", "！"))

    def record(start, end, surface, emotion, weight, source,
               negated=False, intensified=False, note=None):
        counts[emotion] += weight
        ev = {"start": base + start, "end": base + end, "text": surface,
              "emotion": emotion, "weight": round(weight, 2),
              "source": source, "negated": negated, "intensified": intensified}
        if note:
            ev["note"] = note
        evidence.append(ev)

    # --- emoji (unambiguous, strong) ---
    emoji_total: Dict[str, float] = {}
    for idx, ch in enumerate(seg):
        emo = _EMOJI.get(ch)
        if not emo or emoji_total.get(emo, 0.0) >= 6.0:
            continue
        w = EMOJI_WEIGHT * (BANG_BOOST if bang else 1.0)
        emoji_total[emo] = emoji_total.get(emo, 0.0) + w
        end = idx + 2 if seg[idx + 1:idx + 2] == _VS16 else idx + 1
        record(idx, end, seg[idx:end], emo, w, "emoji")

    # --- no-space scripts: substring matching ---
    if lang in NO_SPACE_LANGS:
        s = unicodedata.normalize("NFC", seg).casefold()
        if len(s) == len(seg):
            used = [False] * len(s)
            for entry, emos in _SUBSTR.get(lang, ()):
                start = 0
                while True:
                    idx = s.find(entry, start)
                    if idx < 0:
                        break
                    start = idx + 1
                    if any(used[idx:idx + len(entry)]):
                        continue
                    for j in range(idx, idx + len(entry)):
                        used[j] = True
                    share = HIT_WEIGHT / len(emos) * (BANG_BOOST if bang else 1.0)
                    for e in emos:
                        record(idx, idx + len(entry), seg[idx:idx + len(entry)],
                               e, share, "lexicon")

    # --- space-delimited scripts: token matching ---
    toks = tokenize(seg)
    keys = [_key(t) for _, _, t in toks]
    langs_to_try: List[Tuple[str, str, float]] = []
    if lang not in NO_SPACE_LANGS and lang in LEXICON_LANGS:
        langs_to_try.append((lang, "lexicon", 1.0))
    if lang != "en":
        langs_to_try.append(("en", "english-fallback", FALLBACK_FACTOR))

    i = 0
    while i < len(toks):
        s0, e0, surface = toks[i]
        hit, used_lang, source, factor = None, None, None, 1.0
        for l, src, fac in langs_to_try:
            hit = _lookup(l, keys, i, _single_variants(keys[i], l))
            if hit:
                used_lang, source, factor = l, src, fac
                break
        if not hit:
            i += 1
            continue
        n, emos = hit
        window = _context_window(seg, toks, keys, i)
        negated = any(k in _NEG.get(used_lang, ()) for k in window)
        intensified = any(k in _INT.get(used_lang, ()) for k in window)
        elongated = bool(_ELONGATED.search(surface))
        shout = (len(surface) >= 3 and surface.isupper() and not is_all_caps)

        w = HIT_WEIGHT * factor / len(emos)
        if intensified:
            w *= INTENSIFIER_BOOST
        if elongated:
            w *= ELONGATION_BOOST
        if shout:
            w *= CAPS_BOOST
        if bang:
            w *= BANG_BOOST
        end = toks[i + n - 1][1]
        text_span = seg[s0:end]
        for e in emos:
            if negated:
                if e == "joy":
                    record(s0, end, text_span, "sadness", w * NEGATION_REDIRECT,
                           source, negated=True, intensified=intensified,
                           note="negated joy counts as sadness")
                else:
                    evidence.append({
                        "start": base + s0, "end": base + end, "text": text_span,
                        "emotion": e, "weight": 0.0, "source": source,
                        "negated": True, "intensified": intensified,
                        "note": "negated, signal cancelled"})
            else:
                record(s0, end, text_span, e, w, source, intensified=intensified)
        i += n
    return counts, evidence


# ---------------------------------------------------------------------------
# Distribution, sentiment, confidence
# ---------------------------------------------------------------------------
def _distribution(counts: Dict[str, float]) -> Dict[str, float]:
    raw = {e: (counts.get(e, 0.0) + CLASS_PRIOR) ** SHARPEN for e in EMOTIONS}
    raw["neutral"] = NEUTRAL_PRIOR ** SHARPEN
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


def _sentiment(distribution: Dict[str, float]) -> Dict[str, object]:
    score = sum(SENTIMENT_WEIGHTS[e] * p for e, p in distribution.items())
    positivity = max(0.0, min(1.0, (score + 1.0) / 2.0)) * 100.0
    label = ("negative" if positivity < 40 else
             "positive" if positivity > 60 else "neutral")
    return {"score": score, "positivity": positivity, "label": label}


def _confidence(evidence_total: float, distribution: Dict[str, float],
                lang_confidence: float, lexicon_available: bool) -> float:
    """How much to trust the *primary label*, in [0, 0.99].

    Grows with the amount of lexical evidence and with the margin between the
    top two classes (mixed emotions are ambiguous by definition), and is
    discounted when the language is uncertain or has no lexicon."""
    ranked = sorted(distribution.values(), reverse=True)
    margin = ranked[0] - ranked[1]
    if evidence_total <= 0:
        conf = 0.30  # "nothing found" is a weak claim: the lexicon may just be missing the words
    else:
        strength = math.sqrt(1.0 - math.exp(-evidence_total / 2.5))
        conf = 0.99 * strength * min(1.0, 0.4 + 1.2 * margin)
    conf *= 0.85 + 0.15 * max(0.0, min(1.0, lang_confidence))
    if not lexicon_available:
        conf = min(conf, 0.25)
    return max(0.05, min(0.99, conf))


def _pct(p: float) -> int:
    return int(round(p * 100))


def _summary(distribution: Dict[str, float], evidence_total: float,
             sentiment: Dict[str, object], lang_name: str) -> str:
    if evidence_total <= 0:
        return (f"No clear emotional wording found in this {lang_name} text, "
                "so it reads as neutral.")
    ranked = sorted(((k, v) for k, v in distribution.items() if k != "neutral"),
                    key=lambda kv: -kv[1])
    (top, p1), (second, p2) = ranked[0], ranked[1]
    neutral = distribution["neutral"]
    if neutral > p1:
        head = f"Mostly neutral ({_pct(neutral)}%), with a touch of {top} ({_pct(p1)}%)"
    elif p2 >= 0.18:
        head = f"Mainly {top} ({_pct(p1)}%), mixed with {second} ({_pct(p2)}%)"
    else:
        head = f"Mainly {top} ({_pct(p1)}%)"
    return f"{head}. Overall tone: {sentiment['label']}."


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------
_SENTENCE = re.compile(r"[^.!?…。！？\n]*[^.!?…。！？\n\s][^.!?…。！？\n]*[.!?…。！？]*")


def _segments(text: str) -> List[Tuple[int, str]]:
    return [(m.start(), m.group()) for m in _SENTENCE.finditer(text)
            if any(c.isalnum() or c in _EMOJI for c in m.group())]


def _has_content(text: str) -> bool:
    return any(c.isalnum() or c in _EMOJI for c in text)


def _warnings(lang_code: str, lang_name: str, truncated: bool) -> List[str]:
    out = []
    if lang_code in NO_SPACE_LANGS:
        out.append(f"Negation and intensifiers are not handled for {lang_name}.")
    if lang_code != "und" and lang_code not in LEXICON_LANGS:
        out.append(f"No emotion lexicon for {lang_name} yet: only emoji and "
                   "English words were analysed, so treat the result as low confidence.")
    if lang_code == "und":
        out.append("Language could not be determined; English vocabulary was used.")
    if truncated:
        out.append(f"Input was truncated to {MAX_INPUT_CHARS} characters.")
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def emotion_detector(text_to_analyse) -> dict:
    """Analyse ``text_to_analyse`` and return a JSON-serialisable dict.

    On success ``ok`` is True and the payload contains the language, the
    7-way probability ``distribution``, ``sentiment``, ``confidence``, a
    sentence-level ``timeline``, and the matched ``evidence`` with character
    spans. On bad input ``ok`` is False with a human-readable ``message``.
    """
    t0 = time.perf_counter()

    def elapsed() -> float:
        return (time.perf_counter() - t0) * 1000.0

    text = _clean_text(text_to_analyse)
    if not text:
        return {"ok": False, "elapsed_ms": round(elapsed(), 1),
                "message": "Please enter some text to analyze, even a single sentence is fine."}
    if not _has_content(text):
        return {"ok": False, "elapsed_ms": round(elapsed(), 1),
                "message": "That doesn't look like text yet. Try a sentence in any language."}
    truncated = len(text) > MAX_INPUT_CHARS
    if truncated:
        text = text[:MAX_INPUT_CHARS]

    lang_code, lang_name, lang_conf = detect_language(text)
    scoring_lang = "en" if lang_code == "und" else lang_code
    cased = [c for c in text if c.lower() != c.upper()]
    all_caps = len(cased) > 3 and all(c.isupper() for c in cased)

    totals = _new_counts()
    evidence: List[dict] = []
    timeline: List[dict] = []
    for start, seg in _segments(text):
        counts, ev = _score_segment(seg, start, scoring_lang, all_caps)
        for e, v in counts.items():
            totals[e] += v
        evidence.extend(ev)
        if len(timeline) < MAX_TIMELINE:
            seg_dist = _distribution(counts)
            seg_sent = _sentiment(seg_dist)
            timeline.append({
                "text": seg.strip(), "start": start, "end": start + len(seg),
                "primary_emotion": max(seg_dist.items(), key=lambda kv: kv[1])[0],
                "positivity": round(seg_sent["positivity"], 1),
                "evidence_count": sum(1 for x in ev if x["weight"] > 0),
            })

    evidence.sort(key=lambda x: (x["start"], x["emotion"]))
    evidence_total = sum(totals.values())
    distribution = _distribution(totals)
    primary = max(distribution.items(), key=lambda kv: kv[1])[0]
    sentiment = _sentiment(distribution)
    lexicon_available = scoring_lang in LEXICON_LANGS or evidence_total > 0
    confidence = _confidence(evidence_total, distribution, lang_conf, lexicon_available)

    crisis = detect_crisis(text, lang_code)
    return {
        "ok": True,
        "message": None,
        "language": lang_name,
        "language_code": lang_code,
        "language_confidence": round(lang_conf, 3),
        "greeting": GREETINGS.get(lang_code, "Hello!"),
        "primary_emotion": primary,
        "distribution": {k: round(v, 4) for k, v in distribution.items()},
        "distribution_percent": {k: round(v * 100.0, 1) for k, v in distribution.items()},
        "sentiment": {
            "score": round(sentiment["score"], 3),
            "positivity": round(sentiment["positivity"], 1),
            "label": sentiment["label"],
        },
        "confidence": round(confidence, 3),
        "summary": _summary(distribution, evidence_total, sentiment, lang_name),
        "evidence": evidence[:MAX_EVIDENCE],
        "evidence_total": round(evidence_total, 2),
        "timeline": timeline,
        "warnings": _warnings(lang_code, lang_name, truncated),
        "crisis": crisis,
        "crisis_resources": crisis_resources_for(lang_code) if crisis else None,
        "model": {"name": MODEL_NAME, "version": MODEL_VERSION,
                  "lexicon_languages": len(LEXICON_LANGS)},
        "elapsed_ms": round(elapsed(), 2),
        "normalized_text": text,
        "word_count": len(tokenize(text)),
        "char_count": len(text),
        "truncated": truncated,
    }


def format_legacy(result: dict) -> str:
    """The original Watson-lab one-line report (kept for the lab's tests)."""
    if not result.get("ok"):
        return result.get("message") or "Invalid text! Please try again."
    d = result["distribution_percent"]
    return (
        "For the given statement, the system response is "
        f"'anger': {d['anger']}, 'disgust': {d['disgust']}, 'fear': {d['fear']}, "
        f"'joy': {d['joy']} and 'sadness': {d['sadness']}. "
        f"The dominant emotion is {result['primary_emotion']}."
    )
