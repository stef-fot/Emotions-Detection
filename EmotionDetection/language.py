"""
Zero-dependency language identification.

Strategy (cheap, deterministic, explainable):

1. Count which Unicode script each letter belongs to and pick the dominant
   one (Greek, Cyrillic, Han, Arabic, ...).
2. Scripts used by exactly one language we know (Greek, Thai, Hebrew, ...)
   are done at this point.
3. Scripts shared by several languages are disambiguated with diagnostic
   letters (e.g. "ї" is Ukrainian, "ё"/"ы" Russian, "پ" Persian, kana means
   Japanese) and, for Latin script, with common function words, lexicon hits
   and diacritic hints.

It identifies a *language*, never the emotion; it is a means to choose the
right lexicon, and it reports how sure it is.
"""

from __future__ import annotations

import unicodedata
from functools import lru_cache
from typing import Dict, List, Tuple

from .lexicon import (DIACRITIC_HINTS, LANG_NAMES, LEXICON, STOPWORDS)

# (first, last, script) code point ranges. Order only matters for speed.
_RANGES: List[Tuple[int, int, str]] = [
    (0x0041, 0x005A, "Latin"), (0x0061, 0x007A, "Latin"),
    (0x00C0, 0x024F, "Latin"), (0x1E00, 0x1EFF, "Latin"),
    (0x0370, 0x03FF, "Greek"), (0x1F00, 0x1FFF, "Greek"),
    (0x0400, 0x052F, "Cyrillic"),
    (0x0530, 0x058F, "Armenian"),
    (0x0590, 0x05FF, "Hebrew"),
    (0x0600, 0x06FF, "Arabic"), (0x0750, 0x077F, "Arabic"),
    (0x08A0, 0x08FF, "Arabic"), (0xFB50, 0xFDFF, "Arabic"),
    (0xFE70, 0xFEFF, "Arabic"),
    (0x0780, 0x07BF, "Thaana"),
    (0x0900, 0x097F, "Devanagari"), (0x0980, 0x09FF, "Bengali"),
    (0x0A00, 0x0A7F, "Gurmukhi"), (0x0A80, 0x0AFF, "Gujarati"),
    (0x0B00, 0x0B7F, "Oriya"), (0x0B80, 0x0BFF, "Tamil"),
    (0x0C00, 0x0C7F, "Telugu"), (0x0C80, 0x0CFF, "Kannada"),
    (0x0D00, 0x0D7F, "Malayalam"), (0x0D80, 0x0DFF, "Sinhala"),
    (0x0E00, 0x0E7F, "Thai"), (0x0E80, 0x0EFF, "Lao"),
    (0x0F00, 0x0FFF, "Tibetan"), (0x1000, 0x109F, "Myanmar"),
    (0x10A0, 0x10FF, "Georgian"), (0x1200, 0x139F, "Ethiopic"),
    (0x13A0, 0x13FF, "Cherokee"), (0x16A0, 0x16FF, "Runic"),
    (0x1780, 0x17FF, "Khmer"),
    (0x3040, 0x30FF, "Kana"),
    (0x3400, 0x4DBF, "Han"), (0x4E00, 0x9FFF, "Han"), (0xF900, 0xFAFF, "Han"),
    (0x20000, 0x2A6DF, "Han"),
    (0x1100, 0x11FF, "Hangul"), (0x3130, 0x318F, "Hangul"),
    (0xAC00, 0xD7AF, "Hangul"),
]

SCRIPT_TO_LANG: Dict[str, str] = {
    "Latin": "en", "Cyrillic": "ru", "Greek": "el", "Arabic": "ar",
    "Hebrew": "he", "Devanagari": "hi", "Bengali": "bn", "Tamil": "ta",
    "Telugu": "te", "Gujarati": "gu", "Gurmukhi": "pa", "Thai": "th",
    "Han": "zh", "Kana": "ja", "Hangul": "ko", "Georgian": "ka",
    "Armenian": "hy", "Ethiopic": "am", "Khmer": "km", "Lao": "lo",
    "Myanmar": "my", "Tibetan": "bo", "Sinhala": "si", "Malayalam": "ml",
    "Kannada": "kn", "Oriya": "or", "Cherokee": "chr", "Runic": "non",
    "Thaana": "dv",
}

# Diagnostic letters for scripts shared by several languages.
_CYRILLIC_HINTS = {
    "uk": "іїєґ", "sr": "ђјљњћџ", "ru": "ыэё", "bg": "ъщ", "be": "ў",
}
_ARABIC_HINTS = {"fa": "پچژگکی", "ur": "ٹڈڑںےھ"}
_DEVANAGARI_HINTS = {"mr": "ळ"}
_YUE_CHARS = set("嘅咗冇喺哋啲嚟唔係咁嘢乜")

_APOSTROPHES = ("'", "’", "ʼ")

_RANGE_TABLE = sorted(_RANGES)


def _is_word_char(ch: str) -> bool:
    # Combining marks (Mn/Mc) belong to the word: Hindi "खुश" or Thai vowels
    # would otherwise be split into meaningless fragments.
    return ch.isalpha() or unicodedata.category(ch)[0] == "M"


def tokenize(text: str) -> List[Tuple[int, int, str]]:
    """Split ``text`` into ``(start, end, surface)`` word tokens.

    Letters (plus combining marks) form words; an apostrophe is kept only
    *inside* a word ("don't", "j'aime"). Digits and punctuation are skipped.
    """
    tokens: List[Tuple[int, int, str]] = []
    n = len(text)
    i = 0
    while i < n:
        if not _is_word_char(text[i]):
            i += 1
            continue
        j = i + 1
        while j < n:
            c = text[j]
            if _is_word_char(c):
                j += 1
            elif c in _APOSTROPHES and j + 1 < n and text[j + 1].isalpha():
                j += 1
            else:
                break
        tokens.append((i, j, text[i:j]))
        i = j
    return tokens


@lru_cache(maxsize=4096)
def _script_of_cp(cp: int) -> str:
    for lo, hi, name in _RANGE_TABLE:
        if lo <= cp <= hi:
            return name
    return ""


def script_of(ch: str) -> str:
    """Script name of a letter, or '' for punctuation / digits / emoji."""
    cp = ord(ch)
    if cp < 0x41:
        return ""
    s = _script_of_cp(cp)
    if s == "Latin" and not ch.isalpha():
        return ""
    return s


def fold(text: str) -> str:
    """Case-fold and strip accents/diacritics (used for matching, never shown)."""
    decomposed = unicodedata.normalize("NFD", text.casefold())
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", stripped)


@lru_cache(maxsize=1)
def _latin_lexicon_words() -> Dict[str, Dict[str, set]]:
    """lang -> single-token lexicon words (lower-cased), for language ID."""
    out: Dict[str, set] = {}
    for by_lang in LEXICON.values():
        for lang, words in by_lang.items():
            bucket = out.setdefault(lang, set())
            for w in words:
                w = w.casefold().replace("'", "").replace("’", "")
                if " " not in w and len(w) >= 3:
                    bucket.add(w)
    return out


@lru_cache(maxsize=1)
def _stopword_sets() -> Dict[str, set]:
    return {lang: {w.casefold() for w in words if len(w) > 1 or w in ("i", "y")}
            for lang, words in STOPWORDS.items()}


def _identify_latin(text: str) -> Tuple[str, float]:
    tokens = [t.casefold().replace("'", "").replace("’", "")
              for _, _, t in tokenize(text)]
    scores: Dict[str, float] = {}
    stop = _stopword_sets()
    lex = _latin_lexicon_words()
    for tok in set(tokens):
        for lang, words in stop.items():
            if tok in words:
                scores[lang] = scores.get(lang, 0.0) + 1.0
        for lang, words in lex.items():
            if lang in stop and tok in words:
                scores[lang] = scores.get(lang, 0.0) + 1.5
    for ch in text.casefold():
        for lang in DIACRITIC_HINTS.get(ch, ()):
            scores[lang] = scores.get(lang, 0.0) + 0.8
    if not scores:
        return ("und", 0.0)
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0] != "en"))
    best, best_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    if best_score < 1.0:
        return ("und", 0.0)
    return (best, best_score / (best_score + runner_up + 0.5))


def _pick_by_hints(text: str, hints: Dict[str, str], default: str) -> str:
    lowered = text.casefold()
    best, best_n = default, 0
    for lang, chars in hints.items():
        n = sum(lowered.count(c) for c in chars)
        if n > best_n:
            best, best_n = lang, n
    return best


def detect_language(text: str) -> Tuple[str, str, float]:
    """Return ``(code, name, confidence)`` for ``text``.

    ``code`` is ``"und"`` when the text has no evidence for any language
    (for example ``"asdf qwer"`` or a lone emoji).
    """
    counts: Dict[str, int] = {}
    for ch in text:
        s = script_of(ch)
        if s:
            counts[s] = counts.get(s, 0) + 1
    if not counts:
        return ("und", LANG_NAMES["und"], 0.0)

    # Any kana means Japanese, even when Han characters outnumber it.
    if counts.get("Kana"):
        return ("ja", LANG_NAMES["ja"], 0.95)

    total = sum(counts.values())
    primary, n = max(counts.items(), key=lambda kv: kv[1])
    confidence = n / total

    if primary == "Latin":
        code, conf = _identify_latin(text)
        if code == "und":
            return ("und", "Undetermined (Latin script)", 0.0)
        return (code, LANG_NAMES.get(code, code), round(conf * confidence, 3))

    code = SCRIPT_TO_LANG.get(primary, "und")
    if primary == "Cyrillic":
        code = _pick_by_hints(text, _CYRILLIC_HINTS, "ru")
    elif primary == "Arabic":
        code = _pick_by_hints(text, _ARABIC_HINTS, "ar")
    elif primary == "Devanagari":
        code = _pick_by_hints(text, _DEVANAGARI_HINTS, "hi")
    elif primary == "Han" and any(c in _YUE_CHARS for c in text):
        code = "yue"
    return (code, LANG_NAMES.get(code, primary), round(0.6 + 0.4 * confidence, 3))
