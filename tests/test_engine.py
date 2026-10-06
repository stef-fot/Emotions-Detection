import pytest

from EmotionDetection import LABELS, emotion_detector, format_legacy


def primary(text):
    r = emotion_detector(text)
    assert r["ok"], r
    return r["primary_emotion"]


@pytest.mark.parametrize("text,expected", [
    ("I just got promoted today!", "joy"),
    ("i want my ex back", "sadness"),
    ("μου λείπει ο/η πρώην μου", "sadness"),
    ("Estoy muy enojado contigo", "anger"),
    ("今日とても嬉しいです", "joy"),
    ("This is the worst day ever", "sadness"),
    ("Ich bin sehr traurig", "sadness"),
    ("я очень рад", "joy"),
    ("Ben bugün çok mutluyum", "joy"),
    ("खुश हूँ बहुत", "joy"),
    ("오늘 정말 행복해요", "joy"),
    ("Wow, I never expected to see you here!", "surprise"),
    ("The meeting is at three o'clock", "neutral"),
])
def test_primary_emotion(text, expected):
    assert primary(text) == expected


def test_distribution_is_a_probability_distribution():
    r = emotion_detector("I am so happy but also a little scared")
    d = r["distribution"]
    assert set(d) == set(LABELS)
    assert sum(d.values()) == pytest.approx(1.0, abs=1e-3)
    assert all(0.0 <= v <= 1.0 for v in d.values())


def test_gibberish_is_neutral_and_low_confidence():
    r = emotion_detector("asdf qwer zxcv")
    assert r["primary_emotion"] == "neutral"
    assert r["confidence"] < 0.4
    assert r["language_code"] == "und"
    assert r["sentiment"]["label"] == "neutral"
    assert r["warnings"]


# --- the bugs the rewrite fixed -------------------------------------------
def test_apostrophes_survive_cleaning():
    # the old control-char regex stripped "'" so "don't" became "don t"
    r = emotion_detector("I don't feel happy")
    assert r["primary_emotion"] == "sadness"
    assert r["evidence"][0]["negated"] is True


def test_emoji_only_input_is_accepted():
    assert primary("😭") == "sadness"
    assert primary("❤️") == "joy"  # emoji written with a variation selector


def test_greek_works_without_accents_and_in_capitals():
    assert primary("χαρουμενος") == "joy"
    assert primary("ΧΑΡΟΥΜΕΝΟΣ") == "joy"


def test_greek_inflections_match():
    assert primary("Είμαι λυπημένη") == "sadness"
    assert primary("Είμαι λυπημένος") == "sadness"


def test_letter_z_is_latin():
    # 'z' was missing from the old Latin range
    assert emotion_detector("Zoe is happy")["language_code"] == "en"


# --- context handling ------------------------------------------------------
def test_negation_flips_joy_to_sadness():
    assert primary("I am not happy") == "sadness"
    assert primary("Δεν είμαι καθόλου χαρούμενος") == "sadness"


def test_negation_does_not_cross_a_comma():
    assert primary("I had no idea, that's astonishing!") == "surprise"


def test_idiom_could_not_be_happier_is_joy():
    assert primary("I could not be happier") == "joy"


def test_intensifier_raises_confidence():
    plain = emotion_detector("I am happy")["distribution"]["joy"]
    boosted = emotion_detector("I am extremely happy")["distribution"]["joy"]
    assert boosted > plain


def test_english_suffixes():
    assert primary("That food was disgusting") == "disgust"
    assert primary("She is crying") == "sadness"


# --- explainability / structure -------------------------------------------
def test_evidence_spans_point_at_the_matched_text():
    r = emotion_detector("What a wonderful day! I feel awful about the news.")
    text = r["normalized_text"]
    assert r["evidence"]
    for ev in r["evidence"]:
        assert text[ev["start"]:ev["end"]] == ev["text"]


def test_timeline_tracks_emotional_arc():
    r = emotion_detector("I am so happy today! Then it all went wrong and I cried.")
    tl = r["timeline"]
    assert len(tl) == 2
    assert tl[0]["positivity"] > tl[1]["positivity"]


def test_result_is_json_serialisable():
    import json
    json.dumps(emotion_detector("hello 😊"))


def test_rejects_empty_and_punctuation_only():
    for bad in ("", "   ", None, "!!!???"):
        r = emotion_detector(bad)
        assert r["ok"] is False and r["message"]


def test_long_input_is_truncated_not_rejected():
    r = emotion_detector("happy " * 3000)
    assert r["ok"] and r["truncated"] and r["char_count"] == 4000


def test_unknown_language_is_flagged_honestly():
    r = emotion_detector("მე ძალიან მიყვარხარ")  # Georgian: no lexicon yet
    assert r["ok"]
    assert any("No emotion lexicon" in w for w in r["warnings"])
    assert r["confidence"] <= 0.25


def test_crisis_text_surfaces_resources():
    r = emotion_detector("I want to kill myself")
    assert r["crisis"] is True
    assert r["crisis_resources"]["lines"]
    g = emotion_detector("Θέλω να πεθάνω")
    assert g["crisis"] is True and "1018" in " ".join(g["crisis_resources"]["lines"])
    assert emotion_detector("I love my life")["crisis"] is False


def test_legacy_formatter():
    assert "dominant emotion is joy" in format_legacy(emotion_detector("I am so happy"))
    assert format_legacy({"ok": False, "message": "nope"}) == "nope"


def test_performance_budget():
    r = emotion_detector("I just got promoted today! " * 20)
    assert r["elapsed_ms"] < 50
