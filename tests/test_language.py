import pytest

from EmotionDetection.language import detect_language, tokenize


@pytest.mark.parametrize("text,code", [
    ("i want my ex back", "en"),
    ("μου λείπει ο/η πρώην μου", "el"),
    ("Estoy muy enojado contigo", "es"),
    ("Ich bin heute sehr traurig", "de"),
    ("Je suis très content", "fr"),
    ("Sono molto felice", "it"),
    ("Estou muito feliz hoje", "pt"),
    ("Jag är så glad idag", "sv"),
    ("Ben bugün çok mutluyum", "tr"),
    ("я очень рад", "ru"),
    ("Я дуже радий, і це чудово ї", "uk"),
    ("今日とても嬉しいです", "ja"),
    ("我今天很开心", "zh"),
    ("오늘 정말 행복해요", "ko"),
    ("मैं आज बहुत खुश हूँ", "hi"),
    ("אני שמח מאוד", "he"),
])
def test_language_identification(text, code):
    assert detect_language(text)[0] == code


def test_kana_beats_han_majority():
    # more Han than kana, but any kana means Japanese
    assert detect_language("今日明日嬉しい")[0] == "ja"


@pytest.mark.parametrize("text", ["asdf qwer zxcv", "zzz zebra", "", "😍", "12345"])
def test_no_evidence_is_undetermined(text):
    assert detect_language(text)[0] == "und"


def test_tokenizer_keeps_apostrophes_and_indic_marks():
    toks = [t for _, _, t in tokenize("I don't know, खुश! 'quoted'")]
    assert toks == ["I", "don't", "know", "खुश", "quoted"]
