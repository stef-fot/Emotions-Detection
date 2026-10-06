"""
Crisis-safety layer.

If a text contains a self-harm / suicide signal, the analysis result carries
``crisis: true`` plus a set of helpline resources so the UI can show them
immediately. The check is plain pattern matching (no model, no network), so
it is deterministic and always runs, regardless of what any other part of
the pipeline concludes.

Keep the pattern list conservative in the "safe" direction: a helpline
banner shown to someone who did not need it costs little, a missed signal
costs a lot.

Helpline numbers change. They were last reviewed when this module was
written; re-check them before relying on this in production.
"""

from __future__ import annotations

import re
from typing import Dict, List

from .language import fold

CRISIS_RESOURCES: Dict[str, Dict[str, object]] = {
    "el": {
        "intro": "Αν σκέφτεσαι να βλάψεις τον εαυτό σου ή διατρέχεις κίνδυνο, σε παρακαλώ μίλησε με κάποιον αμέσως. Δεν χρειάζεται να το αντιμετωπίσεις μόνος/η σου:",
        "lines": [
            "Γραμμή Παρέμβασης για την Αυτοκτονία (ΚΛΙΜΑΚΑ): 1018, 24/7, δωρεάν, εμπιστευτικά",
            "Γραμμή Ψυχοκοινωνικής Υποστήριξης: 10306, 24/7, δωρεάν",
            "Ευρωπαϊκός αριθμός έκτακτης ανάγκης: 112",
        ],
    },
    "en": {
        "intro": "If you're thinking about harming yourself or you're in danger right now, please reach out immediately. You don't have to go through this alone:",
        "lines": [
            "US / Canada: call or text 988 (Suicide & Crisis Lifeline), 24/7",
            "UK / ROI: call 116 123 (Samaritans), 24/7, free",
            "Elsewhere: contact your local emergency number, or find a helpline near you at befrienders.org",
        ],
    },
    "es": {
        "intro": "Si estás pensando en hacerte daño o corres peligro ahora mismo, por favor busca ayuda de inmediato. No tienes que pasar por esto solo/a:",
        "lines": [
            "España: llama al 024 (Línea de Atención a la Conducta Suicida), 24/7, gratuita y confidencial",
            "Emergencias: 112",
            "Otro país: busca una línea de ayuda cercana en befrienders.org",
        ],
    },
    "default": {
        "intro": "If you're thinking about harming yourself or you're in danger right now, please reach out immediately. You don't have to go through this alone:",
        "lines": [
            "Contact your local emergency number right away",
            "Find a crisis line near you at befrienders.org (international directory)",
        ],
    },
}

# Deliberately broad, common phrases (not an exhaustive clinical list).
# Matched against accent-folded, lower-cased text, so write them unaccented.
_CRISIS_PATTERNS: Dict[str, List[str]] = {
    "en": [
        r"kill myself", r"killing myself", r"end my life", r"suicid\w*",
        r"want to die", r"wanna die", r"don'?t want to (live|be alive)",
        r"no reason to live", r"self.?harm", r"hurt myself", r"cut myself",
        r"better off dead", r"end it all", r"take my own life",
    ],
    "el": [
        r"αυτοκτον\w*", r"θελω να πεθανω", r"δεν θελω να ζω",
        r"να τελειωσω τη ζωη μου", r"να κανω κακο στον εαυτο μου",
        r"να χτυπησω τον εαυτο μου", r"θελω να εξαφανιστω",
    ],
    "es": [
        r"suicid\w*", r"quiero morir", r"no quiero vivir",
        r"acabar con mi vida", r"hacerme dano", r"lastimarme", r"quitarme la vida",
    ],
    "fr": [r"suicid\w*", r"me tuer", r"veux mourir", r"mettre fin a mes jours"],
    "de": [r"suizid\w*", r"selbstmord", r"mich umbringen", r"will sterben"],
    "it": [r"suicid\w*", r"uccidermi", r"voglio morire", r"togliermi la vita"],
    "pt": [r"suicid\w*", r"me matar", r"quero morrer", r"tirar a minha vida"],
}
_CRISIS_RE = {
    lang: re.compile("|".join(pats), re.IGNORECASE | re.UNICODE)
    for lang, pats in _CRISIS_PATTERNS.items()
}


def detect_crisis(text: str, lang_code: str = "en") -> bool:
    """True if ``text`` matches a crisis phrase in its own language, or in
    English (crisis phrases are often typed in English regardless of the
    surrounding language)."""
    if not text:
        return False
    folded = fold(text)
    for lang in {lang_code, "en"}:
        rx = _CRISIS_RE.get(lang)
        if rx and rx.search(folded):
            return True
    return False


def crisis_resources_for(lang_code: str) -> Dict[str, object]:
    return CRISIS_RESOURCES.get(lang_code, CRISIS_RESOURCES["default"])
