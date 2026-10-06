"""Command line: python -m EmotionDetection "I just got promoted!" [--json]"""

import json
import sys

from .emotion_detection import emotion_detector


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in args
    text = " ".join(a for a in args if a != "--json") or sys.stdin.read()
    r = emotion_detector(text)
    if as_json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return 0 if r["ok"] else 1
    if not r["ok"]:
        print(r["message"])
        return 1
    print(f"{r['language']} ({r['language_confidence']:.0%})  ->  {r['primary_emotion']}  "
          f"| positivity {r['sentiment']['positivity']:.0f}%  | confidence {r['confidence']:.0%}")
    print(r["summary"])
    for e in r["evidence"]:
        print(f"  {e['text']!r:24} {e['emotion']:9} {e['weight']:+.2f}"
              + ("  negated" if e["negated"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
