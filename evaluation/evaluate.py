"""
Measure the emotion model on the hand-labelled multilingual set.

    python -m evaluation.evaluate                # report on every split
    python -m evaluation.evaluate --split test   # held-out split only
    python -m evaluation.evaluate --errors       # also list every mistake
    python -m evaluation.evaluate --markdown     # tables ready for the README

``dataset.csv`` has a ``dev`` split (used while building and tuning the
lexicon) and a ``test`` split (written first and never used for tuning), so
the ``test`` numbers are the honest ones. The set is small and hand-written:
treat it as a regression suite and a sanity check, not as a benchmark.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from EmotionDetection import LABELS, emotion_detector  # noqa: E402

DATASET = Path(__file__).with_name("dataset.csv")


def load(split: str):
    with DATASET.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if split == "all" or r["split"] == split]


def predict(rows):
    return [(r, emotion_detector(r["text"])) for r in rows]


def prf(confusion, label):
    tp = confusion[label][label]
    fp = sum(confusion[o][label] for o in LABELS if o != label)
    fn = sum(confusion[label][o] for o in LABELS if o != label)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f, tp + fn


def evaluate(rows):
    confusion = {a: {b: 0 for b in LABELS} for a in LABELS}
    errors, by_lang = [], defaultdict(lambda: [0, 0])
    for row, res in predict(rows):
        pred = res["primary_emotion"] if res.get("ok") else "neutral"
        confusion[row["label"]][pred] += 1
        ok = pred == row["label"]
        by_lang[row["lang"]][0] += ok
        by_lang[row["lang"]][1] += 1
        if not ok:
            errors.append((row, pred, res))
    n = len(rows)
    correct = sum(confusion[l][l] for l in LABELS)
    per = {l: prf(confusion, l) for l in LABELS}
    supported = [l for l in LABELS if per[l][3]]
    macro_f1 = sum(per[l][2] for l in supported) / len(supported)
    majority = Counter(r["label"] for r in rows).most_common(1)[0]
    return {"n": n, "accuracy": correct / n, "macro_f1": macro_f1,
            "baseline": majority[1] / n, "baseline_label": majority[0],
            "per_class": per, "confusion": confusion,
            "by_lang": dict(by_lang), "errors": errors}


def print_report(name, r, show_errors, markdown):
    print(f"\n=== {name}: {r['n']} examples ===")
    print(f"accuracy {r['accuracy']:.1%}   macro-F1 {r['macro_f1']:.3f}   "
          f"(majority-class baseline '{r['baseline_label']}': {r['baseline']:.1%})")
    if markdown:
        print("\n| Emotion | Precision | Recall | F1 | Support |\n| --- | ---: | ---: | ---: | ---: |")
        for l in LABELS:
            p, rc, f, s = r["per_class"][l]
            print(f"| {l} | {p:.2f} | {rc:.2f} | {f:.2f} | {s} |")
    else:
        print(f"\n{'':10}{'prec':>7}{'rec':>7}{'f1':>7}{'n':>5}")
        for l in LABELS:
            p, rc, f, s = r["per_class"][l]
            print(f"{l:10}{p:7.2f}{rc:7.2f}{f:7.2f}{s:5d}")
    print("\nconfusion (rows = truth, cols = prediction)")
    print(" " * 10 + "".join(f"{l[:5]:>7}" for l in LABELS))
    for a in LABELS:
        print(f"{a:10}" + "".join(f"{r['confusion'][a][b]:7d}" for b in LABELS))
    langs = sorted(r["by_lang"].items(), key=lambda kv: -kv[1][1])
    print("\nby language: " + "  ".join(f"{k} {v[0]}/{v[1]}" for k, v in langs))
    if show_errors and r["errors"]:
        print("\nmistakes:")
        for row, pred, res in r["errors"]:
            print(f"  [{row['lang']}] truth={row['label']:8} pred={pred:8} {row['text']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", choices=["dev", "test1", "holdout", "blind", "all"], default=None)
    ap.add_argument("--errors", action="store_true")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--min-accuracy", type=float, default=None,
                    help="exit 1 if accuracy on the chosen split falls below this")
    args = ap.parse_args(argv)
    splits = [args.split] if args.split else ["blind", "holdout", "test1", "dev", "all"]
    last = None
    for s in splits:
        last = evaluate(load(s))
        print_report(s, last, args.errors, args.markdown)
    if args.min_accuracy is not None and last["accuracy"] < args.min_accuracy:
        print(f"\nFAIL: accuracy {last['accuracy']:.1%} < {args.min_accuracy:.1%}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
