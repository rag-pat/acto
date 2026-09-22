"""Offline eval harness for the classifier.

Runs a frozen ground-truth set through `classify` only — never the endpoint, no
Redis, no model calls. The set is held out from the labeled examples the
classifier routes against, so nothing is ever scored against its own store.

Scoring is precision-first and clinical-weighted: a clinical query that does not
come back labeled clinical would reach a model instead of a human, so it is the
worst failure the system can make and it dominates the headline number.
"""

import csv
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.cache import embed, normalize
from app.classifier import EXAMPLES_PATH, classify, load_examples

DATA = Path(__file__).parent.parent / "data"
EVAL_SET_PATH = DATA / "eval_set.csv"

CLINICAL_LABEL = "clinical"
CLINICAL_WEIGHT = 0.6  # clinical recall is more than half the headline score
UNSURE = "__unsure__"  # margin below the variant's threshold


@dataclass
class Variant:
    name: str
    examples_path: Path
    threshold: float = 0.0  # minimum margin to commit to a label


@dataclass
class Result:
    variant: Variant
    total: int
    correct: int
    per_label: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion: Counter = field(default_factory=Counter)
    errors: list[tuple[str, str, str, float]] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total

    @property
    def misses(self) -> int:
        return self.total - self.correct

    @property
    def clinical_recall(self) -> float:
        return self.per_label.get(CLINICAL_LABEL, {}).get("recall", 0.0)

    @property
    def score(self) -> float:
        """Headline number. Clinical recall outweighs everything else."""
        return CLINICAL_WEIGHT * self.clinical_recall + (1 - CLINICAL_WEIGHT) * self.accuracy


def load_eval_set(path: Path = EVAL_SET_PATH) -> list[tuple[str, str]]:
    with open(path, newline="") as f:
        return [(row["query"], row["label"]) for row in csv.DictReader(f)]


def assert_held_out(cases: list[tuple[str, str]], *example_paths: Path) -> None:
    """The eval set must share no query with any example set it scores."""
    eval_queries = {normalize(q) for q, _ in cases}
    for path in example_paths:
        with open(path, newline="") as f:
            examples = {normalize(row["question"]) for row in csv.DictReader(f)}
        overlap = eval_queries & examples
        if overlap:
            raise AssertionError(f"{path.name} overlaps the eval set: {sorted(overlap)}")


def run(variant: Variant, cases: list[tuple[str, str]], vectors: list[np.ndarray]) -> Result:
    examples = load_examples(variant.examples_path)

    predictions = []
    for (query, gold), vector in zip(cases, vectors):
        c = classify(query, vector=vector, examples=examples)
        label = c.label if c.confidence >= variant.threshold else UNSURE
        predictions.append((query, gold, label, c.confidence))

    result = Result(
        variant=variant,
        total=len(predictions),
        correct=sum(1 for _, gold, pred, _ in predictions if pred == gold),
    )

    gold_counts = Counter(gold for _, gold, _, _ in predictions)
    pred_counts = Counter(pred for _, _, pred, _ in predictions)
    hits = defaultdict(int)

    for query, gold, pred, margin in predictions:
        if pred == gold:
            hits[gold] += 1
        else:
            result.confusion[(gold, pred)] += 1
            result.errors.append((query, gold, pred, margin))

    for label in sorted(gold_counts):
        tp = hits[label]
        result.per_label[label] = {
            "precision": tp / pred_counts[label] if pred_counts[label] else 0.0,
            "recall": tp / gold_counts[label],
            "support": gold_counts[label],
        }

    return result


def report(result: Result) -> None:
    v = result.variant
    print(f"\n{'=' * 66}")
    print(f"{v.name}  ({v.examples_path.name}, threshold {v.threshold})")
    print("=" * 66)
    print(f"accuracy         {result.accuracy:.1%}  ({result.correct}/{result.total})")
    print(f"misclassified    {result.misses}")
    print(f"clinical recall  {result.clinical_recall:.1%}")
    print(f"weighted score   {result.score:.4f}")

    print(f"\n{'label':<14}{'precision':>11}{'recall':>9}{'support':>9}")
    for label, m in result.per_label.items():
        marker = "  <-- weighted" if label == CLINICAL_LABEL else ""
        print(f"{label:<14}{m['precision']:>10.1%}{m['recall']:>9.1%}{m['support']:>9}{marker}")

    if result.confusion:
        print("\nconfusion pairs (gold -> predicted)")
        for (gold, pred), n in result.confusion.most_common():
            flag = "  ** clinical missed" if gold == CLINICAL_LABEL else ""
            print(f"  {n:>3}x  {gold} -> {pred}{flag}")


def compare(a: Result, b: Result) -> None:
    print(f"\n{'=' * 66}")
    print("A/B")
    print("=" * 66)
    print(f"{'':<18}{a.variant.name:>22}{b.variant.name:>22}")
    for name, fa, fb, fmt in [
        ("accuracy", a.accuracy, b.accuracy, "{:.1%}"),
        ("misclassified", a.misses, b.misses, "{}"),
        ("clinical recall", a.clinical_recall, b.clinical_recall, "{:.1%}"),
        ("weighted score", a.score, b.score, "{:.4f}"),
    ]:
        print(f"{name:<18}{fmt.format(fa):>22}{fmt.format(fb):>22}")

    winner = max((a, b), key=lambda r: r.score)
    delta = b.misses - a.misses
    direction = "fewer" if delta < 0 else "more" if delta > 0 else "same"
    change = "unchanged" if delta == 0 else f"{abs(delta)} {direction}"
    print(f"\nmisclassified: {a.misses} -> {b.misses} ({change})")
    print(f"winner on weighted score: {winner.variant.name}")


VARIANTS = [
    Variant("v1 baseline", EXAMPLES_PATH),
    Variant("v2 expanded", DATA / "labeled_examples_v2.csv"),
]


def main(variants: list[Variant] = VARIANTS) -> list[Result]:
    cases = load_eval_set()
    assert_held_out(cases, *{v.examples_path for v in variants})

    # Embedded once, shared across every variant — the same reuse the request
    # path relies on, and it keeps the comparison honest.
    vectors = [embed(q) for q, _ in cases]

    results = [run(v, cases, vectors) for v in variants]
    for r in results:
        report(r)
    if len(results) == 2:
        compare(*results)
    return results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main([Variant(Path(p).stem, Path(p)) for p in sys.argv[1:]])
    else:
        main()
