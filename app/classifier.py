"""Embedding-based query classifier.

Nearest labeled example wins. Confidence is the margin between the best two
labels, not the raw similarity: what matters for routing isn't how close the
query sits to anything, it's whether one label is clearly better than the rest.

Costs nothing extra at request time — it reuses the vector the cache already
computed on the way in.
"""

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.cache import embed

EXAMPLES_PATH = Path(__file__).parent.parent / "data" / "labeled_examples.csv"

_examples: dict[str, np.ndarray] | None = None


def load_examples(path: Path = EXAMPLES_PATH) -> dict[str, np.ndarray]:
    """label -> matrix of that label's example vectors."""
    global _examples
    if _examples is None:
        by_label = defaultdict(list)
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                by_label[row["label"]].append(embed(row["question"]))
        _examples = {label: np.array(vecs) for label, vecs in by_label.items()}
    return _examples


@dataclass
class Classification:
    label: str
    confidence: float
    scores: dict[str, float]


def classify(query: str, vector: np.ndarray | None = None) -> Classification:
    vector = embed(query) if vector is None else vector
    examples = load_examples()

    # Best-matching example per label, rather than a centroid: a label can
    # cluster in several places and still be one label.
    scores = {label: float((vecs @ vector).max()) for label, vecs in examples.items()}

    ranked = sorted(scores.values(), reverse=True)
    margin = ranked[0] - ranked[1] if len(ranked) > 1 else ranked[0]
    best = max(scores, key=scores.get)

    return Classification(label=best, confidence=round(margin, 4), scores=scores)
