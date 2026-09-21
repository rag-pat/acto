"""Semantic response cache.

Every query is embedded and compared against the cached entries by cosine
similarity. There is deliberately no exact-string lookup: the same question
asked different ways should land on the same entry.
"""

import csv
import hashlib
import json
import re
import sys
from dataclasses import dataclass

import numpy as np
import redis
from sentence_transformers import SentenceTransformer

# Local stand-in. Production pick is BAAI/bge-large-en-v1.5, self-hosted.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Model-specific. MiniLM puts paraphrases near 0.75; bge-large sits nearer 0.92.
# Recalibrate whenever the embedding model changes.
THRESHOLD = 0.75

KEY_PREFIX = "acto:cache:"

_model: SentenceTransformer | None = None
_redis: redis.Redis | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_redis() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.Redis(host="localhost", port=6379, decode_responses=True)
    return _redis


@dataclass
class CacheHit:
    answer: str
    score: float
    question: str
    source: str


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text.rstrip("?.!")


def embed(text: str) -> np.ndarray:
    """L2-normalized, so cosine similarity is just a dot product."""
    return _get_model().encode(normalize(text), normalize_embeddings=True)


def add(question: str, answer: str, source: str = "sheet", ttl: int | None = None) -> str:
    """Store one entry. Harvested entries pass a ttl so they expire nightly."""
    normalized = normalize(question)
    key = KEY_PREFIX + hashlib.sha256(normalized.encode()).hexdigest()[:16]
    entry = {
        "question": normalized,
        "answer": answer,
        "source": source,
        "embedding": embed(question).tolist(),
    }
    _get_redis().set(key, json.dumps(entry), ex=ttl)
    return key


def entries() -> list[dict]:
    r = _get_redis()
    keys = list(r.scan_iter(match=KEY_PREFIX + "*"))
    if not keys:
        return []
    return [json.loads(raw) for raw in r.mget(keys) if raw]


def lookup(
    query: str,
    vector: np.ndarray | None = None,
    threshold: float = THRESHOLD,
) -> CacheHit | None:
    cached = entries()
    if not cached:
        return None

    vectors = np.array([e["embedding"] for e in cached])
    scores = vectors @ (embed(query) if vector is None else vector)

    best = int(np.argmax(scores))
    if scores[best] < threshold:
        return None

    entry = cached[best]
    return CacheHit(
        answer=entry["answer"],
        score=float(scores[best]),
        question=entry["question"],
        source=entry["source"],
    )


def load_sheet(path: str) -> int:
    """Load a question,answer CSV of approved common queries."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        add(row["question"], row["answer"], source="sheet")
    return len(rows)


def clear(source: str | None = None) -> int:
    """Clear the cache, or just one source — the nightly flush drops 'harvested'."""
    r = _get_redis()
    keys = list(r.scan_iter(match=KEY_PREFIX + "*"))
    if source is not None:
        keys = [
            k for k, raw in zip(keys, r.mget(keys))
            if raw and json.loads(raw)["source"] == source
        ]
    return r.delete(*keys) if keys else 0


if __name__ == "__main__":
    print(f"loaded {load_sheet(sys.argv[1])} entries")
