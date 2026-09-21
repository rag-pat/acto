"""Query log and the nightly harvest.

The request path never writes to the cache — it appends here. Once a day this
job groups the day's queries semantically (counting exact strings would
undercount the paraphrases that are the whole point), promotes the most common
into the cache, and drops yesterday's harvested entries.
"""

import json
import sys
from datetime import date

import numpy as np

from app import cache

LOG_PREFIX = "acto:log:"
CLUSTER_THRESHOLD = 0.40  # measured: paraphrases sit >0.49, unrelated <0.29 on MiniLM
TOP_N = 20
HARVEST_TTL = 60 * 60 * 26  # expires on its own if the job doesn't run


def _log_key(day: date | None = None) -> str:
    return LOG_PREFIX + (day or date.today()).isoformat()


def log_query(query: str, answer: str, model: str, vector=None) -> None:
    """Append one answered query. The vector is kept so the nightly job
    doesn't have to re-embed the whole day."""
    vector = cache.embed(query) if vector is None else vector
    entry = {
        "query": query,
        "answer": answer,
        "model": model,
        "embedding": np.asarray(vector).tolist(),
    }
    cache._get_redis().rpush(_log_key(), json.dumps(entry))


def read_log(day: date | None = None) -> list[dict]:
    raw = cache._get_redis().lrange(_log_key(day), 0, -1)
    return [json.loads(r) for r in raw]


def cluster(entries: list[dict], threshold: float = CLUSTER_THRESHOLD) -> list[list[dict]]:
    """Greedy grouping: an entry joins the first cluster it's close enough to."""
    clusters: list[list[dict]] = []
    centroids: list[np.ndarray] = []

    for entry in entries:
        vector = np.array(entry["embedding"])
        for i, centroid in enumerate(centroids):
            if float(centroid @ vector) >= threshold:
                clusters[i].append(entry)
                members = np.array([e["embedding"] for e in clusters[i]])
                mean = members.mean(axis=0)
                centroids[i] = mean / np.linalg.norm(mean)
                break
        else:
            clusters.append([entry])
            centroids.append(vector)

    return clusters


def harvest(day: date | None = None, top_n: int = TOP_N) -> list[tuple[str, int]]:
    """Promote the day's most common questions into the cache."""
    entries = read_log(day)
    if not entries:
        return []

    groups = sorted(cluster(entries), key=len, reverse=True)[:top_n]

    cache.clear(source="harvested")  # yesterday's promotions expire tonight

    promoted = []
    for group in groups:
        # Representative = the phrasing closest to the group's centre.
        vectors = np.array([e["embedding"] for e in group])
        mean = vectors.mean(axis=0)
        mean /= np.linalg.norm(mean)
        best = group[int(np.argmax(vectors @ mean))]

        cache.add(best["query"], best["answer"], source="harvested", ttl=HARVEST_TTL)
        promoted.append((best["query"], len(group)))

    return promoted


if __name__ == "__main__":
    day = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else None
    for query, count in harvest(day):
        print(f"{count:>4}x  {query}")
