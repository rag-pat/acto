# acto

An LLM routing service. A query comes in; the answer comes back with the
metadata that explains how it got there — label, model used, cache hit, latency.

Most support traffic is the same handful of questions asked a thousand ways.
Sending all of it to a frontier model is the expensive way to answer it. This
service puts a semantic cache and a classifier in front of the models so repeats
never reach one, and everything else goes to the cheapest model that can handle
it.

## Flow

```
query -> embed once
      -> cache lookup (cosine >= threshold?)  -> hit: return, no model call
      -> classify (reuses the same vector)
      -> route (label -> model, from config)  -> clinical: escalate to a human
      -> call model, log the query
```

A nightly job clusters the day's log, promotes the most common questions into
the cache, and clears the previous night's promotions.

## Setup

```bash
brew services start redis
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m app.cache data/common_queries.csv   # load the seed sheet
```

Model calls need `ANTHROPIC_API_KEY`. Without it the endpoint still runs — the
cache, classifier and router all work; the model call returns a stub.

## Run

```bash
.venv/bin/python -m uvicorn app.main:app --reload --port 8000
```

```bash
curl -X POST localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"I need to see a specialist, how do I set that up"}'
```

Interactive docs at http://localhost:8000/docs

## Other entry points

```bash
.venv/bin/python -m app.eval          # A/B two classifier variants offline
.venv/bin/python -m app.harvest       # the nightly promotion job
```

## Layout

| file | what it does |
| --- | --- |
| `app/main.py` | the endpoint, and the order the pieces run in |
| `app/cache.py` | embedding, semantic lookup, Redis storage |
| `app/classifier.py` | nearest labeled example, margin as confidence |
| `app/router.py` | label to model, fallback, escalation |
| `app/harvest.py` | query log and the nightly promotion job |
| `app/eval.py` | offline scoring, clinical-weighted |
| `data/` | seed sheet, labeled examples, routing config, eval set |

## Tuning

Thresholds are model-specific and live next to the code that uses them:
`THRESHOLD` in `cache.py`, `CLUSTER_THRESHOLD` in `harvest.py`,
`confidence_threshold` in `data/routing.json`. Changing the embedding model
means recalibrating all three. `app/eval.py` is how you check that a change
was an improvement.
