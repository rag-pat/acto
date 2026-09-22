# acto

**A personal rebuild, not a production system.** I worked on an LLM routing
service and rebuilt a scaled-down version of it here to understand it end to end
and to have something concrete to point at. The architecture is the same; the
scale is not. Local embedding model instead of a self-hosted one, a handful of
seeded entries instead of real traffic, invented data throughout. None of the
client's code or data is here.

## The problem

Most support traffic is the same handful of questions asked a thousand different
ways. Sending all of it to a frontier model is the expensive way to answer it.
This service puts a semantic cache and a classifier in front of the models, so
repeats never reach one and everything else goes to the cheapest model that can
handle it.

## Flow

```
query
  -> embed once                      one vector, reused by the next two stages
  -> cache lookup                    cosine >= threshold?  hit: return, no model
  -> classify                        nearest labeled example; margin = confidence
  -> route                           label -> model, from config
                                     low margin -> fallback model
                                     clinical   -> escalate to a human
  -> call model, log the query
```

**Embed once.** The query is normalized, embedded, and L2-normalized, which
makes every comparison downstream a plain dot product. Both the cache and the
classifier use that one vector, so classification costs nothing extra.

**Cache.** Semantic rather than exact, so one intent phrased many ways lands on
one entry. The set is small and curated — a client-approved sheet plus last
night's promotions — so a hit returns a vetted answer, not an arbitrary old
generation.

**Classify.** Nearest labeled example wins. Confidence is the margin between the
top two labels, not the raw similarity: what matters isn't how close the query
sits to anything, it's whether one label clearly beats the rest.

**Route.** A config-driven `label -> model` table. Clinical escalates to a human
and is checked before confidence, so a low-confidence clinical query can't fall
through to a model. A narrow margin goes to the fallback model — unsure means
spend more, not less.

**Write-back.** Nothing on the request path writes to the cache; it appends to a
log. A nightly job clusters that log semantically, promotes the most common
questions, and clears the previous night's promotions.

## Pieces

| file | responsibility |
| --- | --- |
| `app/main.py` | the endpoint, and the order the stages run in |
| `app/cache.py` | embedding, semantic lookup, Redis storage |
| `app/classifier.py` | nearest labeled example, margin as confidence |
| `app/router.py` | label to model, fallback, escalation |
| `app/harvest.py` | query log and the nightly promotion job |
| `app/eval.py` | offline scoring, weighted toward clinical recall |
| `data/` | seed sheet, labeled examples, routing config, eval set |

## Why it's tuned the way it is

Every threshold in the system trades the same two costs against each other: a
miss costs one model call, while a false hit serves someone a confidently wrong
answer. That asymmetry is why the thresholds sit high and the eval is weighted
toward precision rather than accuracy.

Thresholds are specific to the embedding model, so they live next to the code
that uses them — `THRESHOLD` in `cache.py`, `CLUSTER_THRESHOLD` in `harvest.py`,
`confidence_threshold` in `data/routing.json`. Swapping the embedding model
means recalibrating all three, and `app/eval.py` is how you check the change was
actually an improvement.
