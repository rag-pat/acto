import time

from fastapi import FastAPI
from pydantic import BaseModel

from app.cache import embed, lookup
from app.classifier import classify
from app.harvest import log_query
from app.router import call_model, route

app = FastAPI(title="acto")

ESCALATION_ANSWER = (
    "This one needs a clinician. I'm passing you to a member of staff."
)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    label: str
    confidence: float
    model_used: str
    cache_hit: bool
    latency_ms: float


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    start = time.perf_counter()

    # Embedded once, used by both the cache and the classifier.
    vector = embed(req.query)

    hit = lookup(req.query, vector=vector)
    if hit:
        return QueryResponse(
            answer=hit.answer,
            label="cached",
            confidence=round(hit.score, 4),
            model_used="none",
            cache_hit=True,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
        )

    classification = classify(req.query, vector=vector)
    decision = route(classification.label, classification.confidence)

    if decision.model is None:
        answer = ESCALATION_ANSWER
    else:
        try:
            answer = call_model(req.query, decision.model)
        except Exception as exc:  # no key configured, provider down, etc.
            answer = f"[no model call: {type(exc).__name__}]"
        log_query(req.query, answer, decision.model, vector=vector)

    return QueryResponse(
        answer=answer,
        label=classification.label,
        confidence=classification.confidence,
        model_used=decision.model or "escalated",
        cache_hit=False,
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
    )
