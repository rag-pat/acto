import time

from fastapi import FastAPI
from pydantic import BaseModel

from app.cache import lookup

app = FastAPI(title="acto")


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    label: str
    model_used: str
    cache_hit: bool
    latency_ms: float


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    start = time.perf_counter()

    hit = lookup(req.query)
    if hit:
        answer = hit.answer
    else:
        answer = f"hardcoded answer for: {req.query}"

    return QueryResponse(
        answer=answer,
        label="unclassified",
        model_used="none",
        cache_hit=hit is not None,
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
    )
