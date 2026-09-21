"""Label -> model routing.

The mapping is the client's config, not code: which tier each label needs is a
claim the eval harness has to check, so it has to be editable without a deploy.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import anthropic

CONFIG_PATH = Path(__file__).parent.parent / "data" / "routing.json"

_config: dict | None = None
_client: anthropic.Anthropic | None = None


def load_config(path: Path = CONFIG_PATH) -> dict:
    global _config
    if _config is None:
        _config = json.loads(path.read_text())
    return _config


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


@dataclass
class Route:
    model: str | None  # None means no model call — escalate to a human
    reason: str


def route(label: str, confidence: float, config: dict | None = None) -> Route:
    config = config or load_config()

    if label in config["escalate"]:
        return Route(model=None, reason="escalated")

    # An unsure classifier is a reason to spend more, not less.
    if confidence < config["confidence_threshold"]:
        return Route(model=config["fallback_model"], reason="low_confidence")

    model = config["routes"].get(label)
    if model is None:
        return Route(model=config["fallback_model"], reason="unknown_label")

    return Route(model=model, reason="label")


def call_model(query: str, model: str) -> str:
    response = _get_client().messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": query}],
    )
    return "".join(b.text for b in response.content if b.type == "text")
