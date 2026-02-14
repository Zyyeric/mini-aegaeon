from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .metadata_store import RequestPhase


@dataclass(slots=True)
class RequestEnvelope:
    request_id: str
    model: str
    payload: dict[str, Any]
    phase: RequestPhase


@dataclass(slots=True)
class RouteDecision:
    request_id: str
    instance_id: str
    endpoint: str
    phase: RequestPhase
    queued: bool = False
