from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ScheduledRequest:
    request_id: str
    model: str
    input_ids: list[int]
    sampling_params: dict[str, Any]
