from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

BatchPhase = Literal["prefill", "decode", "colocate"]


@dataclass(slots=True)
class Request:
    request_id: str
    model: str
    input_ids: list[int]
    uid: int
    sampling_params: Any


@dataclass(slots=True)
class Batch:
    requests: list[Request]
    phase: BatchPhase

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    @property
    def is_colocate(self) -> bool:
        return self.phase == "colocate"


@dataclass(slots=True)
class BatchResult:
    outputs: dict[str, Any]
