from __future__ import annotations

import asyncio
import random
import statistics
import time
from dataclasses import dataclass
from typing import Any, List, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from openai import AsyncOpenAI as OpenAI
else:
    OpenAI = Any


@dataclass(frozen=True)
class BenchmarkTrace:
    """One request in a benchmark trace.

    timestamp is in seconds relative to the start of replay.
    """

    timestamp: float
    message: str
    output_length: int
    input_length: int | None = None

    def __post_init__(self) -> None:
        if self.timestamp < 0:
            raise ValueError("timestamp must be >= 0")
        if not self.message:
            raise ValueError("message must be non-empty")
        if self.output_length <= 0:
            raise ValueError("output_length must be > 0")
        if self.input_length is not None and self.input_length <= 0:
            raise ValueError("input_length must be > 0 when provided")


@dataclass(frozen=True)
class BenchOneResult:
    """Result of one request.

    tics stores token/event times in seconds relative to request dispatch time.
    """

    tics: List[float]
    input_len: int
    output_len: int

    def __post_init__(self) -> None:
        if self.input_len < 0:
            raise ValueError("input_len must be >= 0")
        if self.output_len < 0:
            raise ValueError("output_len must be >= 0")
        if any(t < 0 for t in self.tics):
            raise ValueError("tics must be non-negative")

    def as_json(self) -> List[float]:
        return [float(self.input_len), float(self.output_len), *self.tics]

    @staticmethod
    def from_json(raw: List[float]) -> "BenchOneResult":
        if len(raw) < 2:
            raise ValueError("raw must contain at least [input_len, output_len]")

        in_len = float(raw[0])
        out_len = float(raw[1])
        if not in_len.is_integer() or not out_len.is_integer():
            raise ValueError("raw[0] and raw[1] must be integers")

        tics = [float(x) for x in raw[2:]]
        return BenchOneResult(tics=tics, input_len=int(in_len), output_len=int(out_len))

    @property
    def ttft_s(self) -> float | None:
        return self.tics[0] if self.tics else None

    @property
    def tbt_mean_s(self) -> float | None:
        if len(self.tics) < 2:
            return None
        gaps = [self.tics[i + 1] - self.tics[i] for i in range(len(self.tics) - 1)]
        return float(statistics.fmean(gaps)) if gaps else None


@dataclass(frozen=True)
class BenchmarkResult:
    raw_data: List[BenchOneResult]

    def as_json(self) -> List[List[float]]:
        return [r.as_json() for r in self.raw_data]

    @staticmethod
    def from_json(raw: List[List[float]]) -> "BenchmarkResult":
        return BenchmarkResult(raw_data=[BenchOneResult.from_json(r) for r in raw])

    def summary(self) -> dict[str, float | int | None]:
        ttfts = [x.ttft_s for x in self.raw_data if x.ttft_s is not None]
        tbts = [x.tbt_mean_s for x in self.raw_data if x.tbt_mean_s is not None]
        return {
            "requests": len(self.raw_data),
            "ttft_avg_s": float(statistics.fmean(ttfts)) if ttfts else None,
            "tbt_avg_s": float(statistics.fmean(tbts)) if tbts else None,
            "total_output_tokens": sum(x.output_len for x in self.raw_data),
        }


@dataclass(frozen=True)
class TraceGenerator:
    """Deterministic trace generator for OpenAI-style text benchmarks."""

    output_length: int = 64
    start_timestamp: float = 0.0
    interval_s: float = 0.5
    jitter_s: float = 0.0
    seed: int = 0
    prompt_template: str = "Request {i}: summarize serving latency in 3 concise bullet points."
    input_length: int | None = None

    def generate(self, num_requests: int) -> list[BenchmarkTrace]:
        if num_requests <= 0:
            raise ValueError("num_requests must be > 0")
        if self.interval_s <= 0:
            raise ValueError("interval_s must be > 0")
        if self.jitter_s < 0:
            raise ValueError("jitter_s must be >= 0")

        rng = random.Random(self.seed)
        traces: list[BenchmarkTrace] = []
        for i in range(num_requests):
            base_ts = self.start_timestamp + i * self.interval_s
            ts = max(0.0, base_ts + rng.uniform(-self.jitter_s, self.jitter_s))
            message = self.prompt_template.format(i=i)
            traces.append(
                BenchmarkTrace(
                    timestamp=ts,
                    message=message,
                    output_length=self.output_length,
                    input_length=self.input_length,
                )
            )
        return sorted(traces, key=lambda x: x.timestamp)


async def benchmark_trace(
    client: OpenAI,
    model: str,
    trace: Sequence[BenchmarkTrace],
    *,
    temperature: float = 0.0,
    max_concurrency: int = 16,
) -> BenchmarkResult:
    """Replay a trace against an AsyncOpenAI-like client.

    Expected client API:
      await client.chat.completions.create(..., stream=True)
    """

    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be > 0")

    sem = asyncio.Semaphore(max_concurrency)
    replay_start = time.perf_counter()
    ordered = sorted(enumerate(trace), key=lambda x: x[1].timestamp)
    raw: list[BenchOneResult | None] = [None] * len(trace)

    async def run_one(idx: int, item: BenchmarkTrace) -> None:
        target = replay_start + item.timestamp
        wait_s = target - time.perf_counter()
        if wait_s > 0:
            await asyncio.sleep(wait_s)

        async with sem:
            req_start = time.perf_counter()
            tics: list[float] = []
            input_len = item.input_length if item.input_length is not None else 0
            output_len = item.output_length

            stream = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": item.message}],
                max_tokens=item.output_length,
                temperature=temperature,
                stream=True,
                stream_options={"include_usage": True},
            )

            async for chunk in stream:
                now = time.perf_counter()

                choices = getattr(chunk, "choices", None) or []
                if choices:
                    delta = getattr(choices[0], "delta", None)
                    text = getattr(delta, "content", None) if delta is not None else None
                    if text:
                        tics.append(now - req_start)

                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    in_tokens = getattr(usage, "prompt_tokens", None)
                    out_tokens = getattr(usage, "completion_tokens", None)
                    if isinstance(in_tokens, int) and in_tokens >= 0:
                        input_len = in_tokens
                    if isinstance(out_tokens, int) and out_tokens >= 0:
                        output_len = out_tokens

            if output_len == item.output_length and tics:
                # Fallback when usage isn't emitted: treat streamed events as approximate token count.
                output_len = len(tics)

            raw[idx] = BenchOneResult(tics=tics, input_len=input_len, output_len=output_len)

    await asyncio.gather(*(run_one(i, item) for i, item in ordered))
    return BenchmarkResult(raw_data=[x for x in raw if x is not None])
