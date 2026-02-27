from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, List, TYPE_CHECKING, overload
from tqdm import tqdm

if TYPE_CHECKING:
    from openai import AsyncOpenAI as OpenAI
else:
    OpenAI = Any

logger = logging.getLogger(__name__)


class Unset:
    pass


UNSET = Unset()


@dataclass
class Counter:
    value: int = 0
    history_max: int = 0

    def inc(self, n: int = 1) -> None:
        self.value += n
        if self.value > self.history_max:
            self.history_max = self.value

    def dec(self, n: int = 1) -> None:
        self.value = max(0, self.value - n)


@dataclass
class Console:
    input_pbar: Any
    output_pbar: Any
    prefill_pbar: Any
    decode_pbar: Any
    disabled: bool = False
    inflight_counter: Counter = field(default_factory=Counter)
    queue_counter: Counter = field(default_factory=Counter)

    def update_input(self, n: int = 1) -> None:
        self.input_pbar.update(n)
        self.input_pbar.refresh()
        self.inflight_counter.inc(n)
        self.queue_counter.inc(n)

    def update_output(self, n: int = 1) -> None:
        self.output_pbar.update(n)
        self.output_pbar.refresh()
        self.inflight_counter.dec(n)

    def update_prefill(self, n: int = 1) -> None:
        self.prefill_pbar.update(n)
        self.prefill_pbar.refresh()
        self.queue_counter.dec(n)

    def update_decode(self, n: int = 1) -> None:
        self.decode_pbar.update(n)

    def close(self) -> None:
        self.input_pbar.close()
        self.output_pbar.close()
        self.prefill_pbar.close()
        self.decode_pbar.close()

    @contextlib.contextmanager
    def inflight(self, n: int = 1):
        self.update_input(n)
        try:
            yield
        finally:
            self.update_output(n)

    @contextlib.contextmanager
    def log_stats(self):
        try:
            yield
        finally:
            self.close()
            if not self.disabled:
                max_inflight = self.inflight_counter.history_max
                max_queue = self.queue_counter.history_max
                logger.info("Max inflight requests: %d, Max queued requests: %d", max_inflight, max_queue)


def make_console(num_requests: int, sum_output_length: int, use_pbar: bool = True) -> Console:
    """Create progress bars for benchmark replay and token progress."""
    if num_requests <= 0:
        raise ValueError("num_requests must be > 0")
    if sum_output_length < 0:
        raise ValueError("sum_output_length must be >= 0")

    prefill_tokens = num_requests
    decode_tokens = max(0, sum_output_length - prefill_tokens)
    disabled = not use_pbar

    bar_format = (
        "{desc:<10} {percentage:3.0f}%|{bar}|"
        " {n_fmt:>5}/{total_fmt} "
        "[{rate_fmt:>12} {elapsed:>8}/{remaining:<8}]"
    )
    align = max(5, len(str(decode_tokens)))
    if align != 5:
        bar_format = bar_format.replace("{n_fmt:>5}", "{n_fmt:>" + str(align) + "}")

    input_pbar = tqdm(total=num_requests, desc="Requests sent", position=0, bar_format=bar_format, disable=disabled)
    output_pbar = tqdm(total=num_requests, desc="Requests done", position=1, bar_format=bar_format, disable=disabled)
    prefill_pbar = tqdm(total=prefill_tokens, desc="Prefill token", position=2, bar_format=bar_format, disable=disabled)
    decode_pbar = tqdm(total=decode_tokens, desc="Decode token ", position=3, bar_format=bar_format, disable=disabled)
    return Console(
        input_pbar=input_pbar,
        output_pbar=output_pbar,
        prefill_pbar=prefill_pbar,
        decode_pbar=decode_pbar,
        disabled=disabled,
    )


def generate_prompt(tokenizer: Any, n: int, *, rng: random.Random | None = None) -> str:
    """Generate a prompt of approximately `n` tokens using the provided tokenizer."""
    if n <= 0:
        raise ValueError("n must be > 0")
    if tokenizer is None:
        raise ValueError("tokenizer is required")

    rnd = rng or random.Random()
    vocab_size_raw = int(getattr(tokenizer, "vocab_size", 0))
    if vocab_size_raw <= 0:
        raise ValueError("tokenizer.vocab_size must be > 0")

    vocab_size = max(2, vocab_size_raw // 2)
    token_ids = [rnd.randint(0, vocab_size - 1) for _ in range(n)]

    for _ in range(64):
        prompt = tokenizer.decode(token_ids)
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(token_ids) == n:
            return prompt
        if len(token_ids) < n:
            need = n - len(token_ids)
            token_ids.extend([rnd.randint(0, vocab_size - 1) for _ in range(need)])
        else:
            token_ids = token_ids[:n]

    raise ValueError("Failed to generate a message of the desired length.")


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
class RawResult:
    tics: List[float]
    output_len: int
    message: str = ""
    input_len: int | None = None


@overload
def process_benchmark_results(raw_data: List[RawResult], tokenizer: Any) -> BenchmarkResult: ...


@overload
def process_benchmark_results(raw_data: List[RawResult]) -> None: ...


def process_benchmark_results(
    raw_data: List[RawResult],
    tokenizer: Any = UNSET,
) -> BenchmarkResult | None:
    results = [r.tics for r in raw_data if len(r.tics) >= 2]
    if not results:
        raise ValueError("Need at least one result with >=2 tics.")

    first_times: List[float] = []
    accum_times: List[float] = []
    for tics in results:
        deltas = [tics[i + 1] - tics[i] for i in range(len(tics) - 1)]
        if not deltas:
            continue
        first_times.append(deltas[0])
        accum_times.extend(deltas[1:])

    e2e_times = [tics[-1] - tics[0] for tics in results]
    first_times.sort()
    accum_times.sort()
    e2e_times.sort()

    def _print_stats(times: List[float], scale: float = 1.0) -> tuple[float, float, float, float, float]:
        if not times:
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        n = len(times)
        return (
            scale * sum(times) / n,
            scale * times[min(int(n * 0.5), n - 1)],
            scale * times[min(int(n * 0.9), n - 1)],
            scale * times[min(int(n * 0.99), n - 1)],
            scale * max(times),
        )

    def _fmt(x: float) -> str:
        if x >= 1000:
            return f"{int(x):>6}"
        if x >= 10:
            return f"{x:>6.2f}"
        return f"{x:>6.4f}"

    avg_ttft, p50_ttft, p90_ttft, p99_ttft, max_ttft = _print_stats(first_times, 1000.0)
    avg_tpot, p50_tpot, p90_tpot, p99_tpot, max_tpot = _print_stats(accum_times, 1000.0)
    avg_e2e, p50_e2e, p90_e2e, p99_e2e, max_e2e = _print_stats(e2e_times, 1.0)

    min_time = min(min(t) for t in results)
    max_time = max(max(t) for t in results)
    dur = max_time - min_time
    if dur <= 0:
        raise ValueError("Duration must be positive")

    num_tokens = sum(max(len(tics) - 1, 0) for tics in results)
    num_requests = len(results)

    logger.info("Num requests: #%d, Num tokens: #%d", num_requests, num_tokens)
    logger.info(
        "TTFT: %s ms (p50: %s ms, p90: %s ms, p99: %s ms, max: %s ms)",
        _fmt(avg_ttft),
        _fmt(p50_ttft),
        _fmt(p90_ttft),
        _fmt(p99_ttft),
        _fmt(max_ttft),
    )
    logger.info(
        "TPOT: %s ms (p50: %s ms, p90: %s ms, p99: %s ms, max: %s ms)",
        _fmt(avg_tpot),
        _fmt(p50_tpot),
        _fmt(p90_tpot),
        _fmt(p99_tpot),
        _fmt(max_tpot),
    )
    logger.info(
        "E2E:  %s  s (p50: %s  s, p90: %s  s, p99: %s  s, max: %s  s)",
        _fmt(avg_e2e),
        _fmt(p50_e2e),
        _fmt(p90_e2e),
        _fmt(p99_e2e),
        _fmt(max_e2e),
    )
    logger.info("Duration: %s s", _fmt(dur))
    logger.info("Throughput: %s token/s, %s req/s", _fmt(num_tokens / dur), _fmt(num_requests / dur))

    if isinstance(tokenizer, Unset):
        return None

    return BenchmarkResult(
        raw_data=[
            BenchOneResult(
                tics=r.tics,
                input_len=(
                    r.input_len
                    if r.input_len is not None
                    else len(tokenizer.encode(r.message, add_special_tokens=False))
                ),
                output_len=r.output_len,
            )
            for r in raw_data
            if len(r.tics) >= 2
        ]
    )


async def benchmark_trace(
    client: OpenAI,
    msgs: List[BenchmarkTrace],
    model: str,
    *,
    pbar: Console | bool = True,
    temperature: float = 0.0,
) -> List[RawResult]:
    if not msgs:
        return []
    if isinstance(pbar, bool):
        sum_output_len = sum(msg.output_length for msg in msgs)
        pbar = make_console(len(msgs), sum_output_len, use_pbar=pbar)

    start = time.perf_counter()
    offset = min(msg.timestamp for msg in msgs) - 1.0

    async def benchmark_timed(msg: BenchmarkTrace) -> RawResult:
        target = start + msg.timestamp - offset
        await asyncio.sleep(max(0.0, target - time.perf_counter()))
        return await benchmark_one(
            client=client,
            message=msg.message,
            output_length=msg.output_length,
            model=model,
            pbar=pbar,
            input_length=msg.input_length,
            temperature=temperature,
        )

    tasks = [benchmark_timed(msg) for msg in msgs]
    with pbar.log_stats():
        return await asyncio.gather(*tasks)


async def benchmark_one(
    client: OpenAI,
    message: str,
    output_length: int,
    model: str,
    *,
    pbar: Console,
    input_length: int | None = None,
    temperature: float = 0.0,
) -> RawResult:
    if output_length <= 0:
        raise ValueError("output_length must be > 0")

    tics: List[float] = []
    prompt_tokens = input_length
    completion_tokens: int | None = None
    saw_first_token = False
    streamed_events = 0

    with pbar.inflight(1):
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": message}],
            max_tokens=output_length,
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
                    streamed_events += 1
                    tics.append(now)
                    if not saw_first_token:
                        pbar.update_prefill(1)
                        saw_first_token = True
                    else:
                        pbar.update_decode(1)

            usage = getattr(chunk, "usage", None)
            if usage is not None:
                in_tokens = getattr(usage, "prompt_tokens", None)
                out_tokens = getattr(usage, "completion_tokens", None)
                if isinstance(in_tokens, int) and in_tokens >= 0:
                    prompt_tokens = in_tokens
                if isinstance(out_tokens, int) and out_tokens >= 0:
                    completion_tokens = out_tokens

    if completion_tokens is None:
        completion_tokens = max(streamed_events, 0)
    if prompt_tokens is None:
        prompt_tokens = 0

    return RawResult(
        tics=tics,
        output_len=completion_tokens,
        message=message,
        input_len=prompt_tokens,
    )
