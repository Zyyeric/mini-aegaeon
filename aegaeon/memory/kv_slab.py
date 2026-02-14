from __future__ import annotations

from dataclasses import dataclass


class SlabAllocationError(RuntimeError):
    pass


@dataclass(slots=True)
class SlabHandle:
    slab_id: int
    chunks: list[int]


class SlabKVBackend:
    """Fixed-size chunk slab allocator for shared KV memory."""

    def __init__(self, total_bytes: int, chunk_bytes: int) -> None:
        if chunk_bytes <= 0 or total_bytes <= 0:
            raise ValueError("total_bytes and chunk_bytes must be > 0")
        if total_bytes % chunk_bytes != 0:
            raise ValueError("total_bytes must be divisible by chunk_bytes")

        self._chunk_bytes = chunk_bytes
        self._total_chunks = total_bytes // chunk_bytes
        self._free = list(range(self._total_chunks))
        self._allocs: dict[int, SlabHandle] = {}
        self._next_id = 1

    def allocate(self, bytes_needed: int) -> SlabHandle:
        chunks_needed = (bytes_needed + self._chunk_bytes - 1) // self._chunk_bytes
        if chunks_needed > len(self._free):
            raise SlabAllocationError("insufficient KV slab space")

        picked = [self._free.pop() for _ in range(chunks_needed)]
        handle = SlabHandle(slab_id=self._next_id, chunks=picked)
        self._allocs[handle.slab_id] = handle
        self._next_id += 1
        return handle

    def free(self, slab_id: int) -> None:
        handle = self._allocs.pop(slab_id, None)
        if not handle:
            return
        self._free.extend(handle.chunks)

    def stats(self) -> dict[str, int]:
        used_chunks = self._total_chunks - len(self._free)
        return {
            "chunk_bytes": self._chunk_bytes,
            "total_chunks": self._total_chunks,
            "used_chunks": used_chunks,
            "free_chunks": len(self._free),
        }
