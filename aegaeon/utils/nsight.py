from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def nvtx_range(name: str) -> Iterator[None]:
    """Create an NVTX range if available, otherwise fall back to no-op."""
    try:
        import torch.cuda.nvtx as nvtx
    except Exception:
        yield
        return

    with nvtx.range(name):
        yield


def nvtx_annotate(name: str, layer_id_field: str | None = None):
    """Decorator to wrap a method call in an NVTX range."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            display_name = name
            if layer_id_field and hasattr(self, layer_id_field):
                display_name = name.format(getattr(self, layer_id_field))
            with nvtx_range(display_name):
                return fn(self, *args, **kwargs)

        return wrapper

    return decorator
