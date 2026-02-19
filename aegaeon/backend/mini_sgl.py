from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from aegaeon.memory.weight_manager import TensorMap, WeightManager

from .base import Inference_Backend


def _ensure_minisgl_importable() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    minisgl_python_dir = repo_root / "third-party" / "mini-sglang" / "python"
    path_str = str(minisgl_python_dir)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


_ensure_minisgl_importable()

from minisgl.core import SamplingParams  # type: ignore  # noqa: E402
from minisgl.llm import LLM as MiniSGLLLM  # type: ignore  # noqa: E402


class MiniSGL_Backend(Inference_Backend, MiniSGLLLM):

    def __init__(self, *args: Any, weight_manager: WeightManager, **kwargs: Any) -> None:
        super().__init__(*args, weight_manager=weight_manager, **kwargs)

    def load_weights_for_batch(self, state_dict: TensorMap) -> None:
        self.engine.model.load_state_dict(state_dict)
