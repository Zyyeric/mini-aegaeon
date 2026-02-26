from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass
from typing import Dict

import torch

from .model_cache import ModelCache, ModelCacheEntry

LOGGER = logging.getLogger(__name__)


def _merge_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    filtered_state_dict: Dict[str, torch.Tensor] = {}
    for key in list(state_dict.keys()):
        if key.count(".q_proj"):
            q_proj = state_dict[key]
            k_proj = state_dict[key.replace(".q_proj", ".k_proj")]
            v_proj = state_dict[key.replace(".q_proj", ".v_proj")]
            new_key = key.replace(".q_proj", ".qkv_proj")
            filtered_state_dict[new_key] = torch.cat([q_proj, k_proj, v_proj], dim=0)
            del state_dict[key]
            del state_dict[key.replace(".q_proj", ".k_proj")]
            del state_dict[key.replace(".q_proj", ".v_proj")]
        elif key.count(".gate_proj"):
            gate_proj = state_dict[key]
            up_proj = state_dict[key.replace(".gate_proj", ".up_proj")]
            new_key = key.replace(".gate_proj", ".gate_up_proj")
            filtered_state_dict[new_key] = torch.cat([gate_proj, up_proj], dim=0)
            del state_dict[key]
            del state_dict[key.replace(".gate_proj", ".up_proj")]
        elif key.count(".k_proj") or key.count(".v_proj") or key.count("up_proj"):
            continue
        else:
            filtered_state_dict[key] = state_dict[key]
    return filtered_state_dict


def load_hf_weight(
    model_path: str,
    device: torch.device,
    *,
    dtype: torch.dtype | None = None,
    pin_cpu_weight: bool = True
) -> Dict[str, torch.Tensor]:

    if os.path.isdir(model_path):
        hf_folder = model_path
    else:
        try:
            from huggingface_hub import snapshot_download

            hf_folder = snapshot_download(
                model_path,
                allow_patterns=["*.safetensors"],
            )
        except Exception as exc:
            raise ValueError(
                f"Model path '{model_path}' is neither a local directory nor a valid HuggingFace repository ID"
            ) from exc

    files = glob.glob(f"{hf_folder}/*.safetensors")
    state_dict: Dict[str, torch.Tensor] = {}
    try:
        import safetensors
    except Exception as exc:
        raise RuntimeError("safetensors is required to load hf weights") from exc

    for file in sorted(files):
        with safetensors.safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                if dtype is not None:
                    tensor = tensor.to(dtype)
                state_dict[name] = tensor

    state_dict = _merge_state_dict(state_dict)

    if pin_cpu_weight:
        pinned: Dict[str, torch.Tensor] = {}
        for name, tensor in state_dict.items():
            t = tensor.to("cpu")
            if not t.is_pinned():
                try:
                    t = t.pin_memory()
                except Exception:
                    LOGGER.warning(
                        "pin_memory() failed for '%s'; continuing with non-pinned CPU tensor",
                        name,
                    )
            pinned[name] = t
        state_dict = pinned
    else:        
        state_dict = {k: v.to(device) for k, v in state_dict.items()}

    return state_dict


@dataclass(slots=True)
class HFWeightLoader:
    dtype: torch.dtype | None = None

    def load_state_dict(self, model: str) -> Dict[str, torch.Tensor]:
        return load_hf_weight(
            model,
            device=torch.device("cpu"),
            dtype=self.dtype,
            pin_cpu_weight=True,
        )

    def fetch_to_model_cache(self, model: str, cache: ModelCache) -> ModelCacheEntry:
        state_dict = self.load_state_dict(model)
        chunk_sizes = [int(t.numel()) * int(t.element_size()) for t in state_dict.values()]
        return cache.put_state_dict(
            model=model,
            state_dict=state_dict,
            raw_chunk_sizes=chunk_sizes,
        )
