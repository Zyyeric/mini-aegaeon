from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass


@dataclass(slots=True)
class AcceleratorSlot:
    slot_id: str
    kind: str  # mig | gpu | cpu
    cuda_visible_devices: str | None


def detect_accelerator_slots() -> list[AcceleratorSlot]:
    """Detect local accelerators from `nvidia-smi -L`.

    If MIG partitions are present, return MIG slots.
    Otherwise return whole-GPU slots.
    If NVIDIA tooling is unavailable, return one CPU slot.
    """

    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return [AcceleratorSlot(slot_id="cpu-0", kind="cpu", cuda_visible_devices=None)]

    text = (out.stdout or "").strip()
    if out.returncode != 0 or not text:
        return [AcceleratorSlot(slot_id="cpu-0", kind="cpu", cuda_visible_devices=None)]

    gpus: list[AcceleratorSlot] = []
    migs: list[AcceleratorSlot] = []

    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("GPU "):
            m = re.match(r"GPU\s+(\d+):", line)
            if m:
                gpu_idx = m.group(1)
                gpus.append(
                    AcceleratorSlot(
                        slot_id=f"gpu-{gpu_idx}",
                        kind="gpu",
                        cuda_visible_devices=gpu_idx,
                    )
                )
            continue

        if line.startswith("MIG "):
            uuid_match = re.search(r"UUID:\s*([^\)]+)\)", line)
            mig_uuid = uuid_match.group(1).strip() if uuid_match else None
            mig_idx = len(migs)
            migs.append(
                AcceleratorSlot(
                    slot_id=f"mig-{mig_idx}",
                    kind="mig",
                    cuda_visible_devices=mig_uuid,
                )
            )

    if migs:
        return migs
    if gpus:
        return gpus
    return [AcceleratorSlot(slot_id="cpu-0", kind="cpu", cuda_visible_devices=None)]
