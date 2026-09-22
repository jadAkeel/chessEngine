from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch

from app.infra.config import AppConfig, SystemConfig, get_current_config

logger = logging.getLogger(__name__)

_RUNTIME_STATE: dict[str, int | str | None] = {
    "intra_threads": None,
    "interop_threads": None,
    "device": None,
    "role": None,
    "workers": None,
}


@dataclass(frozen=True)
class RuntimeThreadPlan:
    intra_threads: int
    interop_threads: int
    worker_count: int
    device: str
    role: str


def _cpu_count() -> int:
    return max(1, int(os.cpu_count() or 1))


def _cap_for_role(role: str, device: str) -> int:
    if device.startswith("cuda") or device == "mps":
        return 4
    if role in {"training", "trainer"}:
        return 16
    return 8


def resolve_thread_plan(
    cfg: AppConfig | SystemConfig | None = None,
    *,
    device: str,
    role: str = "general",
    worker_count: int = 1,
) -> RuntimeThreadPlan:
    root_cfg = cfg or get_current_config()
    system_cfg = root_cfg.system if hasattr(root_cfg, "system") else root_cfg

    total_cpus = _cpu_count()
    worker_count = max(1, int(worker_count))
    requested_threads = int(getattr(system_cfg, "cpu_threads", 0) or 0)
    requested_interop = int(getattr(system_cfg, "interop_threads", 0) or 0)
    policy = str(getattr(system_cfg, "worker_thread_policy", "auto") or "auto").lower()

    if requested_threads > 0:
        intra_threads = requested_threads
    else:
        share = max(1, total_cpus // worker_count)
        if policy == "per_worker":
            intra_threads = share
        elif policy == "fixed":
            intra_threads = total_cpus
        else:
            intra_threads = share if worker_count > 1 else total_cpus
        intra_threads = min(intra_threads, _cap_for_role(role, device))

    intra_threads = max(1, min(int(intra_threads), total_cpus))

    if requested_interop > 0:
        interop_threads = requested_interop
    else:
        if device.startswith("cuda") or device == "mps":
            interop_threads = 1
        elif worker_count > 1:
            interop_threads = 1
        else:
            interop_threads = min(2, intra_threads)

    interop_threads = max(1, min(int(interop_threads), intra_threads))

    return RuntimeThreadPlan(
        intra_threads=intra_threads,
        interop_threads=interop_threads,
        worker_count=worker_count,
        device=str(device),
        role=str(role),
    )


def apply_thread_plan(plan: RuntimeThreadPlan) -> RuntimeThreadPlan:
    thread_value = str(plan.intra_threads)
    os.environ["OMP_NUM_THREADS"] = thread_value
    os.environ["MKL_NUM_THREADS"] = thread_value
    os.environ["OPENBLAS_NUM_THREADS"] = thread_value
    os.environ["NUMEXPR_NUM_THREADS"] = thread_value

    try:
        torch.set_num_threads(int(plan.intra_threads))
    except Exception as exc:
        logger.debug("Failed to set torch intra-op threads: %s", exc)

    try:
        torch.set_num_interop_threads(int(plan.interop_threads))
    except RuntimeError:
        previous = _RUNTIME_STATE.get("interop_threads")
        if previous != int(plan.interop_threads):
            logger.debug(
                "torch inter-op threads already initialized; keeping previous=%s requested=%s",
                previous,
                plan.interop_threads,
            )
    except Exception as exc:
        logger.debug("Failed to set torch inter-op threads: %s", exc)

    _RUNTIME_STATE.update(
        {
            "intra_threads": int(plan.intra_threads),
            "interop_threads": int(plan.interop_threads),
            "device": str(plan.device),
            "role": str(plan.role),
            "workers": int(plan.worker_count),
        }
    )
    return plan


def configure_torch_runtime(
    cfg: AppConfig | SystemConfig | None = None,
    *,
    device: str,
    role: str = "general",
    worker_count: int = 1,
) -> RuntimeThreadPlan:
    plan = resolve_thread_plan(cfg, device=device, role=role, worker_count=worker_count)
    applied = apply_thread_plan(plan)
    logger.debug(
        "Configured runtime threads role=%s device=%s workers=%s intra=%s interop=%s",
        applied.role,
        applied.device,
        applied.worker_count,
        applied.intra_threads,
        applied.interop_threads,
    )
    return applied
