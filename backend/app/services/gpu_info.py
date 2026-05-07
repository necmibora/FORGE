"""GPU + system info via pynvml/psutil. Degrades gracefully if NVML missing."""
from __future__ import annotations

import os
import socket
from typing import Optional

import psutil

from app.schemas import GPUInfo, SystemInfo


def _gpus() -> list[GPUInfo]:
    try:
        import pynvml  # type: ignore
    except ImportError:
        return []

    try:
        pynvml.nvmlInit()
    except Exception:
        return []

    out: list[GPUInfo] = []
    try:
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            try:
                temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = None
            out.append(
                GPUInfo(
                    index=i,
                    name=name,
                    total_memory_mb=mem.total // (1024 * 1024),
                    free_memory_mb=mem.free // (1024 * 1024),
                    used_memory_mb=mem.used // (1024 * 1024),
                    utilization_pct=int(util.gpu),
                    temperature_c=temp,
                )
            )
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return out


def gpu_snapshot(index: int = 0) -> Optional[dict]:
    """Lightweight single-GPU sample for benchmark monitoring.

    Returns dict {name, total_mb, used_mb, util_pct, temp_c} or None if NVML
    is unavailable. Caller should swallow exceptions (best-effort sampling).
    """
    try:
        import pynvml  # type: ignore
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
    except Exception:
        return None
    try:
        if pynvml.nvmlDeviceGetCount() <= index:
            return None
        h = pynvml.nvmlDeviceGetHandleByIndex(index)
        name = pynvml.nvmlDeviceGetName(h)
        if isinstance(name, bytes):
            name = name.decode()
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        try:
            temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            temp = None
        return {
            "name": name,
            "total_mb": mem.total // (1024 * 1024),
            "used_mb": mem.used // (1024 * 1024),
            "util_pct": int(util.gpu),
            "temp_c": temp,
        }
    except Exception:
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def collect_system_info() -> SystemInfo:
    vm = psutil.virtual_memory()
    return SystemInfo(
        hostname=socket.gethostname(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        slurm_node=os.environ.get("SLURMD_NODENAME") or os.environ.get("SLURM_NODELIST"),
        cpu_count=psutil.cpu_count(logical=True) or 0,
        ram_total_mb=vm.total // (1024 * 1024),
        ram_available_mb=vm.available // (1024 * 1024),
        gpus=_gpus(),
    )
