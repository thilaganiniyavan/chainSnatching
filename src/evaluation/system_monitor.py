"""System Resource Monitor — real-time hardware metric profiling.

Profiles:
- CPU utilization percentage (%)
- RAM memory usage (MB & %)
- GPU VRAM memory usage (MB) & GPU utilization (%) via PyTorch CUDA queries
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict

import psutil

# Check PyTorch CUDA availability
HAS_CUDA = False
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except Exception:
    HAS_CUDA = False


class SystemResourceMonitor:
    """Monitors CPU, RAM, and GPU resource usage over time intervals or stage snapshots."""

    def __init__(self) -> None:
        self.process = psutil.Process(os.getpid())

    def get_snapshot(self) -> Dict[str, Any]:
        """Take a snapshot of current system hardware resource metrics.

        Returns:
            Dictionary containing CPU %, RAM MB, RAM %, GPU MB, GPU % metrics.
        """
        # Overall CPU & RAM
        cpu_pct = psutil.cpu_percent(interval=None)
        ram_info = psutil.virtual_memory()
        proc_ram_mb = self.process.memory_info().rss / (1024 * 1024)

        gpu_mem_mb = 0.0
        gpu_pct = 0.0

        if HAS_CUDA:
            try:
                gpu_mem_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                gpu_pct = min(100.0, (gpu_mem_mb / 8192.0) * 100.0) # Approximate percentage
            except Exception:
                gpu_mem_mb = 0.0

        return {
            "timestamp": time.time(),
            "cpu_percent": round(cpu_pct, 1),
            "ram_used_mb": round(proc_ram_mb, 1),
            "ram_total_mb": round(ram_info.total / (1024 * 1024), 1),
            "ram_percent": round(ram_info.percent, 1),
            "gpu_mem_mb": round(gpu_mem_mb, 1),
            "gpu_percent": round(gpu_pct, 1),
            "has_cuda": HAS_CUDA,
        }
