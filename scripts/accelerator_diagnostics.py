from __future__ import annotations

import argparse
import json
import platform
import sys


def collect_diagnostics() -> dict:
    """Collect platform and device diagnostics without credentials."""
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
    }

    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda or "unknown"
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            info["gpu_memory"] = [torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())]
        else:
            info["cuda_version"] = "N/A"
            info["gpu_count"] = 0
            info["gpu_names"] = []
            info["gpu_memory"] = []
    except ImportError:
        info["torch"] = "not installed"
        info["cuda_available"] = False
        info["cuda_version"] = "N/A"
        info["gpu_count"] = 0
        info["gpu_names"] = []
        info["gpu_memory"] = []

    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except ImportError:
        info["ram_gb"] = "unknown"

    return info


def format_text(info: dict) -> str:
    """Format diagnostics as human-readable text."""
    lines = [
        "=== Genesis Accelerator Diagnostics ===",
        f"Platform:   {info['platform']}",
        f"Python:     {info['python']}",
        f"Machine:    {info['machine']}",
        f"Processor:  {info['processor']}",
        f"RAM:        {info['ram_gb']} GB" if isinstance(info["ram_gb"], (int, float)) else f"RAM:        {info['ram_gb']}",
        "",
        "=== PyTorch ===",
        f"Version:    {info['torch']}",
        f"CUDA:       {'available' if info['cuda_available'] else 'not available'}",
        f"CUDA ver:   {info['cuda_version']}",
    ]

    if info["gpu_count"] > 0:
        lines.append(f"GPU count:  {info['gpu_count']}")
        for i, (name, mem) in enumerate(zip(info["gpu_names"], info["gpu_memory"])):
            mem_gb = round(mem / (1024 ** 3), 2) if isinstance(mem, (int, float)) else mem
            lines.append(f"  GPU {i}:    {name} ({mem_gb} GB)")
    else:
        lines.append("GPU count:  0 (CPU-only mode)")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Genesis accelerator diagnostics report.")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--check", action="store_true", help="Verify diagnostics can be collected")
    args = parser.parse_args()

    info = collect_diagnostics()

    if args.check:
        print("diagnostics collection successful")
        return

    if args.json:
        print(json.dumps(info, indent=2))
    else:
        print(format_text(info))


if __name__ == "__main__":
    main()
