from __future__ import annotations

import platform
import sys

import torch


def collect_diagnostics() -> dict[str, object]:
    cuda_available = torch.cuda.is_available()

    if cuda_available:
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
    else:
        device = "cpu"
        device_name = platform.processor() or "CPU"
        cuda_version = None

    return {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "device": device,
        "device_name": device_name,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
    }


def format_diagnostics(info: dict[str, object]) -> str:
    lines = [
        "Genesis AI accelerator diagnostics",
        f"Platform: {info['platform']} {info['platform_release']}",
        f"Python: {info['python_version']}",
        f"PyTorch: {info['pytorch_version']}",
        f"Device: {info['device']}",
        f"Device name: {info['device_name']}",
        f"CUDA available: {info['cuda_available']}",
        f"CUDA version: {info['cuda_version'] or 'not available'}",
    ]

    if not info["cuda_available"]:
        lines.append(
            "Accelerator status: CPU-only runtime detected; "
            "GPU acceleration is not available."
        )

    return "\n".join(lines)


def main() -> int:
    print(format_diagnostics(collect_diagnostics()))
    return 0


if __name__ == "__main__":
    sys.exit(main())