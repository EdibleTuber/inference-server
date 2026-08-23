"""
GPU information retrieval via nvidia-smi.

Provides GPU name and VRAM usage for the /status endpoint. Queries
on-demand since VRAM usage changes as models load/unload. Falls back
to safe defaults if nvidia-smi is unavailable (e.g., during development).
"""
import asyncio
import subprocess
import logging

logger = logging.getLogger(__name__)


async def get_gpu_info_async() -> dict:
    """Async wrapper: runs get_gpu_info() in an executor.

    get_gpu_info() shells out via subprocess.run, which blocks the calling
    thread. Called from an async handler, that would freeze the single
    event loop (and every queued request behind it) for as long as
    nvidia-smi takes to return.
    """
    return await asyncio.get_running_loop().run_in_executor(None, get_gpu_info)


def get_gpu_info() -> dict:
    """Query NVIDIA GPU for name and VRAM usage.

    Returns dict with name, vram_total_mb, vram_used_mb.
    Falls back to "unknown"/0 if nvidia-smi fails.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=gpu_name,memory.total,memory.used",
                "--format=csv",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        lines = result.stdout.strip().split("\n")
        if len(lines) < 2:
            logger.warning("nvidia-smi unexpected output: %s", result.stdout)
            return _unknown_gpu()

        values = [v.strip() for v in lines[1].split(",")]
        return {
            "name": values[0],
            "vram_total_mb": int(values[1].replace(" MiB", "")),
            "vram_used_mb": int(values[2].replace(" MiB", "")),
        }

    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        logger.warning("Could not query GPU info: %s", e)
        return _unknown_gpu()


def _unknown_gpu() -> dict:
    return {"name": "unknown", "vram_total_mb": 0, "vram_used_mb": 0}
