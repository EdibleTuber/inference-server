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
    """Query NVIDIA GPUs for name and VRAM usage.

    Returns dict with a "gpus" list (one entry per GPU nvidia-smi reports,
    each with index/name/vram_total_mb/vram_used_mb) plus top-level
    name/vram_total_mb/vram_used_mb mirroring the first GPU, for consumers
    of the original single-GPU shape.

    Falls back to an empty "gpus" list and "unknown"/0 top-level values if
    nvidia-smi fails or its output can't be parsed.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,gpu_name,memory.total,memory.used",
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

        gpus = []
        for line in lines[1:]:
            if not line.strip():
                continue
            values = [v.strip() for v in line.split(",")]
            if len(values) < 4:
                logger.warning("nvidia-smi unexpected row: %s", line)
                continue
            try:
                gpus.append(
                    {
                        "index": int(values[0]),
                        "name": values[1],
                        "vram_total_mb": int(values[2].replace(" MiB", "")),
                        "vram_used_mb": int(values[3].replace(" MiB", "")),
                    }
                )
            except ValueError:
                logger.warning("nvidia-smi unparseable row: %s", line)
                continue

        if not gpus:
            return _unknown_gpu()

        return {
            "gpus": gpus,
            "name": gpus[0]["name"],
            "vram_total_mb": gpus[0]["vram_total_mb"],
            "vram_used_mb": gpus[0]["vram_used_mb"],
        }

    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        logger.warning("Could not query GPU info: %s", e)
        return _unknown_gpu()


def _unknown_gpu() -> dict:
    return {"gpus": [], "name": "unknown", "vram_total_mb": 0, "vram_used_mb": 0}
