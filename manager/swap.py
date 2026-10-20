"""
Model swap orchestration for one slot.

A ModelSwapper instance is bound to one backing llama-server process via
its SlotState and operates on that slot's env file and systemd unit. Which
slots exist is configuration; the manager instantiates one swapper per
configured slot at startup.
"""
import asyncio
import re
import subprocess
import logging
import time
from pathlib import Path

import httpx

from manager.config import ManagerConfig
from manager.slots import SlotState

logger = logging.getLogger(__name__)


def sampling_args_for_model(model_path: str) -> str:
    """Read per-model sampling args from a sidecar .args file next to the GGUF.

    Returns the file contents (stripped) or empty string if no sidecar exists.
    An empty result means llama-server will use its built-in defaults — the safe
    fallback for any model the operator hasn't explicitly configured.
    """
    args_file = Path(model_path).with_suffix(".args")
    if args_file.is_file():
        return args_file.read_text().strip()
    return ""


class ModelSwapper:
    """Orchestrates swap for one slot: env rewrite, systemd restart, health poll."""

    def __init__(self, config: ManagerConfig, slot: SlotState):
        self._config = config
        self._slot = slot

    def update_env_file(self, model_path: str, sampling_args: str = "") -> None:
        """Rewrite MODEL_PATH and SAMPLING_ARGS in the slot's env file."""
        with open(self._slot.env_file, "r") as f:
            content = f.read()

        content = re.sub(
            r"^MODEL_PATH=.*$",
            f"MODEL_PATH={model_path}",
            content,
            flags=re.MULTILINE,
        )

        if re.search(r"^SAMPLING_ARGS=", content, flags=re.MULTILINE):
            content = re.sub(
                r"^SAMPLING_ARGS=.*$",
                f"SAMPLING_ARGS={sampling_args}",
                content,
                flags=re.MULTILINE,
            )
        else:
            content = re.sub(
                r"^(MODEL_PATH=.*)$",
                rf"\1\nSAMPLING_ARGS={sampling_args}",
                content,
                flags=re.MULTILINE,
            )

        with open(self._slot.env_file, "w") as f:
            f.write(content)

        logger.info("slot=%s updated env file MODEL_PATH=%s SAMPLING_ARGS=%s",
                     self._slot.name, model_path, sampling_args or "(defaults)")

    async def restart_llama_server(self) -> None:
        """Restart the slot's systemd unit via sudo."""
        unit = self._slot.systemd_unit
        logger.info("slot=%s restarting %s", self._slot.name, unit)

        result = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: subprocess.run(
                ["sudo", "systemctl", "restart", unit],
                capture_output=True,
                text=True,
                timeout=30,
            ),
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to restart {unit}: {result.stderr}"
            )

        logger.info("slot=%s restart succeeded", self._slot.name)

    async def wait_for_health(self) -> bool:
        """Poll the slot's /health until it responds or we time out."""
        url = f"{self._slot.url}/health"
        timeout = self._config.swap_timeout
        poll_interval = 2

        logger.info("slot=%s waiting for %s (timeout=%ds)", self._slot.name, url, timeout)

        start = time.monotonic()
        deadline = start + timeout
        async with httpx.AsyncClient() as client:
            while time.monotonic() < deadline:
                try:
                    response = await client.get(url, timeout=5)
                    if response.status_code == 200:
                        elapsed = int(time.monotonic() - start)
                        logger.info("slot=%s healthy after %ds", self._slot.name, elapsed)
                        return True
                except (httpx.ConnectError, httpx.TimeoutException, ConnectionError):
                    pass

                await asyncio.sleep(poll_interval)

        logger.error("slot=%s health timed out after %ds", self._slot.name, timeout)
        return False

    async def swap_to(self, model_path: str, sampling_args: str = "") -> bool:
        """Full swap sequence. Returns True on success, False on health timeout."""
        logger.info("slot=%s starting swap to %s", self._slot.name, model_path)
        self.update_env_file(model_path, sampling_args)
        await self.restart_llama_server()
        healthy = await self.wait_for_health()

        if healthy:
            logger.info("slot=%s swap complete: %s", self._slot.name, model_path)
        else:
            logger.error("slot=%s swap failed (health timeout): %s", self._slot.name, model_path)

        return healthy
