"""
Model swap orchestration for one slot.

A ModelSwapper instance is bound to one backing llama-server process via
its SlotState and operates on that slot's generated model env file and
systemd unit. The caller renders the model's profile section for the slot
into env text (see manager.profiles.render_model_env); a swap writes that
file atomically, restarts the service, and polls for health. Which slots
exist is configuration; the manager instantiates one swapper per
configured slot at startup.
"""
import asyncio
import logging
import os
import subprocess
import tempfile
import time

import httpx

from manager.config import ManagerConfig
from manager.slots import SlotState

logger = logging.getLogger(__name__)


def write_model_env(path: str, text: str) -> None:
    """Atomically publish `text` at `path` (Q32).

    The content goes to a temp file in the target's own directory, is
    flushed and fsynced, chmodded 0644, and os.replace'd over the target,
    so readers only ever see a complete file and the target is never
    written in place. On any error the temp file is unlinked and the
    exception re-raised: the old file stays intact.
    """
    directory = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class ModelSwapper:
    """Orchestrates swap for one slot: env rewrite, systemd restart, health poll."""

    def __init__(self, config: ManagerConfig, slot: SlotState):
        self._config = config
        self._slot = slot

    async def restart_llama_server(self) -> None:
        """Restart the slot's systemd unit via sudo."""
        unit = self._slot.systemd_unit
        logger.info("slot=%s restarting %s", self._slot.name, unit)

        timeout = self._config.restart_timeout
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    ["sudo", "systemctl", "restart", unit],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                ),
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Restart of {unit} timed out after {timeout}s"
            ) from exc

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

    async def swap_to(self, model_name: str, env_text: str) -> bool:
        """Full swap sequence. Returns True on success, False on health timeout.

        Writes the slot's generated model env file, restarts the service,
        and polls health, in that order. If the write raises, no restart
        happens and the exception propagates: the caller
        (ensure_model_on_slot) turns a raising swap into a failed swap.
        """
        logger.info("slot=%s starting swap to %s", self._slot.name, model_name)
        write_model_env(self._slot.model_env_file, env_text)
        await self.restart_llama_server()
        healthy = await self.wait_for_health()

        if healthy:
            logger.info("slot=%s swap complete: %s", self._slot.name, model_name)
        else:
            logger.error("slot=%s swap failed (health timeout): %s", self._slot.name, model_name)

        return healthy
