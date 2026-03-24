# manager/swap.py
"""
Model swap orchestration for the model manager.

Handles the three-step process of switching models:
1. Update llama-server's env file with the new MODEL_PATH
2. Restart llama-server via systemctl (requires sudoers entry)
3. Poll llama-server's health endpoint until it's ready

This is the reason _llama-mgr needs a sudoers entry — it's the only
place that calls systemctl restart.
"""
import asyncio
import re
import subprocess
import logging
import httpx

from manager.config import ManagerConfig

logger = logging.getLogger(__name__)


class ModelSwapper:
    """Orchestrates model swaps: env file update, systemd restart, health poll."""

    def __init__(self, config: ManagerConfig):
        self._config = config

    def update_env_file(self, model_path: str) -> None:
        """Rewrite MODEL_PATH in the llama-server env file.

        Preserves all other env vars. Uses regex replacement so it works
        whether MODEL_PATH is empty or has an existing value.
        """
        env_path = self._config.llama_server_env

        with open(env_path, "r") as f:
            content = f.read()

        content = re.sub(
            r"^MODEL_PATH=.*$",
            f"MODEL_PATH={model_path}",
            content,
            flags=re.MULTILINE,
        )

        with open(env_path, "w") as f:
            f.write(content)

        logger.info("Updated env file: MODEL_PATH=%s", model_path)

    async def restart_llama_server(self) -> None:
        """Restart llama-server via systemctl.

        Runs in a thread executor to avoid blocking the async event loop.
        Requires sudoers entry for _llama-mgr.
        """
        logger.info("Restarting llama-server...")

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: subprocess.run(
                ["sudo", "systemctl", "restart", "llama-server.service"],
                capture_output=True,
                text=True,
                timeout=30,
            ),
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to restart llama-server: {result.stderr}"
            )

        logger.info("llama-server restart command succeeded")

    async def wait_for_health(self) -> bool:
        """Poll llama-server's health endpoint until it responds.

        Returns True if healthy within timeout, False if timed out.
        Polls every 2 seconds.
        """
        url = f"{self._config.llama_server_url}/health"
        timeout = self._config.swap_timeout
        poll_interval = 2

        logger.info("Waiting for llama-server at %s (timeout: %ds)", url, timeout)

        elapsed = 0
        async with httpx.AsyncClient() as client:
            while elapsed < timeout:
                try:
                    response = await client.get(url, timeout=5)
                    if response.status_code == 200:
                        logger.info("llama-server healthy after %ds", elapsed)
                        return True
                except (httpx.ConnectError, httpx.TimeoutException, ConnectionError):
                    pass

                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

        logger.error("Health check timed out after %ds", timeout)
        return False

    async def swap_to(self, model_path: str) -> bool:
        """Execute the full model swap sequence.

        Returns True if swap succeeded, False if health check timed out.
        """
        logger.info("Starting model swap to: %s", model_path)

        self.update_env_file(model_path)
        await self.restart_llama_server()
        healthy = await self.wait_for_health()

        if healthy:
            logger.info("Model swap complete: %s", model_path)
        else:
            logger.error("Model swap failed — health timed out: %s", model_path)

        return healthy
