"""Tests for GPU information retrieval via nvidia-smi."""
import pytest
from unittest.mock import patch, MagicMock
from manager.gpu import get_gpu_info


SAMPLE_NVIDIA_SMI_OUTPUT = """gpu_name, memory.total [MiB], memory.used [MiB]
Tesla P40, 24576 MiB, 18200 MiB"""


def test_parse_nvidia_smi_output():
    """Should parse GPU name, total VRAM, and used VRAM."""
    mock_result = MagicMock()
    mock_result.stdout = SAMPLE_NVIDIA_SMI_OUTPUT
    mock_result.returncode = 0

    with patch("subprocess.run", return_value=mock_result):
        info = get_gpu_info()

    assert info["name"] == "Tesla P40"
    assert info["vram_total_mb"] == 24576
    assert info["vram_used_mb"] == 18200


def test_gpu_info_when_nvidia_smi_fails():
    """Should return unknown values if nvidia-smi is not available."""
    with patch("subprocess.run", side_effect=FileNotFoundError):
        info = get_gpu_info()

    assert info["name"] == "unknown"
    assert info["vram_total_mb"] == 0
    assert info["vram_used_mb"] == 0
