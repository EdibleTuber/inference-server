"""Tests for GPU information retrieval via nvidia-smi."""
import pytest
from unittest.mock import patch, MagicMock
from manager.gpu import get_gpu_info


SAMPLE_NVIDIA_SMI_OUTPUT = """index, gpu_name, memory.total [MiB], memory.used [MiB]
0, Tesla P40, 24576 MiB, 18200 MiB"""


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


_TWO_GPU_CSV = (
    "index, name, memory.total [MiB], memory.used [MiB]\n"
    "0, Tesla PG500-216, 32768 MiB, 19039 MiB\n"
    "1, Tesla P40, 24576 MiB, 20710 MiB\n"
)


def _run(stdout):
    class R:
        pass
    r = R()
    r.stdout = stdout
    return r


def test_reports_every_gpu():
    with patch("manager.gpu.subprocess.run", return_value=_run(_TWO_GPU_CSV)):
        info = get_gpu_info()
    assert len(info["gpus"]) == 2
    assert info["gpus"][1]["name"] == "Tesla P40"
    assert info["gpus"][1]["vram_total_mb"] == 24576
    assert info["gpus"][1]["vram_used_mb"] == 20710


def test_gpus_carry_their_index():
    with patch("manager.gpu.subprocess.run", return_value=_run(_TWO_GPU_CSV)):
        info = get_gpu_info()
    assert [g["index"] for g in info["gpus"]] == [0, 1]


def test_top_level_keys_mirror_the_first_gpu():
    """Backwards compatibility for any consumer of the single-GPU shape."""
    with patch("manager.gpu.subprocess.run", return_value=_run(_TWO_GPU_CSV)):
        info = get_gpu_info()
    assert info["name"] == info["gpus"][0]["name"] == "Tesla PG500-216"
    assert info["vram_total_mb"] == info["gpus"][0]["vram_total_mb"]


def test_single_gpu_still_reports_one_entry():
    csv = "index, name, memory.total [MiB], memory.used [MiB]\n0, Tesla PG500-216, 32768 MiB, 19039 MiB\n"
    with patch("manager.gpu.subprocess.run", return_value=_run(csv)):
        info = get_gpu_info()
    assert len(info["gpus"]) == 1
    assert info["name"] == "Tesla PG500-216"


def test_skipped_row_does_not_renumber_the_survivor():
    """A short/unparseable first row must not shift the second physical
    card's reported index -- index now comes from nvidia-smi's own index
    column (manager/gpu.py), not from len(gpus), so a skipped row can no
    longer cause GPU 1 to be misreported as index 0."""
    csv = (
        "index, name, memory.total [MiB], memory.used [MiB]\n"
        "0, bad row\n"
        "1, Tesla P40, 24576 MiB, 20710 MiB\n"
    )
    with patch("manager.gpu.subprocess.run", return_value=_run(csv)):
        info = get_gpu_info()
    assert len(info["gpus"]) == 1
    assert info["gpus"][0]["index"] == 1
    assert info["gpus"][0]["name"] == "Tesla P40"


def test_nvidia_smi_failure_returns_empty_gpu_list():
    with patch("manager.gpu.subprocess.run", side_effect=FileNotFoundError):
        info = get_gpu_info()
    assert info["gpus"] == []
    assert info["name"] == "unknown"
