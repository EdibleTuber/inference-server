import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

def _list_slots(env_file: Path) -> list[str]:
    out = subprocess.run(
        ["bash", str(REPO / "scripts/doctor.sh"), "--list-slots"],
        env={**os.environ, "DOCTOR_MGR_ENV": str(env_file)},
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.splitlines()

def test_list_slots_matches_the_template_env():
    lines = _list_slots(REPO / "config/manager.env")
    assert lines == [
        "main llama-server.service /etc/llama/llama-server.env",
        "egpu llama-server-egpu.service /etc/llama/llama-server-egpu.env",
    ]

def test_list_slots_honours_legacy_main_keys(tmp_path):
    env_file = tmp_path / "manager.env"
    env_file.write_text("SLOTS=main\nLLAMA_SERVER_ENV=/tmp/x.env\n")
    lines = _list_slots(env_file)
    assert len(lines) == 1
    assert lines[0].split()[0] == "main"
    assert lines[0].endswith("/tmp/x.env")
