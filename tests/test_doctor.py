import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

def _list_slots(env_file: Path, cwd: Path | None = None) -> list[str]:
    out = subprocess.run(
        ["bash", str(REPO / "scripts/doctor.sh"), "--list-slots"],
        env={**os.environ, "DOCTOR_MGR_ENV": str(env_file)},
        capture_output=True, text=True, cwd=cwd,
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

def test_list_slots_ignores_a_manager_package_in_the_callers_cwd(tmp_path):
    """A stdin script gets the cwd on sys.path ahead of PYTHONPATH, so from a
    directory holding another manager/ doctor read that copy's slot config."""
    decoy = tmp_path / "manager"
    decoy.mkdir()
    (decoy / "__init__.py").write_text("")
    (decoy / "slot_config.py").write_text("raise SystemExit('decoy slot_config imported')\n")
    lines = _list_slots(REPO / "config/manager.env", cwd=tmp_path)
    assert [line.split()[0] for line in lines] == ["main", "egpu"]
