import os, re, subprocess
from pathlib import Path
from manager.slot_config import build_slots

REPO = Path(__file__).resolve().parent.parent

def _template_env() -> dict[str, str]:
    env = {}
    for line in (REPO / "config/manager.env").read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env

def test_every_template_slot_has_its_unit_and_env_file_in_the_repo():
    for s in build_slots(getenv=_template_env().get):
        assert (REPO / "systemd" / s.systemd_unit).is_file(), s.systemd_unit
        assert (REPO / "config" / os.path.basename(s.env_file)).is_file(), s.env_file

def test_no_live_file_names_the_old_re_slot():
    # Historical records under docs/ keep their wording; everything else must not.
    # The [E]/[e] classes stop the pattern matching this file's own source.
    out = subprocess.run(
        ["git", "-C", str(REPO), "grep", "-n", "-E", r"SLOT_R[E]_|llama-server-r[e]\b",
         "--", ".", ":!docs/"],
        capture_output=True, text=True,
    )
    assert out.stdout == "", out.stdout
