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
    # scripts/doctor.sh names the old unit on purpose: it checks it is gone.
    # The [E]/[e] classes stop the pattern matching this file's own source.
    out = subprocess.run(
        ["git", "-C", str(REPO), "grep", "-n", "-E", r"SLOT_R[E]_|llama-server-r[e]\b",
         "--", ".", ":!docs/", ":!scripts/doctor.sh"],
        capture_output=True, text=True,
    )
    assert out.stdout == "", out.stdout

def test_every_top_level_requirement_is_pinned_in_the_lock():
    def name(line): return re.split(r"[<>=\[ ]", line.strip(), maxsplit=1)[0].lower().replace("_", "-")
    reqs = [l for l in (REPO / "manager/requirements.txt").read_text().splitlines()
            if l.strip() and not l.startswith("#")]
    lock = {name(l): l for l in (REPO / "manager/requirements.lock").read_text().splitlines()
            if l.strip() and not l.startswith("#")}
    for r in reqs:
        assert name(r) in lock and "==" in lock[name(r)], r
