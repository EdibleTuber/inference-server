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


def test_the_template_names_the_profile_and_model_env_dirs_the_code_defaults_to(monkeypatch):
    """manager.env spells out PROFILES_DIR and MODEL_ENV_DIR, and they are the
    code's own defaults, so the template and an unset key agree."""
    from manager.config import ManagerConfig
    env = _template_env()
    for key in ("PROFILES_DIR", "MODEL_ENV_DIR", "SLOTS"):
        monkeypatch.delenv(key, raising=False)
    assert env["PROFILES_DIR"] == ManagerConfig.from_env().profiles_dir
    default_main = build_slots(getenv=lambda k: None)[0]
    assert default_main.model_env_file == f"{env['MODEL_ENV_DIR']}/main.env"


def test_the_template_has_no_pre_profile_settings():
    text = (REPO / "config/manager.env").read_text()
    assert not [k for k in _template_env() if re.fullmatch(r"SLOT_\w+_MODELS", k)]
    # The home map's comment and the in-place MODEL_PATH rewrite are both gone.
    assert "home_slot" not in text
    assert "MODEL_PATH" not in text
