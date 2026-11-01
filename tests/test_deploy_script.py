# tests/test_deploy_script.py
"""Behavioural tests for scripts/deploy-manager.sh.

Each test runs the script from a throwaway clone of this repo, with
DEPLOY_DEST/DEPLOY_MGR_ENV/DEPLOY_PROFILES_DEST/DEPLOY_MODELS_DIR pointed at
tmp paths: the real checkout, the deploy destination and the host are never
touched, and nothing here needs root (the paths exercised exit before the
root check). The clone gets one committed, valid profile, so profile
validation passes unless a test breaks it.
"""
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

GIT_ID = ["-c", "user.name=t", "-c", "user.email=t@t"]

SAMPLING = ("temp = 1.0\ntop_k = 20\ntop_p = 0.95\nmin_p = 0.0\n"
            "presence_penalty = 0.0\nfrequency_penalty = 0.0\nrepeat_penalty = 1.0\n")


def _profile(*slots: str) -> str:
    text = f'home = "{slots[0]}"\n'
    for slot in slots:
        text += (f"\n[slots.{slot}]\nctx_size = 4096\nn_gpu_layers = -1\n"
                 f"\n[slots.{slot}.sampling]\n{SAMPLING}")
    return text


GOOD = "Good-Model-Q4"


def _commit(clone, files: dict[str, str]):
    for rel, text in files.items():
        path = clone / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    subprocess.run(["git", "-C", str(clone), "add", *files], check=True)
    subprocess.run(["git", *GIT_ID, "-C", str(clone), "commit", "-q", "-m", "test"], check=True)


def _clone(tmp_path):
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(REPO), str(clone)], check=True)
    # The clone carries the committed tree (clean, known HEAD); the script
    # under test is the working tree's, so the tests track uncommitted edits.
    _commit(clone, {
        "scripts/deploy-manager.sh": (REPO / "scripts" / "deploy-manager.sh").read_text(encoding="utf-8"),
        f"config/models/{GOOD}.toml": _profile("main"),
    })
    models = tmp_path / "models"
    models.mkdir()
    (models / f"{GOOD}.gguf").write_bytes(b"GGUF")
    return clone


def _run_script(clone, *args, tmp_path, mgr_env="SLOTS=main\n", **overrides):
    (tmp_path / "manager.env").write_text(mgr_env, encoding="utf-8")
    env = {"PATH": "/usr/bin:/bin",
           "DEPLOY_DEST": str(tmp_path / "dest"),
           "DEPLOY_MGR_ENV": str(tmp_path / "manager.env"),
           "DEPLOY_PROFILES_DEST": str(tmp_path / "models.d"),
           "DEPLOY_MODELS_DIR": str(tmp_path / "models"),
           "DEPLOY_PYTHON": sys.executable}
    env.update({k: str(v) for k, v in overrides.items() if v is not None})
    for k in [k for k, v in overrides.items() if v is None]:
        env.pop(k)
    return subprocess.run(
        ["bash", str(clone / "scripts/deploy-manager.sh"), *args],
        capture_output=True, text=True, env=env,
    )


def _deploy_everything(clone, tmp_path):
    """Make dest and models.d match the clone, as a finished deploy would."""
    dest = tmp_path / "dest"
    dest.mkdir(exist_ok=True)
    for src in [*(clone / "manager").glob("*.py"),
                clone / "manager" / "requirements.txt", clone / "manager" / "requirements.lock"]:
        if src.exists():
            shutil.copy(src, dest / src.name)
    profiles = tmp_path / "models.d"
    profiles.mkdir(exist_ok=True)
    for src in (clone / "config" / "models").glob("*.toml"):
        shutil.copy(src, profiles / src.name)


def _line(out: str, name: str) -> str:
    lines = [l for l in out.splitlines() if l.split()[:1] == [name]]
    assert len(lines) == 1, out
    return lines[0]


def test_deploy_refuses_a_dirty_tree_before_asking_for_root(tmp_path):
    clone = _clone(tmp_path)
    (clone / "manager" / "app.py").write_text("# dirty\n", encoding="utf-8")
    r = _run_script(clone, tmp_path=tmp_path)
    assert r.returncode == 3, r.stdout + r.stderr
    assert "uncommitted" in r.stdout


def test_check_reports_an_absent_deployed_file_as_stale(tmp_path):
    # Tree is clean; DEST is an empty dir, so every deployed file is absent
    # and must be reported STALE instead of killing the script.
    clone = _clone(tmp_path)
    (tmp_path / "dest").mkdir()
    r = _run_script(clone, "--check", tmp_path=tmp_path)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "STALE" in r.stdout


# --- profiles in the drift report ---------------------------------------------

def test_check_is_clean_when_code_and_profiles_match(tmp_path):
    clone = _clone(tmp_path)
    _deploy_everything(clone, tmp_path)
    r = _run_script(clone, "--check", tmp_path=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "current" in _line(r.stdout, f"{GOOD}.toml")


def test_check_reports_an_undeployed_profile_as_stale(tmp_path):
    clone = _clone(tmp_path)
    _deploy_everything(clone, tmp_path)
    (tmp_path / "models.d" / f"{GOOD}.toml").unlink()
    r = _run_script(clone, "--check", tmp_path=tmp_path)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "STALE" in _line(r.stdout, f"{GOOD}.toml")
    assert "absent" in _line(r.stdout, f"{GOOD}.toml")


def test_check_reports_a_profile_edited_on_the_host_as_stale(tmp_path):
    clone = _clone(tmp_path)
    _deploy_everything(clone, tmp_path)
    with (tmp_path / "models.d" / f"{GOOD}.toml").open("a") as f:
        f.write("# edited on the host\n")
    r = _run_script(clone, "--check", tmp_path=tmp_path)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "STALE" in _line(r.stdout, f"{GOOD}.toml")


def test_check_reports_a_profile_left_behind_on_the_host(tmp_path):
    """Review Focus 3: a profile deleted from the repo but still in
    models.d would keep serving. It is drift, even with all else current."""
    clone = _clone(tmp_path)
    _deploy_everything(clone, tmp_path)
    (tmp_path / "models.d" / "Gone-Model-Q4.toml").write_text(_profile("main"))
    r = _run_script(clone, "--check", tmp_path=tmp_path)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "EXTRA (removed on deploy)" in _line(r.stdout, "Gone-Model-Q4.toml")
    assert "current" in _line(r.stdout, f"{GOOD}.toml")


def test_a_missing_profiles_dest_reports_every_profile_absent(tmp_path):
    clone = _clone(tmp_path)
    _deploy_everything(clone, tmp_path)
    shutil.rmtree(tmp_path / "models.d")
    r = _run_script(clone, "--check", tmp_path=tmp_path)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "absent" in _line(r.stdout, f"{GOOD}.toml")


# --- profile validation: exit 5, in --check too, before anything installs ------

def test_an_invalid_profile_exits_5_even_with_no_drift(tmp_path):
    clone = _clone(tmp_path)
    _commit(clone, {f"config/models/{GOOD}.toml": _profile("main").replace("ctx_size = 4096\n", "")})
    _deploy_everything(clone, tmp_path)
    r = _run_script(clone, "--check", tmp_path=tmp_path)
    assert r.returncode == 5, r.stdout + r.stderr
    assert "profile validation failed; not deploying" in r.stdout
    assert "ctx_size" in r.stdout + r.stderr          # the validator's own reason is shown


def test_an_invalid_profile_exits_5_before_the_root_check(tmp_path):
    clone = _clone(tmp_path)
    _commit(clone, {"config/models/Broken-Q4.toml": "not toml ["})
    (tmp_path / "models" / "Broken-Q4.gguf").write_bytes(b"GGUF")
    r = _run_script(clone, tmp_path=tmp_path)
    assert r.returncode == 5, r.stdout + r.stderr
    assert not (tmp_path / "models.d").exists()


def test_validation_uses_the_slots_from_manager_env(tmp_path):
    clone = _clone(tmp_path)
    _commit(clone, {f"config/models/{GOOD}.toml": _profile("main", "egpu")})
    _deploy_everything(clone, tmp_path)
    assert _run_script(clone, "--check", tmp_path=tmp_path, mgr_env="SLOTS=main\n").returncode == 5
    assert _run_script(clone, "--check", tmp_path=tmp_path, mgr_env="SLOTS=main,egpu\n").returncode == 0


def test_slots_default_to_main(tmp_path):
    clone = _clone(tmp_path)
    _deploy_everything(clone, tmp_path)
    assert _run_script(clone, "--check", tmp_path=tmp_path, mgr_env="").returncode == 0


def test_validation_needs_each_gguf_in_the_models_dir(tmp_path):
    clone = _clone(tmp_path)
    _deploy_everything(clone, tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    r = _run_script(clone, "--check", tmp_path=tmp_path, DEPLOY_MODELS_DIR=empty)
    assert r.returncode == 5, r.stdout + r.stderr


def test_the_models_dir_defaults_to_manager_envs_models_dir(tmp_path):
    clone = _clone(tmp_path)
    _deploy_everything(clone, tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    env_text = "SLOTS=main\nMODELS_DIR={}\n"
    assert _run_script(clone, "--check", tmp_path=tmp_path, DEPLOY_MODELS_DIR=None,
                       mgr_env=env_text.format(tmp_path / "models")).returncode == 0
    assert _run_script(clone, "--check", tmp_path=tmp_path, DEPLOY_MODELS_DIR=None,
                       mgr_env=env_text.format(empty)).returncode == 5


def test_a_repo_with_no_profiles_exits_5(tmp_path):
    clone = _clone(tmp_path)
    subprocess.run(["git", "-C", str(clone), "rm", "-q", "-r", "config/models"], check=True)
    subprocess.run(["git", *GIT_ID, "-C", str(clone), "commit", "-q", "-m", "rm"], check=True)
    _deploy_everything(clone, tmp_path)
    r = _run_script(clone, "--check", tmp_path=tmp_path)
    assert r.returncode == 5, r.stdout + r.stderr


# --- only committed profiles ship -----------------------------------------------

def test_deploy_refuses_an_untracked_profile(tmp_path):
    """The dirty-tree check ignores untracked files, and a profile dropped
    into config/models/ but never committed would ship unreviewed."""
    clone = _clone(tmp_path)
    (clone / "config" / "models" / "Stray-Q4.toml").write_text(_profile("main"))
    (tmp_path / "models" / "Stray-Q4.gguf").write_bytes(b"GGUF")
    r = _run_script(clone, tmp_path=tmp_path)
    assert r.returncode == 3, r.stdout + r.stderr
    assert "Stray-Q4.toml" in r.stdout
