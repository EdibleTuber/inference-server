# tests/test_deploy_script.py
"""Behavioural tests for scripts/deploy-manager.sh.

Each test runs the script from a throwaway clone of this repo, with
DEPLOY_DEST/DEPLOY_MGR_ENV pointed at tmp paths: the real checkout, the
deploy destination and the host are never touched, and nothing here needs
root (the paths exercised exit before the root check).
"""
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _clone(tmp_path):
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(REPO), str(clone)], check=True)
    # The clone carries the committed tree (clean, known HEAD); the script
    # under test is the working tree's, so the tests track uncommitted edits.
    (clone / "scripts" / "deploy-manager.sh").write_text(
        (REPO / "scripts" / "deploy-manager.sh").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return clone


def _run_script(clone, *args, tmp_path):
    return subprocess.run(
        ["bash", str(clone / "scripts/deploy-manager.sh"), *args],
        capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin",
             "DEPLOY_DEST": str(tmp_path / "dest"),
             "DEPLOY_MGR_ENV": str(tmp_path / "none.env")},
    )


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
