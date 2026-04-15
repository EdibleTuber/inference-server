"""End-to-end: POST reindex -> poll status -> docs indexed."""
import time
from pathlib import Path

import pytest


def _wait_done(client, collection_id, job_id, timeout_s=5.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = client.get(f"/collections/{collection_id}/reindex/{job_id}")
        if resp.status_code == 200 and resp.json()["status"] in ("done", "error"):
            return resp.json()
        time.sleep(0.05)
    raise AssertionError(f"reindex did not finish within {timeout_s}s")


def test_full_reindex_indexes_new_files(collection_client, skills_dir):
    """Writing a new file and triggering a full reindex indexes it."""
    new_file = Path(skills_dir) / "Security" / "Recon" / "FreshNote.md"
    new_file.write_text(
        "---\nname: FreshNote\ndescription: Brand new note added mid-session.\n---\n\n# Fresh\n"
    )

    post = collection_client.post("/collections/skills/reindex", json={})
    assert post.status_code == 202
    final = _wait_done(collection_client, "skills", post.json()["job_id"])
    assert final["status"] == "done", final

    # Document is now retrievable
    # doc_id for a file Security/Recon/FreshNote.md is "Security/Recon/FreshNote"
    resp = collection_client.get("/collections/skills/docs/Security/Recon/FreshNote")
    assert resp.status_code == 200


def test_scoped_reindex_only_indexes_given_path(collection_client, skills_dir):
    """Writing two files but scoping the reindex to one means only that one is indexed."""
    alpha = Path(skills_dir) / "Security" / "Recon" / "Alpha.md"
    beta = Path(skills_dir) / "Security" / "Recon" / "Beta.md"
    alpha.write_text("---\nname: Alpha\ndescription: Alpha note\n---\n\n# Alpha\n")
    beta.write_text("---\nname: Beta\ndescription: Beta note\n---\n\n# Beta\n")

    post = collection_client.post(
        "/collections/skills/reindex",
        json={"paths": [str(alpha)]},
    )
    assert post.status_code == 202
    final = _wait_done(collection_client, "skills", post.json()["job_id"])
    assert final["status"] == "done"

    assert collection_client.get("/collections/skills/docs/Security/Recon/Alpha").status_code == 200
    # beta.md was not in the scoped paths, so it is NOT indexed
    assert collection_client.get("/collections/skills/docs/Security/Recon/Beta").status_code == 404
