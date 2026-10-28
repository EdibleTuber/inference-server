#!/usr/bin/env python3
"""Run the audit inventory passes against a local model and verify citations.

Requests go one at a time with stream=true, so they never hit the manager's
120s non-streaming timeout (incident 2026-10-27, F4).

Every entry whose `quote` is not on its cited line is moved to *.rejects.jsonl.
"""
import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PROMPTS = HERE / "prompts"

PER_FILE_PASSES = ["p1_timeouts_waits", "p2_exceptions_tasks", "p3_state_writes",
                   "p4_external_effects", "p5_claims"]
MANAGER_FILES = sorted(p for p in (REPO / "manager").glob("*.py") if p.stat().st_size)
CONFIG_FILES = (
    [REPO / "manager/config.py", REPO / "manager/slot_config.py"]
    + sorted((REPO / "config").glob("*.env"))
    + sorted((REPO / "systemd").glob("*.service"))
    + sorted((REPO / "scripts").glob("*.sh"))
    + sorted(Path("/etc/llama").glob("*.env"))
    + sorted(Path("/etc/systemd/system").glob("llama-*.service"))
)


def numbered(path: Path) -> list[str]:
    return path.read_text(errors="replace").splitlines()


def render(path: Path, lines: list[str]) -> str:
    return f"===== FILE: {path} =====\n" + "\n".join(
        f"{i:04d}| {line}" for i, line in enumerate(lines, 1))


class Incomplete(Exception):
    pass


def ask(url: str, model: str, prompt: str, max_tokens: int, think: bool) -> str:
    body = json.dumps({
        "model": model, "stream": True, "temperature": 0.3, "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "chat_template_kwargs": {"enable_thinking": think},
    }).encode()
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=body,
                                 headers={"content-type": "application/json"})
    out, finish = [], None
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            choice = json.loads(line[5:])["choices"][0]
            finish = choice.get("finish_reason") or finish
            delta = choice.get("delta", {})
            if delta.get("content"):
                out.append(delta["content"])
                print(delta["content"], end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    text = "".join(out)
    # An empty or length-capped answer is a failed run, not an empty inventory.
    if finish != "stop" or not text.strip():
        raise Incomplete(f"finish_reason={finish!r}, content_chars={len(text)}")
    return text


def verify(text: str, sources: dict[str, list[str]], default_file: str | None):
    good, bad = [], []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            # The model echoes the zero-padded line prefix ("line": 0025), which is invalid JSON.
            obj = json.loads(re.sub(r'(":\s*)0+(\d)', r"\1\2", raw))
        except json.JSONDecodeError:
            bad.append({"reason": "not_json", "raw": raw})
            continue
        if "line" not in obj or "quote" not in obj:
            good.append(obj)  # summary / none rows carry no citation
            continue
        fname = obj.get("file", default_file)
        lines = sources.get(fname)
        n = obj["line"]
        ok = (lines is not None and isinstance(n, int) and 1 <= n <= len(lines)
              and obj["quote"].strip() and obj["quote"].strip() in lines[n - 1])
        (good if ok else bad).append(obj if ok else {"reason": "quote_mismatch", **obj})
        if ok and default_file:
            obj.setdefault("file", fname)
    return good, bad


def write(out_dir: Path, name: str, good, bad):
    for suffix, rows in ((".jsonl", good), (".rejects.jsonl", bad)):
        (out_dir / f"{name}{suffix}").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{name}: {len(good)} kept, {len(bad)} rejected", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://192.168.1.14:11434")
    ap.add_argument("--model", default="Qwen3.8-27B-UD-Q6_K")
    ap.add_argument("--passes", nargs="*", default=PER_FILE_PASSES + ["p6_config_surface"])
    ap.add_argument("--files", nargs="*", help="limit per-file passes to these basenames")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--think", action="store_true", help="enable model reasoning (default off)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    common = (PROMPTS / "_common.md").read_text()
    failures = []

    def run(name, prompt, sources, default_file):
        for stale in out_dir.glob(f"{name}.*"):
            stale.unlink()
        try:
            text = ask(args.url, args.model, prompt, args.max_tokens, args.think)
        except Incomplete as exc:
            (out_dir / f"{name}.FAILED").write_text(str(exc) + "\n")
            failures.append(name)
            print(f"{name}: FAILED ({exc})", flush=True)
            return
        good, bad = verify(text, sources, default_file)
        write(out_dir, name, good, bad)

    for p in args.passes:
        task = (PROMPTS / f"{p}.md").read_text()
        if p == "p6_config_surface":
            files = [f for f in CONFIG_FILES if f.is_file()]
            sources = {str(f): numbered(f) for f in files}
            prompt = common + "\n" + task + "\n\n" + "\n\n".join(
                render(f, sources[str(f)]) for f in files)
            run(p, prompt, sources, None)
            continue
        for f in MANAGER_FILES:
            if args.files and f.name not in args.files:
                continue
            lines = numbered(f)
            prompt = common + "\n" + task + "\n\n" + render(f, lines)
            run(f"{p}__{f.stem}", prompt, {str(f): lines}, str(f))

    if failures:
        print(f"\n{len(failures)} FAILED: {' '.join(failures)}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
