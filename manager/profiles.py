# manager/profiles.py
"""Per-model profiles: parse, validate and render (Q13, Q29, Q33).

A profile (config/models/<stem>.toml) is the whole of one model's
configuration on a slot: ctx_size, n_gpu_layers, the full sampling set,
optional extra flags, optional per-model env vars, and an optional
verified table written by scripts/onboard.py. A swap renders the profile's
section for the slot into the slot's generated env file, so no flag can
carry over from the previous model.

Validation is strict because a bad value is an injection into that
generated file: unknown keys, missing required keys, booleans posing as
ints, reserved flags, and any flag token or env value containing
whitespace, a quote, a backslash, '$' or a backtick are all errors.

This module is pure: parse_profile and render_model_env touch neither the
filesystem nor the environment. Only load_profile / load_profiles read
files, and load_profiles never raises.
"""
from __future__ import annotations

import argparse
import re
import sys
import tomllib
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path

from manager.names import match_key

#: The full sampling set, in render order: every sampler llama.cpp b11136
#: applies with a non-neutral default, plus the three penalties.
SAMPLING_KEYS: tuple[str, ...] = (
    "temp", "top_k", "top_p", "min_p",
    "presence_penalty", "frequency_penalty", "repeat_penalty",
)

_SLOT_OWNED_FLAGS = frozenset({
    "--host", "--port", "--device", "-dev",
    "--parallel", "-np",
    "--cache-ram", "-cram",
})

_RENDERER_FLAGS = frozenset({
    "--model", "-m",
    "--ctx-size", "-c",
    "--n-gpu-layers", "--gpu-layers", "-ngl",
})

_SAMPLING_FLAGS = frozenset(f"--{key.replace('_', '-')}" for key in SAMPLING_KEYS) | {"--temperature"}

#: Flags a profile's `flags` list may not contain, compared on the part
#: before any `=`. The slot owns the first group (its ExecStart), the
#: renderer the second, and the sampling flags are always rendered.
RESERVED_FLAGS: frozenset[str] = _SLOT_OWNED_FLAGS | _RENDERER_FLAGS | _SAMPLING_FLAGS

#: Env keys a profile may not set; they are owned by the slot. Keys starting
#: with LLAMA_ARG_ are rejected too: llama-server reads them as flags, which
#: would let a profile set a flag MODEL_ARGS doesn't show.
RESERVED_ENV: frozenset[str] = frozenset(
    {"HOST", "PORT", "DEVICE", "CUDA_VISIBLE_DEVICES", "MODEL_ARGS"})

_NAME_RE = re.compile(r"[A-Za-z0-9._-]+\Z")
_ENV_KEY_RE = re.compile(r"[A-Z_][A-Z0-9_]*\Z")
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")

#: Characters that would inject extra arguments or extra env lines into the
#: generated file: whitespace splits tokens, quotes and '$' escape, backtick
#: and backslash start expansions.
_FORBIDDEN = frozenset('"\'\\$`')


class ProfileError(ValueError):
    """A profile failed validation. The message names the profile and the offending key."""


@dataclass(frozen=True)
class Verified:
    """The [slots.<slot>.verified] table: a (model, slot) pair passed a trial load."""
    date: str
    llama_cpp_build: int
    llama_cpp_commit: str


@dataclass(frozen=True)
class SlotSection:
    """One [slots.<slot>] table: the whole model configuration for that slot."""
    slot: str
    ctx_size: int
    n_gpu_layers: int
    sampling: tuple[tuple[str, int | float], ...]  # in SAMPLING_KEYS order
    flags: tuple[str, ...]
    env: tuple[tuple[str, str], ...]               # sorted by key
    verified: Verified | None


@dataclass(frozen=True)
class Profile:
    """One model's profile, keyed section by section in file order."""
    name: str
    home: str
    sections: Mapping[str, SlotSection]

    def section(self, slot: str) -> SlotSection | None:
        return self.sections.get(slot)

    def verified_slots(self) -> tuple[str, ...]:
        return tuple(slot for slot in self.sections if self.sections[slot].verified is not None)


@dataclass
class ProfileSet:
    """Every profile that loaded, keyed by names.match_key(name), plus the failures."""
    profiles: dict[str, Profile]
    errors: dict[str, str]  # keyed by file stem (or the dir itself, when it is missing)

    def get(self, model: str) -> Profile | None:
        """Case- and .gguf-insensitive lookup, via names.match_key."""
        key = match_key(model)
        if key is None:
            return None
        return self.profiles.get(key)

    def onboarded(self) -> list[Profile]:
        """Profiles with at least one verified section, sorted by name."""
        return sorted(
            (p for p in self.profiles.values() if any(s.verified is not None for s in p.sections.values())),
            key=lambda p: p.name)

    def home_models(self) -> dict[str, list[str]]:
        """Slot to sorted profile names, over slots that have a home model."""
        out: dict[str, list[str]] = {}
        for p in self.profiles.values():
            out.setdefault(p.home, []).append(p.name)
        return {slot: sorted(names) for slot, names in out.items()}


# ---------------------------------------------------------------------------
# parse_profile: the validator
# ---------------------------------------------------------------------------

def _bad_char(token: str) -> str | None:
    for ch in token:
        if ch.isspace() or ch in _FORBIDDEN:
            return ch
    return None


def _parse_token(name: str, key: str, token: object) -> None:
    """Validate one flag token or env value: a non-empty string that cannot
    inject into the generated env file."""
    if not isinstance(token, str) or token == "":
        raise ProfileError(f"{name}: {key} must be a non-empty string token, got {token!r}")
    bad = _bad_char(token)
    if bad is not None:
        raise ProfileError(
            f"{name}: {key} token {token!r} contains a character that could inject "
            f"into the env file: {bad!r}")


def _parse_verified(name: str, slot: str, value: object) -> Verified | None:
    where = f"{name}: [slots.{slot}].verified"
    if not isinstance(value, dict):
        raise ProfileError(f"{where} must be a table, got {value!r}")
    unknown = set(value) - {"date", "llama_cpp_build", "llama_cpp_commit"}
    if unknown:
        raise ProfileError(f"{where} has unknown key(s): {', '.join(sorted(unknown))}")
    date = value.get("date")
    if not isinstance(date, str) or not _DATE_RE.match(date):
        raise ProfileError(f"{where}.date must be a YYYY-MM-DD string, got {date!r}")
    build = value.get("llama_cpp_build")
    if isinstance(build, bool) or not isinstance(build, int):
        raise ProfileError(f"{where}.llama_cpp_build must be an integer, got {build!r}")
    commit = value.get("llama_cpp_commit")
    if not isinstance(commit, str) or commit == "":
        raise ProfileError(f"{where}.llama_cpp_commit must be a non-empty string, got {commit!r}")
    return Verified(date, build, commit)


def _parse_section(name: str, slot: str, slot_names: Collection[str], body: object) -> SlotSection:
    where = f"{name}: [slots.{slot}]"
    if slot not in slot_names:
        raise ProfileError(
            f"{where} is not a configured slot (configured: {', '.join(sorted(slot_names))})")
    if not isinstance(body, dict):
        raise ProfileError(f"{where} must be a table, got {body!r}")

    unknown = set(body) - {"ctx_size", "n_gpu_layers", "flags", "sampling", "env", "verified"}
    if unknown:
        raise ProfileError(f"{where} has unknown key(s): {', '.join(sorted(unknown))}")

    # Required integers.
    if "ctx_size" not in body:
        raise ProfileError(f"{where} is missing required key 'ctx_size'")
    ctx_size = body["ctx_size"]
    if isinstance(ctx_size, bool) or not isinstance(ctx_size, int):
        raise ProfileError(f"{where}.ctx_size must be an integer, got {ctx_size!r}")
    if ctx_size <= 0:
        raise ProfileError(f"{where}.ctx_size must be above 0, got {ctx_size}")

    if "n_gpu_layers" not in body:
        raise ProfileError(f"{where} is missing required key 'n_gpu_layers'")
    n_gpu_layers = body["n_gpu_layers"]
    if isinstance(n_gpu_layers, bool) or not isinstance(n_gpu_layers, int):
        raise ProfileError(f"{where}.n_gpu_layers must be an integer, got {n_gpu_layers!r}")
    if n_gpu_layers < -1:
        raise ProfileError(f"{where}.n_gpu_layers must be at least -1, got {n_gpu_layers}")

    # Optional flags: a list of single tokens that cannot inject.
    flags: tuple[str, ...] = ()
    if "flags" in body:
        raw = body["flags"]
        if not isinstance(raw, list):
            raise ProfileError(f"{where}.flags must be a list of tokens, got {raw!r}")
        for token in raw:
            _parse_token(name, f"{where}.flags", token)
            base = token.split("=", 1)[0]
            if base in RESERVED_FLAGS:
                raise ProfileError(
                    f"{where}.flags token {token!r} sets reserved flag '{base}', "
                    f"which the slot or the renderer owns")
        flags = tuple(raw)

    # Sampling: exactly SAMPLING_KEYS, all present.
    if "sampling" not in body:
        raise ProfileError(f"{where} is missing required key 'sampling'")
    sampling = body["sampling"]
    if not isinstance(sampling, dict):
        raise ProfileError(f"{where}.sampling must be a table, got {sampling!r}")
    unknown = set(sampling) - set(SAMPLING_KEYS)
    if unknown:
        raise ProfileError(f"{where}.sampling has unknown key(s): {', '.join(sorted(unknown))}")
    missing = [key for key in SAMPLING_KEYS if key not in sampling]
    if missing:
        raise ProfileError(f"{where}.sampling is missing required key(s): {', '.join(missing)}")
    for key in SAMPLING_KEYS:
        value = sampling[key]
        if isinstance(value, bool):
            raise ProfileError(f"{where}.sampling.{key} must be a number, not a boolean")
        if not isinstance(value, (int, float)):
            raise ProfileError(f"{where}.sampling.{key} must be a number, got {value!r}")
        if key == "top_k":
            if not isinstance(value, int):
                raise ProfileError(f"{where}.sampling.top_k must be an integer, got {value!r}")
            if value < 0:
                raise ProfileError(f"{where}.sampling.top_k must be at least 0, got {value}")
    sampling_pairs = tuple((key, sampling[key]) for key in SAMPLING_KEYS)

    # Optional per-model env vars: KEY=VALUE lines in the generated file.
    env: tuple[tuple[str, str], ...] = ()
    if "env" in body:
        raw = body["env"]
        if not isinstance(raw, dict):
            raise ProfileError(f"{where}.env must be a table of KEY=VALUE strings, got {raw!r}")
        for key, value in raw.items():
            if not isinstance(key, str) or not _ENV_KEY_RE.match(key):
                raise ProfileError(f"{where}.env key {key!r} must match [A-Z_][A-Z0-9_]*")
            if key in RESERVED_ENV:
                raise ProfileError(f"{where}.env key '{key}' is reserved for the slot")
            if key.startswith("LLAMA_ARG_"):
                raise ProfileError(
                    f"{where}.env key '{key}' starts with LLAMA_ARG_, which llama-server "
                    f"reads as a flag MODEL_ARGS would not show")
            _parse_token(name, f"{where}.env {key}", value)
        env = tuple(sorted(raw.items()))

    return SlotSection(
        slot=slot,
        ctx_size=ctx_size,
        n_gpu_layers=n_gpu_layers,
        sampling=sampling_pairs,
        flags=flags,
        env=env,
        verified=_parse_verified(name, slot, body["verified"]) if "verified" in body else None,
    )


def parse_profile(name: str, data: dict, slot_names: Collection[str]) -> Profile:
    """Validate a parsed profile table. Pure: no filesystem, no env.

    Raises ProfileError (a ValueError) whose message names the profile and
    the offending key.
    """
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ProfileError(
            f"profile name {name!r} is invalid: it must match [A-Za-z0-9._-]+")
    if not isinstance(data, dict):
        raise ProfileError(f"{name}: profile must be a table, got {data!r}")

    unknown = set(data) - {"home", "slots"}
    if unknown:
        raise ProfileError(f"{name} has unknown top-level key(s): {', '.join(sorted(unknown))}")
    if "home" not in data:
        raise ProfileError(f"{name} is missing required key 'home'")
    home = data["home"]
    if not isinstance(home, str) or home == "":
        raise ProfileError(f"{name}.home must be a non-empty string, got {home!r}")
    if "slots" not in data:
        raise ProfileError(f"{name} is missing required key 'slots'")
    slots = data["slots"]
    if not isinstance(slots, dict) or not slots:
        raise ProfileError(f"{name}.slots must be a non-empty table, got {slots!r}")

    sections: dict[str, SlotSection] = {}
    for slot, body in slots.items():
        sections[slot] = _parse_section(name, slot, slot_names, body)

    if home not in sections:
        raise ProfileError(
            f"{name}.home must name one of the [slots.*] sections; home is '{home}', "
            f"sections are {', '.join(sections)}")

    return Profile(name=name, home=home, sections=sections)


# ---------------------------------------------------------------------------
# load_profile / load_profiles: file access, nothing else
# ---------------------------------------------------------------------------

def load_profile(path: str, slot_names: Collection[str], models_dir: str) -> Profile:
    """Load and validate one profile file; name is the file stem.

    Also requires <models_dir>/<name>.gguf to exist with a size above 0.
    """
    file = Path(path)
    name = file.stem
    try:
        with file.open("rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ProfileError(f"{name}: cannot read profile file {path}: {exc}") from exc
    profile = parse_profile(name, data, slot_names)
    gguf = Path(models_dir) / f"{name}.gguf"
    if not gguf.is_file() or gguf.stat().st_size <= 0:
        raise ProfileError(f"{name}: model file {gguf} is missing or empty; a profile is "
                           f"useless without its gguf")
    return profile


def load_profiles(profiles_dir: str, slot_names: Collection[str], models_dir: str) -> ProfileSet:
    """Load every *.toml in profiles_dir. Never raises: a bad file goes into
    ProfileSet.errors keyed by its stem, and a missing directory sets
    errors[profiles_dir]."""
    profiles: dict[str, Profile] = {}
    errors: dict[str, str] = {}
    pdir = Path(profiles_dir)
    if not pdir.is_dir():
        errors[str(profiles_dir)] = f"profiles directory {profiles_dir} does not exist"
        return ProfileSet(profiles, errors)
    for file in sorted(pdir.glob("*.toml")):
        try:
            profile = load_profile(str(file), slot_names, models_dir)
        except ProfileError as exc:
            errors[file.stem] = str(exc)
            continue
        except Exception as exc:  # never raise: one bad file must not take the manager down
            errors[file.stem] = f"{file.stem}: unexpected error: {exc!r}"
            continue
        key = match_key(profile.name)
        if key is None:
            errors[file.stem] = f"invalid profile name {profile.name!r}"
            continue
        if key in profiles:
            first = profiles.pop(key)
            errors[first.name] = (
                f"{first.name} collides with {file.stem!r} under case-insensitive matching")
            errors[file.stem] = (
                f"{file.stem} collides with {first.name!r} under case-insensitive matching")
            continue
        profiles[key] = profile
    return ProfileSet(profiles, errors)


# ---------------------------------------------------------------------------
# render_model_env (Q32: the whole file is rendered, nothing carries over)
# ---------------------------------------------------------------------------

def render_model_env(profile: Profile, section: SlotSection, models_dir: str) -> str:
    """Render the slot's generated model env file: a header comment, the whole
    MODEL_ARGS line, and one line per env entry. Ends with a newline."""
    tokens = [
        "--model", f"{models_dir}/{profile.name}.gguf",
        "--ctx-size", str(section.ctx_size),
        "--n-gpu-layers", str(section.n_gpu_layers),
    ]
    for key, value in section.sampling:
        tokens.append(f"--{key.replace('_', '-')}")
        tokens.append(repr(value) if isinstance(value, float) else str(value))
    tokens.extend(section.flags)
    lines = [
        f"# Generated by llama-manager from profile {profile.name} [slots.{section.slot}]. "
        f"Every swap rewrites this file; do not edit.",
        "MODEL_ARGS=" + " ".join(tokens),
    ]
    for key, value in section.env:
        lines.append(f"{key}={value}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI: python -m manager.profiles check|render
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m manager.profiles",
        description="Validate and render per-model profiles.")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="validate every profile in a directory")
    check.add_argument("profiles_dir")
    check.add_argument("--slots", required=True, help="comma-separated slot names")
    check.add_argument("--models-dir", required=True, help="directory holding the gguf files")

    render = sub.add_parser("render", help="render one slot's model env file")
    render.add_argument("profiles_dir")
    render.add_argument("model", help="profile name (case- and .gguf-insensitive)")
    render.add_argument("slot")
    render.add_argument("--slots", required=True, help="comma-separated slot names")
    render.add_argument("--models-dir", required=True, help="directory holding the gguf files")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    slot_names = [slot for slot in args.slots.split(",") if slot]
    ps = load_profiles(args.profiles_dir, slot_names, args.models_dir)

    if args.command == "check":
        for stem in sorted(ps.errors):
            print(f"{stem}: {ps.errors[stem]}", file=sys.stderr)
        if ps.errors:
            return 1
        print(f"{len(ps.profiles)} profiles OK")
        return 0

    profile = ps.get(args.model)
    if profile is None:
        print(f"no profile for model {args.model!r}; is it in {args.profiles_dir}?",
              file=sys.stderr)
        return 1
    section = profile.section(args.slot)
    if section is None:
        print(f"profile {profile.name!r} has no [slots.{args.slot}] section; "
              f"it may run on: {', '.join(profile.sections)}", file=sys.stderr)
        return 1
    sys.stdout.write(render_model_env(profile, section, args.models_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
