# tests/systemd_expand.py
"""Expand a unit's ExecStart into argv the way systemd does, for tests.

Models only the subset this repo's llama-server units use, and raises on
anything outside it rather than guessing:

  - [Service] lines, with trailing-backslash continuation and #/; comments;
  - Environment=KEY=VALUE (one or more space-separated, unquoted);
  - EnvironmentFile=PATH (optional leading '-'), applied in file order, so
    a later assignment wins over an earlier one (systemd.exec(5));
  - env files of plain KEY=VALUE lines; a quote or backslash in a value
    raises, since systemd would unquote or unescape it;
  - ExecStart words: a word that is exactly $NAME is replaced by the value
    split on whitespace (empty or unset -> no words at all); ${NAME}
    inside a word is substituted in place and stays one word (unset ->
    the empty string). Both rules are systemd.service(5)'s.
"""
from __future__ import annotations

import re
from collections.abc import Callable

_NAME = r"[A-Za-z_][A-Za-z0-9_]*"


def service_directives(unit_text: str) -> list[tuple[str, str]]:
    """(key, value) pairs from the [Service] section, in file order."""
    out: list[tuple[str, str]] = []
    section = None
    pending = ""
    for raw in unit_text.splitlines():
        line = raw.strip()
        if not pending and (not line or line[0] in "#;"):
            continue
        if pending and line and line[0] in "#;":
            continue                      # comment inside a continuation
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        line = pending + line
        pending = ""
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        if section != "Service":
            continue
        key, sep, value = line.partition("=")
        if not sep:
            raise ValueError(f"not a directive: {line!r}")
        out.append((key.strip(), value.strip()))
    if pending:
        raise ValueError("unit ends inside a line continuation")
    return out


def parse_env_file(text: str) -> dict[str, str]:
    """A systemd EnvironmentFile of plain KEY=VALUE lines."""
    env: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line[0] in "#;":
            continue
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not re.fullmatch(_NAME, key):
            raise ValueError(f"not an env assignment: {raw!r}")
        if any(ch in value for ch in "\"'\\"):
            raise NotImplementedError(f"quoting/escapes are not modelled: {raw!r}")
        env[key] = value
    return env


def _parse_environment(value: str) -> dict[str, str]:
    if any(ch in value for ch in "\"'\\"):
        raise NotImplementedError(f"quoted Environment= is not modelled: {value!r}")
    env: dict[str, str] = {}
    for assignment in value.split():
        key, sep, val = assignment.partition("=")
        if not sep or not re.fullmatch(_NAME, key):
            raise ValueError(f"not an Environment= assignment: {assignment!r}")
        env[key] = val
    return env


def expand_exec_start(unit_text: str, read_env_file: Callable[[str], str]) -> list[str]:
    """The argv systemd would exec for this unit's ExecStart.

    read_env_file(path) returns the text of each EnvironmentFile= path; it
    should raise for any path the test does not expect the unit to load.
    """
    env: dict[str, str] = {}
    command = None
    for key, value in service_directives(unit_text):
        if key == "Environment":
            env.update(_parse_environment(value))
        elif key == "EnvironmentFile":
            env.update(parse_env_file(read_env_file(value.lstrip("-"))))
        elif key == "ExecStart":
            if command is not None:
                raise NotImplementedError("more than one ExecStart= is not modelled")
            command = value
    if command is None:
        raise ValueError("unit has no ExecStart=")
    if command[:1] in "-@+!:" or any(ch in command for ch in "\"'"):
        raise NotImplementedError(f"ExecStart prefixes and quoting are not modelled: {command!r}")

    argv: list[str] = []
    for word in command.split():
        whole = re.fullmatch(rf"\$({_NAME})", word)
        if whole:
            argv.extend(env.get(whole.group(1), "").split())
            continue
        if re.search(r"\$(?!\{)", word):
            raise NotImplementedError(f"$NAME inside a word is not modelled: {word!r}")
        argv.append(re.sub(rf"\$\{{({_NAME})\}}", lambda m: env.get(m.group(1), ""), word))
    return argv
