"""
Model-name normalization for the dual-slot manager.

One vocabulary, used everywhere a model name is stored or compared:
  - display_name: the clean form we STORE and REPORT (basename, no .gguf, original case)
  - match_key:    the case-folded key we COMPARE on
  - same_model:   the single comparison primitive (None-safe)

Model filenames are assumed ASCII; casefold() is locale-independent.

Correctness assumes llama-server reports the model by its file path/stem (no
--alias), so probe() and a swap agree on the same physical file. If --alias is
ever added it must equal the GGUF stem, or probe() must resolve it back.
"""
from __future__ import annotations


def display_name(raw: str | None) -> str | None:
    """Storage/report form: basename, drop one trailing .gguf, keep original case.

    Returns None for falsy/empty input so the empty->None invariant lives in ONE
    place (e.g. a trailing slash like 'foo.gguf/' yields an empty basename).
    """
    if not raw:
        return None
    name = raw.rsplit("/", 1)[-1]            # strip any directory component
    if name.lower().endswith(".gguf"):       # case-insensitive: handles .GGUF
        name = name[: -len(".gguf")]
    return name or None


def match_key(name: str | None) -> str | None:
    """Tolerant comparison key: display form, case-folded. None-total."""
    d = display_name(name)
    return d.casefold() if d is not None else None


def same_model(requested: str | None, loaded: str | None) -> bool:
    """The single routing/guard comparison. None loaded_model never matches."""
    lk = match_key(loaded)
    return lk is not None and lk == match_key(requested)
