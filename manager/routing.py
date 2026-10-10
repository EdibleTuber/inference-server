"""
Pure routing decisions for the model manager.

Given a requested model name and current per-slot state, return which slot
should handle the request, or None if the model is loaded on none of them.
Pure function: no I/O, no side effects.
"""
from typing import Iterable, Optional

from manager.names import same_model
from manager.slot_config import SlotConfig
from manager.slots import SlotState


def resolve_slot(model: str, slots: dict[str, SlotState]) -> Optional[str]:
    """Return the slot that should handle the request, or None if unloaded.

    Walks the slots in iteration order, which the caller sets from
    configuration: the first configured slot holding the model wins. Returning
    None means the model is loaded nowhere and the caller answers 409 --
    implicit swaps happen only via POST /swap. Comparison is
    case/suffix/path-insensitive and None-safe.
    """
    for name, slot in slots.items():
        if slot is not None and same_model(model, slot.loaded_model):
            return name
    return None


def home_slot(model: str, slot_configs: Iterable[SlotConfig]) -> Optional[str]:
    """Return the slot whose SLOT_<NAME>_MODELS lists `model`, or None.

    Used only to place an untargeted POST /swap. build_slots guarantees a model
    has at most one home, so the first match is the only match. Comparison is
    names.same_model: case/suffix/path-insensitive.
    """
    for sc in slot_configs:
        if any(same_model(model, m) for m in sc.models):
            return sc.name
    return None
