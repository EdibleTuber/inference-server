"""
Pure routing decisions for the model manager.

Given a requested model name and current per-slot state, return which slot
should handle the request, or None if the model is loaded on none of them.
Pure function: no I/O, no side effects.
"""
from typing import Optional

from manager.names import same_model
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
