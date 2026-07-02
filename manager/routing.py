"""
Pure routing decisions for the dual-slot model manager.

Given a requested model name and current per-slot state, return which slot
should handle the request, or None if the model is loaded on neither slot.
Pure function: no I/O, no side effects.
"""
from typing import Optional

from manager.names import same_model
from manager.slots import SlotState


def resolve_slot(model: str, slots: dict[str, SlotState]) -> Optional[str]:
    """Return the slot that should handle the request, or None if unloaded.

    Rules:
      1. If the model is loaded on 'main', return 'main'.
      2. Else if loaded on 'batch', return 'batch'.
      3. Else return None. The caller returns 409; implicit swaps happen only
         via POST /swap. Comparison is case/suffix/path-insensitive and None-safe.
    """
    main = slots.get("main")
    if main is not None and same_model(model, main.loaded_model):
        return "main"
    batch = slots.get("batch")
    if batch is not None and same_model(model, batch.loaded_model):
        return "batch"
    return None
