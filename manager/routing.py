"""
Pure routing decisions for the dual-slot model manager.

Given a requested model name and the current per-slot state, returns which
slot should handle the request. Pure function: no I/O, no side effects,
safe to call from anywhere.
"""
from manager.slots import SlotState


def resolve_slot(model: str, slots: dict[str, SlotState]) -> str:
    """Return the slot name that should handle the request.

    Rules:
      1. If the model is loaded on 'main', return 'main'.
      2. Else if the model is loaded on 'batch', return 'batch'.
      3. Else return 'main' (triggers an implicit main swap in the caller,
         preserving the existing single-slot behavior).
    """
    main = slots.get("main")
    if main is not None and main.loaded_model == model:
        return "main"
    batch = slots.get("batch")
    if batch is not None and batch.loaded_model == model:
        return "batch"
    return "main"
