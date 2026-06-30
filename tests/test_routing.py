"""Tests for manager/routing.py: pure routing function over slot state."""
from manager.routing import resolve_slot
from manager.slots import SlotState
from manager.queue import RequestQueue


def _slot(name: str, loaded_model=None) -> SlotState:
    return SlotState(
        name=name,
        host="127.0.0.1",
        port=8081 if name == "main" else 8083,
        env_file="/tmp/env",
        systemd_unit=f"llama-server-{name}.service" if name == "batch" else "llama-server.service",
        queue=RequestQueue(max_size=10),
        loaded_model=loaded_model,
    )


def test_resolve_model_on_main():
    slots = {"main": _slot("main", "model-A"), "batch": _slot("batch", "model-B")}
    assert resolve_slot("model-A", slots) == "main"


def test_resolve_model_on_batch():
    slots = {"main": _slot("main", "model-A"), "batch": _slot("batch", "model-B")}
    assert resolve_slot("model-B", slots) == "batch"


def test_resolve_model_on_neither_returns_none():
    slots = {"main": _slot("main", "model-A"), "batch": _slot("batch", "model-B")}
    assert resolve_slot("unknown-C", slots) is None


def test_resolve_model_on_both_prefers_main():
    """If the same model is somehow loaded on both, main wins (first match)."""
    slots = {"main": _slot("main", "shared"), "batch": _slot("batch", "shared")}
    assert resolve_slot("shared", slots) == "main"


def test_resolve_empty_model_returns_none():
    slots = {"main": _slot("main", None), "batch": _slot("batch", None)}
    assert resolve_slot("", slots) is None


def test_resolve_is_case_and_suffix_and_path_insensitive():
    slots = {"main": _slot("main", "gemma-4-E4B-it-Q4_K_M"), "batch": _slot("batch", None)}
    assert resolve_slot("gemma-4-e4b-it-q4_k_m", slots) == "main"
    assert resolve_slot("GEMMA-4-E4B-IT-Q4_K_M.gguf", slots) == "main"
    assert resolve_slot("/mnt/models/gemma-4-E4B-it-Q4_K_M.gguf", slots) == "main"


def test_resolve_none_loaded_model_is_safe():
    """A slot with loaded_model=None must not crash the comparison."""
    slots = {"main": _slot("main", None), "batch": _slot("batch", None)}
    assert resolve_slot("anything", slots) is None


def test_resolve_main_unloaded_but_batch_loaded():
    """Main has nothing loaded, batch has the model -> route to batch."""
    slots = {"main": _slot("main", None), "batch": _slot("batch", "model-B")}
    assert resolve_slot("model-B", slots) == "batch"
