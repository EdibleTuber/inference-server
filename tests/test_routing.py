"""Tests for manager/routing.py: pure routing function over slot state."""
from manager.routing import resolve_slot
from manager.slots import SlotState
from manager.queue import RequestQueue


def _slot(name: str, loaded_model=None) -> SlotState:
    return SlotState(
        name=name,
        host="127.0.0.1",
        port=18081 if name == "main" else 18083,
        env_file="/tmp/env",
        systemd_unit=f"llama-server-{name}.service" if name == "egpu" else "llama-server.service",
        queue=RequestQueue(max_size=10),
        loaded_model=loaded_model,
    )


def test_resolve_model_on_main():
    slots = {"main": _slot("main", "model-A"), "egpu": _slot("egpu", "model-B")}
    assert resolve_slot("model-A", slots) == "main"


def test_resolve_model_on_egpu():
    slots = {"main": _slot("main", "model-A"), "egpu": _slot("egpu", "model-B")}
    assert resolve_slot("model-B", slots) == "egpu"


def test_resolve_model_on_neither_returns_none():
    slots = {"main": _slot("main", "model-A"), "egpu": _slot("egpu", "model-B")}
    assert resolve_slot("unknown-C", slots) is None


def test_resolve_model_on_both_prefers_main():
    """If the same model is somehow loaded on both, main wins (first match)."""
    slots = {"main": _slot("main", "shared"), "egpu": _slot("egpu", "shared")}
    assert resolve_slot("shared", slots) == "main"


def test_resolve_empty_model_returns_none():
    slots = {"main": _slot("main", None), "egpu": _slot("egpu", None)}
    assert resolve_slot("", slots) is None


def test_resolve_is_case_and_suffix_and_path_insensitive():
    slots = {"main": _slot("main", "gemma-4-E4B-it-Q4_K_M"), "egpu": _slot("egpu", None)}
    assert resolve_slot("gemma-4-e4b-it-q4_k_m", slots) == "main"
    assert resolve_slot("GEMMA-4-E4B-IT-Q4_K_M.gguf", slots) == "main"
    assert resolve_slot("/mnt/models/gemma-4-E4B-it-Q4_K_M.gguf", slots) == "main"


def test_resolve_none_loaded_model_is_safe():
    """A slot with loaded_model=None must not crash the comparison."""
    slots = {"main": _slot("main", None), "egpu": _slot("egpu", None)}
    assert resolve_slot("anything", slots) is None


def test_resolve_main_unloaded_but_egpu_loaded():
    """Main has nothing loaded, egpu has the model -> route to egpu."""
    slots = {"main": _slot("main", None), "egpu": _slot("egpu", "model-B")}
    assert resolve_slot("model-B", slots) == "egpu"


def test_resolves_a_third_slot():
    slots = {
        "main": _slot("main", "coding-model"),
        "egpu": _slot("egpu", "small-model"),
        "aux": _slot("aux", "gemma-4-26b-a4b-it-q4_k_m"),
    }
    assert resolve_slot("gemma-4-26b-a4b-it-q4_k_m", slots) == "aux"


def test_first_configured_slot_wins_when_duplicated():
    """Priority is iteration order, so the earliest configured slot wins."""
    slots = {"aux": _slot("aux", "shared"), "main": _slot("main", "shared")}
    assert resolve_slot("shared", slots) == "aux"


def test_first_configured_slot_wins_when_duplicated_reverse_order():
    """Same guarantee with the dict built in the opposite order, so the test
    cannot pass by accident of a hardcoded 'main'/'egpu'-then-others walk --
    it only passes if resolve_slot genuinely follows dict iteration order."""
    slots = {"main": _slot("main", "shared"), "aux": _slot("aux", "shared")}
    assert resolve_slot("shared", slots) == "main"


def test_unknown_model_across_three_slots_returns_none():
    slots = {
        "main": _slot("main", "a"), "egpu": _slot("egpu", "b"), "aux": _slot("aux", "c"),
    }
    assert resolve_slot("d", slots) is None


def test_third_slot_comparison_is_normalised():
    slots = {"main": _slot("main", None), "aux": _slot("aux", "Gemma-4-26B")}
    assert resolve_slot("/models/gemma-4-26b.gguf", slots) == "aux"


from manager.routing import home_slot
from manager.slot_config import SlotConfig


def _sc(name, *models):
    return SlotConfig(name=name, host="127.0.0.1", port=1, env_file="/tmp/env",
                      systemd_unit="u.service", queue_limit=1, models=tuple(models))


def test_home_slot_finds_the_configured_slot():
    scs = [_sc("main", "model-a"), _sc("egpu", "model-b")]
    assert home_slot("model-b", scs) == "egpu"


def test_home_slot_is_case_and_suffix_insensitive():
    scs = [_sc("main"), _sc("egpu", "Model-B.gguf")]
    assert home_slot("model-b", scs) == "egpu"


def test_home_slot_none_for_unmapped_model():
    scs = [_sc("main", "model-a"), _sc("egpu")]
    assert home_slot("model-z", scs) is None
