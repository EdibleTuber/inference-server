"""Tests for manager/names.py: pure model-name normalization helpers."""
from manager.names import display_name, match_key, same_model


def test_display_name_plain_stem():
    assert display_name("gemma-4-E4B-it-Q4_K_M") == "gemma-4-E4B-it-Q4_K_M"


def test_display_name_strips_gguf():
    assert display_name("gemma-4-E4B-it-Q4_K_M.gguf") == "gemma-4-E4B-it-Q4_K_M"


def test_display_name_strips_gguf_case_insensitive():
    assert display_name("Model.GGUF") == "Model"


def test_display_name_strips_directory():
    assert display_name("/mnt/secondary/llama-models/foo-q4.gguf") == "foo-q4"


def test_display_name_strips_only_one_extension():
    # Path.stem semantics: only the final .gguf is removed.
    assert display_name("x.gguf.gguf") == "x.gguf"


def test_display_name_empty_and_none_are_none():
    assert display_name("") is None
    assert display_name(None) is None
    assert display_name("foo.gguf/") is None  # trailing slash -> empty basename


def test_match_key_folds_case_and_suffix_and_path():
    assert match_key("/p/Gemma-4-E4B.GGUF") == "gemma-4-e4b"
    assert match_key(None) is None
    assert match_key("") is None


def test_same_model_tolerant():
    assert same_model("gemma-4-e4b-it-q4_k_m", "gemma-4-E4B-it-Q4_K_M") is True
    assert same_model("/p/gemma-4-E4B-it-Q4_K_M.gguf", "gemma-4-e4b-it-q4_k_m") is True


def test_same_model_none_never_matches():
    assert same_model("anything", None) is False
    assert same_model(None, None) is False
    assert same_model("", "real-model") is False
