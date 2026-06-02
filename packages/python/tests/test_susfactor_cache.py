"""Tests for SusFactor model-cache helpers."""

from odin_prompt_toolkit.providers.model_cache import (
    ModelCache,
    susfactor_model_dir,
    susfactor_model_files_present,
)


def test_susfactor_model_dir(tmp_path):
    cache = ModelCache(cache_dir=str(tmp_path))
    assert susfactor_model_dir(cache, "susfactor-v1") == tmp_path / "susfactor-v1"


def test_files_present_false_when_missing(tmp_path):
    cache = ModelCache(cache_dir=str(tmp_path))
    assert susfactor_model_files_present(cache, "susfactor-v1") is False


def test_files_present_true_with_full_layout(tmp_path):
    cache = ModelCache(cache_dir=str(tmp_path))
    model_dir = tmp_path / "susfactor-v1"
    encoder = model_dir / "encoder"
    encoder.mkdir(parents=True)
    (encoder / "config.json").write_text("{}")
    (encoder / "model.safetensors").write_bytes(b"\x00")
    (encoder / "tokenizer.json").write_text("{}")
    (model_dir / "head.pt").write_bytes(b"\x00")
    assert susfactor_model_files_present(cache, "susfactor-v1") is True


def test_files_present_false_when_head_missing(tmp_path):
    cache = ModelCache(cache_dir=str(tmp_path))
    model_dir = tmp_path / "susfactor-v1"
    encoder = model_dir / "encoder"
    encoder.mkdir(parents=True)
    (encoder / "config.json").write_text("{}")
    (encoder / "model.safetensors").write_bytes(b"\x00")
    (encoder / "tokenizer.json").write_text("{}")
    # head.pt intentionally missing
    assert susfactor_model_files_present(cache, "susfactor-v1") is False
