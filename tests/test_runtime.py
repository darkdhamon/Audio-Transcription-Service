"""Tests for runtime hardware detection and backend selection."""

from pathlib import Path
import sys

import pytest


# Ensure the ``src`` package is importable when running tests directly.
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from domain.runtime import HardwareProfile, resolve_runtime_selection
from domain.transcription import TranscriptionOptions


def test_resolve_runtime_prefers_cuda_for_faster_whisper(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto mode should prefer CUDA when faster-whisper can use it."""

    profile = HardwareProfile(
        operating_system="Windows",
        gpu_names=["NVIDIA GeForce RTX 4070"],
        nvidia_gpu_names=["NVIDIA GeForce RTX 4070"],
        ctranslate2_cuda_device_count=1,
        torch_cuda_available=True,
    )
    monkeypatch.setattr("domain.runtime.detect_hardware_profile", lambda: profile)

    runtime = resolve_runtime_selection(TranscriptionOptions())

    assert runtime.device == "cuda"
    assert runtime.compute_type == "float16"
    assert runtime.model == "turbo"


def test_resolve_runtime_falls_back_to_cpu_on_amd_systems(monkeypatch: pytest.MonkeyPatch) -> None:
    """AMD/DirectML systems should remain on the CPU path for current backends."""

    profile = HardwareProfile(
        operating_system="Windows",
        gpu_names=["AMD Radeon RX 7900 XTX"],
        amd_gpu_names=["AMD Radeon RX 7900 XTX"],
        directml_available=True,
        ctranslate2_cuda_device_count=0,
        torch_cuda_available=False,
    )
    monkeypatch.setattr("domain.runtime.detect_hardware_profile", lambda: profile)

    runtime = resolve_runtime_selection(TranscriptionOptions())

    assert runtime.device == "cpu"
    assert runtime.compute_type == "int8"
    assert runtime.model == "distil-large-v3"
    assert any("DirectML" in note for note in runtime.notes)


def test_resolve_runtime_rejects_unavailable_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit CUDA requests should fail fast when unsupported."""

    profile = HardwareProfile(
        operating_system="Windows",
        gpu_names=["AMD Radeon RX 7900 XTX"],
        amd_gpu_names=["AMD Radeon RX 7900 XTX"],
        directml_available=True,
        ctranslate2_cuda_device_count=0,
        torch_cuda_available=False,
    )
    monkeypatch.setattr("domain.runtime.detect_hardware_profile", lambda: profile)

    with pytest.raises(RuntimeError):
        resolve_runtime_selection(
            TranscriptionOptions(engine="whisperx", device="cuda")
        )


def test_resolve_runtime_uses_small_for_non_english_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """The English-focused distilled model should not become the multilingual default."""

    profile = HardwareProfile(
        operating_system="Windows",
        gpu_names=["AMD Radeon RX 7900 XTX"],
        amd_gpu_names=["AMD Radeon RX 7900 XTX"],
        directml_available=True,
        ctranslate2_cuda_device_count=0,
        torch_cuda_available=False,
    )
    monkeypatch.setattr("domain.runtime.detect_hardware_profile", lambda: profile)

    runtime = resolve_runtime_selection(TranscriptionOptions(lang="fr"))

    assert runtime.device == "cpu"
    assert runtime.model == "small"


def test_resolve_runtime_prefers_local_model_clone_when_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A locally cloned FasterWhisper model should be used instead of Hub lookup."""

    profile = HardwareProfile(
        operating_system="Windows",
        gpu_names=["AMD Radeon RX 7900 XTX"],
        amd_gpu_names=["AMD Radeon RX 7900 XTX"],
        directml_available=False,
        ctranslate2_cuda_device_count=0,
        torch_cuda_available=False,
    )
    model_dir = tmp_path / "Systran--faster-distil-whisper-large-v3"
    model_dir.mkdir()
    for file_name in ("config.json", "tokenizer.json", "model.bin"):
        (model_dir / file_name).write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr("domain.runtime.detect_hardware_profile", lambda: profile)
    monkeypatch.setattr("domain.runtime.LOCAL_MODEL_ROOT", tmp_path)

    runtime = resolve_runtime_selection(TranscriptionOptions())

    assert runtime.model == "distil-large-v3"
    assert runtime.model_load_target == str(model_dir)
    assert any("Using local model cache" in note for note in runtime.notes)
