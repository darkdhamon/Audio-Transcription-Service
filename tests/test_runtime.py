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
    assert runtime.model == "small"
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
