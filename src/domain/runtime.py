from __future__ import annotations

"""Runtime hardware detection and backend selection helpers.

The application supports two practical execution paths today:

- CPU inference, which works on all supported Windows machines and is the
  preferred fallback for AMD systems.
- NVIDIA CUDA inference, which is available for the ``faster-whisper`` and
  ``whisperx`` backends when the required GPU libraries are installed.

The helpers in this module keep the hardware probing logic separate from the
CLI and transcription backends so that front-ends can explain the selected
runtime in a consistent way.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional, TYPE_CHECKING
import importlib.util
import json
import platform
import subprocess

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    from domain.transcription import TranscriptionOptions


ResolvedDevice = Literal["cpu", "cuda"]


@dataclass
class HardwareProfile:
    """Snapshot of the hardware/runtime features relevant to transcription."""

    operating_system: str
    gpu_names: List[str] = field(default_factory=list)
    nvidia_gpu_names: List[str] = field(default_factory=list)
    amd_gpu_names: List[str] = field(default_factory=list)
    directml_available: bool = False
    ctranslate2_cuda_device_count: int = 0
    torch_cuda_available: bool = False

    @property
    def has_nvidia_cuda_for_faster_whisper(self) -> bool:
        """Return ``True`` when ``faster-whisper`` can use CUDA."""

        return self.ctranslate2_cuda_device_count > 0

    @property
    def has_nvidia_cuda_for_whisperx(self) -> bool:
        """Return ``True`` when ``whisperx`` can use CUDA through PyTorch."""

        return self.torch_cuda_available


@dataclass
class RuntimeSelection:
    """Resolved execution settings derived from user options and hardware."""

    engine: Literal["faster-whisper", "whisperx"]
    model: str
    device: ResolvedDevice
    compute_type: str
    notes: List[str] = field(default_factory=list)
    hardware: Optional[HardwareProfile] = None

    def describe(self) -> str:
        """Return a concise human-readable summary."""

        return (
            f"engine={self.engine}, model={self.model}, "
            f"device={self.device}, compute_type={self.compute_type}"
        )


def _list_windows_gpu_names() -> List[str]:
    """Return installed Windows display adapter names using PowerShell."""

    if platform.system() != "Windows":
        return []

    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                (
                    "Get-CimInstance Win32_VideoController | "
                    "Select-Object -ExpandProperty Name | ConvertTo-Json -Compress"
                ),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    output = (result.stdout or "").strip()
    if not output:
        return []

    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return []

    if isinstance(data, str):
        return [data]
    if isinstance(data, list):
        return [str(item) for item in data]
    return []


def _detect_ctranslate2_cuda_device_count() -> int:
    """Return the number of CUDA devices visible to CTranslate2."""

    try:
        import ctranslate2  # type: ignore
    except ImportError:
        return 0

    try:
        return int(ctranslate2.get_cuda_device_count())
    except Exception:  # pragma: no cover - backend probing is environment-specific
        return 0


def _detect_torch_cuda_available() -> bool:
    """Return whether PyTorch reports an available CUDA device."""

    try:
        import torch  # type: ignore
    except ImportError:
        return False

    try:
        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - backend probing is environment-specific
        return False


def detect_hardware_profile() -> HardwareProfile:
    """Inspect the current machine and return relevant runtime capabilities."""

    gpu_names = _list_windows_gpu_names()
    nvidia_gpu_names = [name for name in gpu_names if "nvidia" in name.lower()]
    amd_gpu_names = [
        name for name in gpu_names if any(token in name.lower() for token in ("amd", "radeon"))
    ]

    return HardwareProfile(
        operating_system=platform.system(),
        gpu_names=gpu_names,
        nvidia_gpu_names=nvidia_gpu_names,
        amd_gpu_names=amd_gpu_names,
        directml_available=importlib.util.find_spec("torch_directml") is not None,
        ctranslate2_cuda_device_count=_detect_ctranslate2_cuda_device_count(),
        torch_cuda_available=_detect_torch_cuda_available(),
    )


def _engine_supports_cuda(profile: HardwareProfile, engine: str) -> bool:
    """Return whether ``engine`` can use CUDA on the current machine."""

    if engine == "faster-whisper":
        return profile.has_nvidia_cuda_for_faster_whisper
    if engine == "whisperx":
        return profile.has_nvidia_cuda_for_whisperx
    return False


def _resolve_model_name(options: "TranscriptionOptions", device: ResolvedDevice) -> str:
    """Choose a practical default model when the user requested ``auto``."""

    if options.model != "auto":
        return options.model

    if options.engine == "faster-whisper" and device == "cuda":
        # ``turbo`` is the modern high-throughput Whisper choice for GPU-backed
        # transcription. CPU fallback remains on ``small`` for broad compatibility.
        return "turbo"

    return "small"


def _resolve_compute_type(options: "TranscriptionOptions", device: ResolvedDevice) -> str:
    """Choose a backend-friendly compute type for ``device``."""

    if options.compute_type != "auto":
        return options.compute_type

    if device == "cuda":
        return "float16"
    return "int8"


def resolve_runtime_selection(options: "TranscriptionOptions") -> RuntimeSelection:
    """Resolve execution settings from user options and current hardware."""

    profile = detect_hardware_profile()
    notes: List[str] = []
    cuda_supported = _engine_supports_cuda(profile, options.engine)

    if options.device == "cuda":
        if not cuda_supported:
            raise RuntimeError(
                f"CUDA was requested, but the '{options.engine}' backend does not "
                "have a usable NVIDIA CUDA device on this machine."
            )
        device: ResolvedDevice = "cuda"
    elif options.device == "cpu":
        device = "cpu"
    else:
        device = "cuda" if cuda_supported else "cpu"

    if device == "cpu":
        if profile.amd_gpu_names:
            if profile.directml_available:
                notes.append(
                    "AMD/DirectML-capable hardware detected, but the current "
                    f"'{options.engine}' backend does not support DirectML. Using CPU."
                )
            else:
                notes.append(
                    "AMD GPU detected. Current transcription backends support CPU "
                    "or NVIDIA CUDA only, so the app is using CPU."
                )
        elif profile.gpu_names and not cuda_supported:
            notes.append(
                f"GPU(s) detected ({', '.join(profile.gpu_names)}), but the current "
                f"'{options.engine}' backend cannot use them. Using CPU."
            )
    else:
        notes.append("NVIDIA CUDA detected. Using GPU acceleration.")

    return RuntimeSelection(
        engine=options.engine,
        model=_resolve_model_name(options, device),
        device=device,
        compute_type=_resolve_compute_type(options, device),
        notes=notes,
        hardware=profile,
    )
