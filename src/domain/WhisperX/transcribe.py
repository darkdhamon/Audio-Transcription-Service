from __future__ import annotations

"""Utilities for performing transcription with WhisperX.

This module provides a small wrapper around the ``whisperx`` package so the
rest of the application does not need to interact with the external
dependency directly.  The primary entry point is :func:`transcribe` which
handles model loading, optional voice activity detection (VAD) and alignment
and returns a normalized list of transcription segments.
"""

from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - used only for type hints
    from domain.transcription import TranscriptionOptions


def transcribe(
    audio_path: str,
    options: "TranscriptionOptions",
    vad_params: Optional[dict],
    offset: float,
    speaker_label: str,
) -> List[Dict]:
    """Transcribe ``audio_path`` using WhisperX and return speech segments.

    Parameters
    ----------
    audio_path:
        Path to the audio file that will be processed.
    options:
        Instance of :class:`~domain.transcription.TranscriptionOptions`
        controlling language, model name and other settings.
    vad_params:
        Optional voice activity detection parameters.  When provided VAD is
        applied during transcription.
    offset:
        Time offset in seconds applied to every returned segment.
    speaker_label:
        Label to associate with the speaker for all emitted segments.

    Returns
    -------
    list of dict
        Each dictionary contains ``start``, ``end``, ``text`` and ``speaker``
        keys describing a portion of speech from the audio.
    """

    try:
        import whisperx  # type: ignore
    except ImportError as exc:  # pragma: no cover - defensive programming
        raise RuntimeError(
            "The 'whisperx' package is required. Install it with 'pip install whisperx'."
        ) from exc

    resolved_model = options.resolved_model or options.model
    resolved_device = options.resolved_device or "cpu"
    resolved_compute_type = options.resolved_compute_type or "int8"

    # Load the WhisperX model using the resolved runtime selection so the same
    # code path can use either CPU inference or NVIDIA CUDA acceleration.
    model = whisperx.load_model(
        resolved_model,
        device=resolved_device,
        compute_type=resolved_compute_type,
        cpu_threads=options.cpu_threads or 0,
    )

    # Prepare arguments for the transcription call.  Some Whisper models do
    # not accept an explicit language parameter so we only include it for
    # smaller multilingual models.
    transcribe_kwargs: Dict = {
        "language": options.lang
        if resolved_model not in ("large", "large-v2", "large-v3")
        else None
    }
    if vad_params:
        # WhisperX expects VAD options under the ``vad_options`` keyword.
        transcribe_kwargs["vad_options"] = vad_params

    result = model.transcribe(audio_path, **transcribe_kwargs)
    segments = result.get("segments", [])

    # Attempt to run alignment to produce more accurate timestamps.  If the
    # alignment model is unavailable we gracefully fall back to the original
    # segment timings.
    try:
        align_model, metadata = whisperx.load_align_model(
            language=options.lang, device=resolved_device
        )
        aligned = whisperx.align(
            segments, align_model, metadata, audio_path, return_char_alignments=False
        )
        segments = aligned.get("segments", segments)
    except Exception:  # pragma: no cover - alignment is best-effort
        pass

    formatted: List[Dict] = []
    for seg in segments:
        start = float(seg.get("start", 0.0)) + offset
        end = float(seg.get("end", 0.0)) + offset
        text = seg.get("text", "").strip()
        formatted.append(
            {"start": start, "end": end, "text": text, "speaker": speaker_label}
        )

    return formatted
