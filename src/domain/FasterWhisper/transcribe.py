from __future__ import annotations

"""Utilities for performing transcription with the ``faster-whisper`` backend."""

from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imported for type hints only
    from domain.transcription import ProgressReporter, TranscriptionOptions


def transcribe(
    audio_path: str,
    options: "TranscriptionOptions",
    vad_params: Optional[dict],
    speaker_label: str,
    progress_callback: Optional["ProgressReporter"] = None,
) -> List[Dict]:
    """Transcribe ``audio_path`` using ``faster-whisper`` and return segments.

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
    speaker_label:
        Label to associate with the speaker for all emitted segments.
    progress_callback:
        Optional callback that receives file-relative decode progress in
        seconds as segments become available from ``faster-whisper``.

    Returns
    -------
    list of dict
        Each dictionary contains ``start``, ``end``, ``text`` and ``speaker``
        keys describing a portion of speech from the audio.
    """

    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as exc:  # pragma: no cover - defensive programming
        raise RuntimeError(
            "The 'faster-whisper' package is required. Install it with 'pip install faster-whisper'."
        ) from exc

    resolved_model = options.resolved_model_load_target or options.resolved_model or options.model
    resolved_device = options.resolved_device or "cpu"
    resolved_compute_type = options.resolved_compute_type or "int8"

    model = WhisperModel(
        resolved_model,
        device=resolved_device,
        compute_type=resolved_compute_type,
        cpu_threads=options.cpu_threads or 0,
    )

    transcribe_kwargs: Dict = dict(
        language=options.lang
        if resolved_model not in ("large", "large-v2", "large-v3")
        else None,
        # ``options.vad`` enables the backend's default VAD even when the
        # caller did not supply custom VAD thresholds.
        vad_filter=options.vad,
        beam_size=options.beam,
        temperature=options.temperature,
    )
    if vad_params:
        transcribe_kwargs["vad_parameters"] = vad_params

    segments, _info = model.transcribe(audio_path, **transcribe_kwargs)

    formatted: List[Dict] = []
    for seg in segments:
        start_seconds = float(getattr(seg, "start", 0.0))
        end_seconds = float(getattr(seg, "end", 0.0))
        if progress_callback is not None:
            progress_callback(end_seconds)
        text = getattr(seg, "text", "").strip()
        formatted.append(
            {"start": start_seconds, "end": end_seconds, "text": text, "speaker": speaker_label}
        )

    return formatted
