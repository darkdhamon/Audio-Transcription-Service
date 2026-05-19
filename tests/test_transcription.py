"""Test suite for core transcription utilities and services.

This module exercises helper functions and service methods used during
transcription.  The tests validate segment de-duplication, repeat
collapsing, VAD setup, and merging output parts.
"""

from pathlib import Path
import json
import queue
import sys
import types

import pytest
from typing import Dict, List


# Ensure the ``src`` package is importable when running tests directly.
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from domain.transcription import (
    TranscriptionOptions,
    TranscriptionService,
    build_vad_parameters,
    collapse_nearby_repeats,
    dedupe_consecutive,
    worker_transcribe,
)


def dummy_worker(
    file_path: str,
    options: TranscriptionOptions,
    vad_params: Dict,
    cpu_threads: int,
    prog_q,
    speaker_label: str,
    transcribe_fn,
) -> None:
    """Lightweight stand-in for ``worker_transcribe`` used in tests.

    The worker immediately reports completion and includes a copy of the VAD
    parameters so tests can verify that every worker receives the same cached
    configuration. It also records which backend was requested by reporting
    the module name of ``transcribe_fn``.
    """

    prog_q.put(
        {
            "type": "done",
            "file": file_path,
            "part": file_path + ".json.part",
            "vad_params": dict(vad_params or {}),
            "backend": getattr(transcribe_fn, "__module__", ""),
        }
    )
class TestSegmentUtilities:
    """Tests for segment de-duplication and collapsing logic."""

    def test_dedupe_consecutive(self) -> None:
        """Consecutive duplicate speaker/text pairs should be removed."""

        segments = [
            {"speaker": "A", "text": "Hello", "start": 0, "end": 1},
            {"speaker": "A", "text": "Hello", "start": 1, "end": 2},
            {"speaker": "A", "text": "Hi", "start": 2, "end": 3},
            {"speaker": "B", "text": "Hey", "start": 3, "end": 4},
            {"speaker": "B", "text": " hey ", "start": 4, "end": 5},
        ]

        result = dedupe_consecutive(segments)
        assert result == [
            {"speaker": "A", "text": "Hello", "start": 0, "end": 1},
            {"speaker": "A", "text": "Hi", "start": 2, "end": 3},
            {"speaker": "B", "text": "Hey", "start": 3, "end": 4},
        ]

    def test_collapse_nearby_repeats(self) -> None:
        """Repeated short phrases within the window are collapsed."""

        segments = [
            {"speaker": "A", "text": "repeat me", "start": 0, "end": 1},
            {"speaker": "A", "text": "Repeat me ", "start": 5, "end": 6},
            {"speaker": "A", "text": "repeat me", "start": 15, "end": 16},
            {"speaker": "A", "text": "a" * 90, "start": 17, "end": 18},
            {"speaker": "A", "text": "a" * 90, "start": 20, "end": 21},
        ]

        result = collapse_nearby_repeats(segments, window_s=12.0, max_len=80)
        assert result == [
            {"speaker": "A", "text": "repeat me", "start": 0, "end": 1},
            {"speaker": "A", "text": "repeat me", "start": 15, "end": 16},
            {"speaker": "A", "text": "a" * 90, "start": 17, "end": 18},
            {"speaker": "A", "text": "a" * 90, "start": 20, "end": 21},
        ]


class TestTranscriptionService:
    """Tests for the ``TranscriptionService`` facade."""

    def test_merge_parts(self, tmp_path: Path) -> None:
        """All ``*.json.part`` files should be merged and outputs written."""

        part1 = tmp_path / "a.json.part"
        part2 = tmp_path / "b.json.part"
        part1.write_text(json.dumps([{ "speaker": "A", "start": 0, "end": 1, "text": "hi" }]))
        part2.write_text(json.dumps([{ "speaker": "B", "start": 1, "end": 2, "text": "there" }]))

        service = TranscriptionService(TranscriptionOptions())
        out_base = tmp_path / "merged"
        count = service.merge_parts(str(tmp_path), str(out_base))

        assert count == 2
        assert (tmp_path / "merged.txt").exists()
        assert (tmp_path / "merged.srt").exists()
        out_json = json.loads((tmp_path / "merged.json").read_text())
        assert len(out_json) == 2

    def test_merge_parts_missing(self, tmp_path: Path) -> None:
        """Attempting to merge without part files should raise an error."""

        service = TranscriptionService(TranscriptionOptions())
        with pytest.raises(FileNotFoundError):
            service.merge_parts(str(tmp_path), str(tmp_path / "out"))


class TestRunParallelCaching:
    """Ensure that VAD parameters are computed once and reused."""

    def test_vad_parameters_cached(self, monkeypatch, tmp_path: Path) -> None:
        """``build_vad_parameters`` should be invoked only once."""

        from domain import transcription

        call_count = 0

        def fake_build_vad_parameters(opts: TranscriptionOptions) -> Dict:
            nonlocal call_count
            call_count += 1
            return {"foo": "bar"}

        monkeypatch.setattr(transcription, "build_vad_parameters", fake_build_vad_parameters)
        monkeypatch.setattr(transcription, "worker_transcribe", dummy_worker)

        options = transcription.TranscriptionOptions(vad=True)
        received_vad_params: List[Dict] = []

        def progress(msg: Dict) -> None:
            if msg.get("type") == "done":
                received_vad_params.append(msg["vad_params"])

        files = [str(tmp_path / "a.wav"), str(tmp_path / "b.wav")]
        transcription.run_parallel(
            files,
            workers=2,
            speakers={},
            options=options,
            progress_callback=progress,
        )

        assert call_count == 1
        assert received_vad_params == [{"foo": "bar"}, {"foo": "bar"}]


class TestVadConfiguration:
    """Ensure VAD configuration is passed through correctly."""

    def test_build_vad_parameters_returns_empty_dict_when_enabled(self) -> None:
        """Plain ``--vad`` should still enable backend-default VAD behavior."""

        assert build_vad_parameters(TranscriptionOptions(vad=True)) == {}

    def test_faster_whisper_enables_default_vad_without_custom_params(
        self, monkeypatch
    ) -> None:
        """The FasterWhisper backend should enable VAD with default settings."""

        from domain.FasterWhisper import transcribe as faster_whisper_transcribe

        captured_kwargs: Dict = {}

        class FakeSegment:
            start = 0.0
            end = 1.0
            text = "hello"

        class FakeWhisperModel:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def transcribe(self, _audio_path: str, **kwargs):
                captured_kwargs.update(kwargs)
                return [FakeSegment()], object()

        monkeypatch.setitem(
            sys.modules,
            "faster_whisper",
            types.SimpleNamespace(WhisperModel=FakeWhisperModel),
        )

        result = faster_whisper_transcribe.transcribe(
            "audio.wav",
            TranscriptionOptions(vad=True),
            {},
            "Speaker",
        )

        assert result == [
            {"start": 0.0, "end": 1.0, "text": "hello", "speaker": "Speaker"}
        ]
        assert captured_kwargs["vad_filter"] is True
        assert "vad_parameters" not in captured_kwargs

    def test_faster_whisper_reports_segment_progress(self, monkeypatch) -> None:
        """The FasterWhisper backend should stream segment end times as progress."""

        from domain.FasterWhisper import transcribe as faster_whisper_transcribe

        class FakeSegment:
            def __init__(self, start: float, end: float, text: str) -> None:
                self.start = start
                self.end = end
                self.text = text

        class FakeWhisperModel:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def transcribe(self, _audio_path: str, **_kwargs):
                return [
                    FakeSegment(0.0, 1.5, "hello"),
                    FakeSegment(1.5, 3.0, "there"),
                ], object()

        monkeypatch.setitem(
            sys.modules,
            "faster_whisper",
            types.SimpleNamespace(WhisperModel=FakeWhisperModel),
        )

        progress_positions: List[float] = []
        result = faster_whisper_transcribe.transcribe(
            "audio.wav",
            TranscriptionOptions(vad=True),
            {},
            "Speaker",
            progress_callback=progress_positions.append,
        )

        assert progress_positions == [1.5, 3.0]
        assert result == [
            {"start": 0.0, "end": 1.5, "text": "hello", "speaker": "Speaker"},
            {"start": 1.5, "end": 3.0, "text": "there", "speaker": "Speaker"},
        ]


class TestWorkerProgress:
    """Ensure worker processes forward streamed backend progress."""

    def test_worker_transcribe_emits_progress_updates(self, monkeypatch, tmp_path: Path) -> None:
        """Streaming backend progress should reach the parent queue for ETA updates."""

        from domain import transcription

        audio_file = tmp_path / "speaker.wav"
        audio_file.write_text("dummy")
        monkeypatch.setattr(transcription, "get_duration_seconds", lambda _path: 10.0)

        tick_values = iter([1.0, 1.3])
        monkeypatch.setattr(transcription.time, "time", lambda: next(tick_values))

        def fake_transcribe(
            audio_path: str,
            options: TranscriptionOptions,
            vad_params: Dict,
            speaker_label: str,
            progress_callback,
        ) -> List[Dict]:
            progress_callback(2.5)
            progress_callback(6.0)
            return [
                {"start": 0.0, "end": 2.5, "text": "hello", "speaker": speaker_label},
                {"start": 2.5, "end": 6.0, "text": "there", "speaker": speaker_label},
            ]

        progress_queue: "queue.Queue[Dict]" = queue.Queue()
        worker_transcribe(
            str(audio_file),
            TranscriptionOptions(),
            {},
            0,
            progress_queue,
            "Speaker",
            fake_transcribe,
        )

        messages = [progress_queue.get(timeout=1) for _ in range(5)]
        assert [message["type"] for message in messages] == [
            "start",
            "progress",
            "progress",
            "progress",
            "done",
        ]
        assert [message["pos"] for message in messages if message["type"] == "progress"] == [
            2.5,
            6.0,
            10.0,
        ]


class TestEngineSelection:
    """Ensure the correct backend is chosen based on the options."""

    def test_run_parallel_selects_backend(self, monkeypatch, tmp_path: Path) -> None:
        from domain import transcription

        # Reuse the ``dummy_worker`` to observe which backend is passed.
        monkeypatch.setattr(transcription, "worker_transcribe", dummy_worker)

        audio = tmp_path / "sample.wav"
        audio.write_text("dummy")
        files = [str(audio)]

        backends: List[str] = []

        def progress(msg: Dict) -> None:
            if msg.get("type") == "done":
                backends.append(msg.get("backend", ""))

        opts = transcription.TranscriptionOptions(engine="whisperx")
        transcription.run_parallel(
            files,
            workers=1,
            speakers={},
            options=opts,
            progress_callback=progress,
        )
        assert backends == ["domain.WhisperX.transcribe"]

        backends.clear()
        opts = transcription.TranscriptionOptions(engine="faster-whisper")
        transcription.run_parallel(
            files,
            workers=1,
            speakers={},
            options=opts,
            progress_callback=progress,
        )
        assert backends == ["domain.FasterWhisper.transcribe"]


class TestSkipExisting:
    """Ensure cached part files still participate in the final merge."""

    def test_run_parallel_returns_skipped_part_files(self, tmp_path: Path) -> None:
        """Skipped cached parts should be returned for downstream merging."""

        from domain import transcription

        audio_one = tmp_path / "speaker_one.wav"
        audio_two = tmp_path / "speaker_two.wav"
        audio_one.write_text("dummy")
        audio_two.write_text("dummy")

        part_one = tmp_path / "speaker_one.wav.json.part"
        part_two = tmp_path / "speaker_two.wav.json.part"
        part_one.write_text("[]")
        part_two.write_text("[]")

        parts = transcription.run_parallel(
            [str(audio_one), str(audio_two)],
            workers=2,
            speakers={},
            options=transcription.TranscriptionOptions(skip_existing=True),
        )

        assert set(parts) == {str(part_one), str(part_two)}
