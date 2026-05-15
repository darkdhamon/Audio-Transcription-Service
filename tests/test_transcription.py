"""Test suite for core transcription utilities and services.

This module exercises helper functions and service methods used during
transcription.  The tests validate timestamp parsing, offset detection,
segment de-duplication, repeat collapsing, and merging output parts.
"""

from datetime import datetime, timezone
from pathlib import Path
import json
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
    filename_offsets,
    parse_ts_from_name,
)


def dummy_worker(
    file_path: str,
    options: TranscriptionOptions,
    vad_params: Dict,
    offset: float,
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


class TestParseTimestamp:
    """Tests for parsing timestamps embedded in filenames."""

    def test_parse_ts_from_name_valid(self) -> None:
        """A well-formed filename should yield the correct UTC timestamp."""

        name = "2023-10-31_15-45-00.123.wav"
        expected = datetime(2023, 10, 31, 15, 45, 0, 123000, tzinfo=timezone.utc).timestamp()
        result = parse_ts_from_name(name)
        assert result == pytest.approx(expected)

    def test_parse_ts_from_name_invalid(self) -> None:
        """Filenames without a timestamp pattern should return ``None``."""

        assert parse_ts_from_name("no_timestamp.wav") is None


class TestFilenameOffsets:
    """Tests for computing filename based time offsets."""

    def test_filename_offsets_capture_baseline(self, tmp_path: Path) -> None:
        """Offsets are computed relative to the capture file when present."""

        capture = tmp_path / "capture_2024-01-01_10-00-00.wav"
        other = tmp_path / "playback_user_2024-01-01_10-00-05.wav"
        capture.write_text("dummy")
        other.write_text("dummy")
        files = [str(other)]

        offsets = filename_offsets(files, str(tmp_path), baseline="capture")
        assert offsets == {other.name.lower(): 5.0}

    def test_filename_offsets_no_candidates(self, tmp_path: Path) -> None:
        """If no files contain timestamps, an empty dictionary is returned."""

        sample = tmp_path / "audio.wav"
        sample.write_text("dummy")
        offsets = filename_offsets([str(sample)], str(tmp_path), baseline="capture")
        assert offsets == {}


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
            offsets={},
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
            0.0,
            "Speaker",
        )

        assert result == [
            {"start": 0.0, "end": 1.0, "text": "hello", "speaker": "Speaker"}
        ]
        assert captured_kwargs["vad_filter"] is True
        assert "vad_parameters" not in captured_kwargs


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
            offsets={},
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
            offsets={},
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
            offsets={},
            options=transcription.TranscriptionOptions(skip_existing=True),
        )

        assert set(parts) == {str(part_one), str(part_two)}
