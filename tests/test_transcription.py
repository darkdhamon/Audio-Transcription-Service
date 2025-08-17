"""Test suite for core transcription utilities and services.

This module exercises helper functions and service methods used during
transcription.  The tests validate timestamp parsing, offset detection,
segment de-duplication, repeat collapsing, and merging output parts.
"""

from datetime import datetime, timezone
from pathlib import Path
import json
import sys

import pytest


# Ensure the ``src`` package is importable when running tests directly.
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from domain.transcription import (
    TranscriptionOptions,
    TranscriptionService,
    collapse_nearby_repeats,
    dedupe_consecutive,
    filename_offsets,
    parse_ts_from_name,
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
