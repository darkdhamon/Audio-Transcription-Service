"""Smoke tests for GUI helper functions that do not require a display."""

from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from domain.session_media import TranscriptSegment
from gui.app import (
    find_segment_index_at_position,
    format_playback_clock,
    format_progress,
    format_transcript_timestamp,
    transcript_dir_from_out_base,
)


def test_format_progress_with_total() -> None:
    """Percentages should be included when the total duration is known."""

    assert format_progress(5.0, 20.0) == " 25.0% (5.0s / 20.0s)"


def test_format_progress_without_total() -> None:
    """Unknown total durations should fall back to a simple seconds display."""

    assert format_progress(3.5, 0.0) == "3.5s"


def test_transcript_dir_from_out_base_returns_parent(tmp_path: Path) -> None:
    """Output helpers should keep transcript paths rooted in the session folder."""

    out_base = tmp_path / "transcript" / "CampaignTranscript"

    assert transcript_dir_from_out_base(out_base) == out_base.parent


def test_format_transcript_timestamp_uses_clock_style() -> None:
    """Transcript timestamps should render as a readable playback clock."""

    assert format_transcript_timestamp(65.9) == "00:01:05"


def test_format_playback_clock_uses_hours_minutes_seconds() -> None:
    """Playback labels should expose the elapsed audio position cleanly."""

    assert format_playback_clock(3_661_000) == "01:01:01"


def test_find_segment_index_at_position_returns_matching_segment() -> None:
    """The transcript viewer should highlight the active spoken segment."""

    segments = [
        TranscriptSegment(start=0.0, end=2.0, speaker="A", text="hello"),
        TranscriptSegment(start=4.0, end=6.0, speaker="B", text="world"),
    ]

    assert find_segment_index_at_position(segments, 4.5) == 1


def test_find_segment_index_at_position_keeps_previous_segment_during_gap() -> None:
    """Brief gaps should retain the previous line until the next one starts."""

    segments = [
        TranscriptSegment(start=0.0, end=2.0, speaker="A", text="hello"),
        TranscriptSegment(start=4.0, end=6.0, speaker="B", text="world"),
    ]

    assert find_segment_index_at_position(segments, 3.0) == 0
