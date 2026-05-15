"""Smoke tests for GUI helper functions that do not require a display."""

from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from gui.app import format_progress, transcript_dir_from_out_base


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
