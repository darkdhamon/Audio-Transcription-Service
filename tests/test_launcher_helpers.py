"""Tests for shared launcher helpers used by CLI and GUI front ends."""

from pathlib import Path
import os
import sys


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from domain.config import GameProfile, Player
from domain.launcher import (
    SessionCatalog,
    build_speaker_assignments,
    list_session_audio_files,
    looks_like_session_directory,
)


def create_session(path: Path, name: str, modified_at: int) -> Path:
    """Create a fake session directory with a controlled modification time."""

    session_path = path / name
    session_path.mkdir()
    os.utime(session_path, (modified_at, modified_at))
    return session_path


def create_audio_file(path: Path, name: str = "speaker.wav") -> Path:
    """Create a small placeholder audio file for launcher tests."""

    audio_path = path / name
    audio_path.write_bytes(b"RIFF")
    return audio_path


def test_session_catalog_detects_direct_session_folder(tmp_path: Path) -> None:
    """Direct session folders should not be treated like parent recordings directories."""

    create_audio_file(tmp_path, "playback_player_001.wav")

    catalog = SessionCatalog.discover(tmp_path)

    assert catalog.is_direct_session is True
    assert catalog.direct_session_path == tmp_path
    assert catalog.sessions == []


def test_session_catalog_lists_sessions_in_recency_order(tmp_path: Path) -> None:
    """Session directories should be sorted from most to least recent."""

    create_session(tmp_path, "older", 100)
    newer = create_session(tmp_path, "newer", 200)

    catalog = SessionCatalog.discover(tmp_path)

    assert [session.name for session in catalog.sessions] == ["newer", "older"]
    assert catalog.find_by_name(newer.name) == catalog.sessions[0]


def test_looks_like_session_directory_ignores_helper_subfolders(tmp_path: Path) -> None:
    """Helper folders should not prevent an audio directory from being recognized."""

    create_audio_file(tmp_path)
    (tmp_path / ".qodo").mkdir()
    (tmp_path / "transcript").mkdir()

    assert looks_like_session_directory(tmp_path) is True


def test_list_session_audio_files_ignores_capture_tracks(tmp_path: Path) -> None:
    """Only transcribable playback files should be returned to front ends."""

    create_audio_file(tmp_path, "playback_alpha_001.wav")
    create_audio_file(tmp_path, "capture_2024-01-01_10-00-00.wav")

    files = list_session_audio_files(tmp_path)

    assert [file.name for file in files] == ["playback_alpha_001.wav"]


def test_build_speaker_assignments_uses_profile_character_names(tmp_path: Path) -> None:
    """Existing game profiles should seed speaker names for GUI editing."""

    audio_file = create_audio_file(tmp_path, "playback_bronze_001.wav")
    profile = GameProfile(
        campaign="Campaign",
        players=[Player(display_name="bronze", character_name="GM")],
    )

    assignments = build_speaker_assignments([audio_file], profile)

    assert assignments[0].display_name == "bronze"
    assert assignments[0].suggested_name == "GM"
