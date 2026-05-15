from __future__ import annotations

"""Shared launcher helpers used by both the CLI wrapper and the GUI.

The project now exposes multiple front ends that need the same answers for
questions such as "is this path a direct session folder?" and "what speaker
name should this file suggest?". Centralizing that logic here keeps the
command-line and GUI launchers consistent.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

from domain.config import GameProfile
from domain.transcription import derive_suggested_label, list_audio_files

AUDIO_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
    ".wma",
}


@dataclass(frozen=True)
class SessionEntry:
    """Represents a selectable recording session."""

    path: Path
    modified_at: float

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def modified_label(self) -> str:
        return datetime.fromtimestamp(self.modified_at).strftime("%Y-%m-%d %H:%M")


@dataclass(frozen=True)
class SpeakerAssignment:
    """Suggested speaker mapping for a single audio file."""

    file_path: Path
    display_name: str
    suggested_name: str


@dataclass
class SessionCatalog:
    """View of the recording path chosen by the user.

    The selected path can be either a parent recordings directory containing
    many session folders or a direct session folder containing audio files.
    """

    source_path: Path
    direct_session_path: Optional[Path] = None
    sessions: List[SessionEntry] = field(default_factory=list)

    @property
    def is_direct_session(self) -> bool:
        return self.direct_session_path is not None

    @classmethod
    def discover(cls, source_path: Path) -> "SessionCatalog":
        """Inspect ``source_path`` and return the matching session view."""

        if looks_like_session_directory(source_path):
            return cls(source_path=source_path, direct_session_path=source_path)

        sessions = [
            SessionEntry(path=path, modified_at=path.stat().st_mtime)
            for path in source_path.iterdir()
            if path.is_dir()
        ]
        sessions.sort(key=lambda session: session.modified_at, reverse=True)
        return cls(source_path=source_path, sessions=sessions)

    def find_by_name(self, session_name: Optional[str]) -> Optional[SessionEntry]:
        """Return the session entry matching ``session_name`` if present."""

        if not session_name:
            return None
        for session in self.sessions:
            if session.name == session_name:
                return session
        return None


def looks_like_session_directory(path: Path) -> bool:
    """Return ``True`` when ``path`` appears to be a direct session folder."""

    if not path.is_dir():
        return False

    try:
        return any(
            child.is_file() and child.suffix.lower() in AUDIO_EXTENSIONS
            for child in path.iterdir()
        )
    except OSError:
        return False


def list_session_audio_files(session_path: Path) -> List[Path]:
    """Return the transcribable audio files contained in ``session_path``."""

    return [Path(path) for path in list_audio_files(str(session_path))]


def build_speaker_assignments(
    files: Iterable[Path],
    profile: Optional[GameProfile] = None,
) -> List[SpeakerAssignment]:
    """Build speaker-name suggestions for the provided audio ``files``."""

    assignments: List[SpeakerAssignment] = []
    for file_path in files:
        display_name = derive_suggested_label(file_path.name)
        suggested_name = display_name
        if profile is not None:
            suggested_name = profile.get_character(display_name) or display_name
        assignments.append(
            SpeakerAssignment(
                file_path=file_path,
                display_name=display_name,
                suggested_name=suggested_name,
            )
        )
    return assignments
