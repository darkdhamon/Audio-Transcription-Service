from __future__ import annotations

"""Configuration models and helpers for the transcription service.

This module manages three configuration files:

- ``appsettings.json``: Stores application wide settings such as the
  default recording directory.
- ``gamesettings.json``: Contains multiple game profiles.  Each profile
  stores the campaign name and a list of players with their TeamSpeak
  display name and character name.
- ``lastsession.json``: Tracks the last recording session that was
  processed and which game profile was used.

These helpers are intentionally free of any user interface code so the
rest of the application can consume them from a CLI, GUI or web
framework without modification.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional
import json

# ---------------------------------------------------------------------------
# App settings
# ---------------------------------------------------------------------------


@dataclass
class AppSettings:
    """Application wide settings.

    Attributes
    ----------
    recording_directory:
        Path where recording sessions are stored.  Each session is a
        sub-directory inside this folder.
    """

    recording_directory: Path

    @classmethod
    def load(cls, path: Path) -> "AppSettings":
        """Load settings from ``path``.  If the file does not exist a
        default configuration is returned."""

        if not path.exists():
            # Default to current working directory if nothing was stored yet
            return cls(recording_directory=Path.cwd())
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(recording_directory=Path(data.get("recording_directory", ".")))

    def save(self, path: Path) -> None:
        """Persist settings to ``path``."""

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump({"recording_directory": str(self.recording_directory)}, f, indent=2)


# ---------------------------------------------------------------------------
# Game settings
# ---------------------------------------------------------------------------


@dataclass
class Player:
    """Represents a single player in a campaign."""

    display_name: str
    character_name: str


@dataclass
class GameProfile:
    """A game profile describing a campaign and its players."""

    campaign: str
    players: List[Player] = field(default_factory=list)


@dataclass
class GameSettings:
    """Collection of named game profiles."""

    profiles: Dict[str, GameProfile] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "GameSettings":
        if not path.exists():
            return cls()
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        profiles = {
            name: GameProfile(
                campaign=p.get("campaign", name),
                players=[Player(**pl) for pl in p.get("players", [])],
            )
            for name, p in raw.get("profiles", {}).items()
        }
        return cls(profiles=profiles)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "profiles": {
                name: {
                    "campaign": profile.campaign,
                    "players": [asdict(p) for p in profile.players],
                }
                for name, profile in self.profiles.items()
            }
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Last session tracking
# ---------------------------------------------------------------------------


@dataclass
class LastSession:
    """Tracks the last processed recording session and game profile."""

    recording_session: Optional[str] = None
    game_profile: Optional[str] = None

    @classmethod
    def load(cls, path: Path) -> "LastSession":
        if not path.exists():
            return cls()
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            recording_session=data.get("recording_session"),
            game_profile=data.get("game_profile"),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "recording_session": self.recording_session,
                    "game_profile": self.game_profile,
                },
                f,
                indent=2,
            )

