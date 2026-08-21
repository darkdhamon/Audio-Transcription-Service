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
from typing import Any, Dict, List, Optional
import json


class JsonConfigStore:
    """Safely load and save the small JSON documents used by the app."""

    @staticmethod
    def load_dict(path: Path) -> Dict[str, Any]:
        """Return the JSON object stored in ``path`` or an empty dict.

        Local state files may be missing or temporarily malformed, for example
        when a merge leaves conflict markers behind.  Returning an empty
        configuration keeps the launcher usable while preserving the file for
        manual inspection.
        """

        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def save_dict(path: Path, data: Dict[str, Any]) -> None:
        """Persist ``data`` as formatted JSON."""

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


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

        data = JsonConfigStore.load_dict(path)
        # Support both the original launcher key and the newer dataclass-style
        # key so existing user configuration keeps working.
        directory = data.get("recording_directory") or data.get("RecordingDirectory")
        if not directory:
            # Default to current working directory if nothing was stored yet.
            return cls(recording_directory=Path.cwd())
        return cls(recording_directory=Path(str(directory)))

    def save(self, path: Path) -> None:
        """Persist settings to ``path``."""

        JsonConfigStore.save_dict(
            path,
            {"RecordingDirectory": str(self.recording_directory)},
        )


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

    def get_character(self, display_name: str) -> Optional[str]:
        """Return the character name for ``display_name`` if known.

        Parameters
        ----------
        display_name:
            TeamSpeak display name of the player.

        Returns
        -------
        Optional[str]
            The previously used character name or ``None`` if the
            player has not been recorded before.
        """

        for player in self.players:
            if player.display_name.lower() == display_name.lower():
                return player.character_name
        return None

    def set_player(self, display_name: str, character_name: str) -> None:
        """Add or update a player's character mapping.

        The method either updates an existing player's character name or
        appends a new :class:`Player` entry if the player is encountered
        for the first time.
        """

        for player in self.players:
            if player.display_name.lower() == display_name.lower():
                player.character_name = character_name
                return
        self.players.append(Player(display_name=display_name, character_name=character_name))


@dataclass
class GameSettings:
    """Collection of named game profiles."""

    profiles: Dict[str, GameProfile] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "GameSettings":
        raw = JsonConfigStore.load_dict(path)
        profiles = {
            name: GameProfile(
                campaign=p.get("campaign", name),
                players=[Player(**pl) for pl in p.get("players", [])],
            )
            for name, p in raw.get("profiles", {}).items()
        }
        return cls(profiles=profiles)

    def save(self, path: Path) -> None:
        data = {
            "profiles": {
                name: {
                    "campaign": profile.campaign,
                    "players": [asdict(p) for p in profile.players],
                }
                for name, profile in self.profiles.items()
            }
        }
        JsonConfigStore.save_dict(path, data)


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
        data = JsonConfigStore.load_dict(path)
        return cls(
            recording_session=data.get("recording_session"),
            game_profile=data.get("game_profile"),
        )

    def save(self, path: Path) -> None:
        JsonConfigStore.save_dict(
            path,
            {
                "recording_session": self.recording_session,
                "game_profile": self.game_profile,
            },
        )

