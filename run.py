#!/usr/bin/env python3
"""Convenience launcher for the audio transcription CLI.

Running this script adds the ``src`` directory to ``sys.path`` so that the
CLI can be executed without manual environment setup. This allows the
application to be started with a single double‑click or by running
``python run.py`` from the repository root.

On first launch the script prompts for the location of the recordings
directory and stores it in ``appsettings.json``. Session folders inside the
recordings directory are listed in order of most recent activity and the user
is asked to choose one. Pressing :kbd:`Enter` without a choice selects the
most recent session. Afterward the user selects or creates a game profile and
the transcript is written to ``<session>/transcript/<CampaignName>Transcript``.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import sys

# Ensure application modules can be imported without installing the package
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "src"))

from typing import Any
from rich.console import Console

from domain.config import GameSettings, GameProfile, LastSession

CONFIG_FILE = "appsettings.json"
GAME_FILE = "gamesettings.json"
LAST_SESSION_FILE = "lastsession.json"


def load_settings(config_path: Path) -> dict[str, Any]:
    if config_path.exists():
        try:
            with config_path.open() as fh:
                return json.load(fh)
        except Exception:
            return {}
    return {}


def save_settings(config_path: Path, settings: dict[str, Any]) -> None:
    config_path.write_text(json.dumps(settings, indent=2))


def ensure_recording_dir(config_path: Path, console: Console) -> Path:
    settings = load_settings(config_path)
    rec_dir = settings.get("RecordingDirectory", "")
    while not rec_dir or not Path(rec_dir).is_dir():
        rec_dir = console.input("Enter full path to recordings directory: ").strip()
        if not rec_dir:
            continue
        if Path(rec_dir).is_dir():
            settings["RecordingDirectory"] = rec_dir
            save_settings(config_path, settings)
            break
        console.print("Directory not found. Please try again.")
        rec_dir = ""
    return Path(rec_dir)


def choose_session(recordings_dir: Path, console: Console) -> Path:
    """Prompt the user to choose a recording session.

    Sessions are ordered by modification time with the most recent first. If
    the user presses :kbd:`Enter` without typing a number the newest session
    is selected automatically.
    """

    sessions = [p for p in recordings_dir.iterdir() if p.is_dir()]
    if not sessions:
        console.print("No session directories found.")
        raise SystemExit(1)

    sessions.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    console.print("\nSessions:")
    for idx, sess in enumerate(sessions, 1):
        stamp = datetime.fromtimestamp(sess.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        console.print(f"  {idx}. {sess.name} ({stamp})")

    while True:
        choice = console.input("Select session number [1]: ").strip()
        if not choice:
            return sessions[0]
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(sessions):
                return sessions[idx - 1]
        console.print("Invalid selection.")


def choose_game_profile(settings_path: Path, console: Console) -> tuple[str, GameProfile]:
    """Select an existing game profile or create a new one.

    Parameters
    ----------
    settings_path:
        Location of the ``gamesettings.json`` file.

    Returns
    -------
    tuple[str, GameProfile]
        The name of the selected profile and the profile instance.
    """

    settings = GameSettings.load(settings_path)
    if settings.profiles:
        console.print("\nGame profiles:")
        names = list(settings.profiles.keys())
        for idx, name in enumerate(names, 1):
            profile = settings.profiles[name]
            console.print(f"  {idx}. {name} ({profile.campaign})")
        choice = console.input(
            "Select profile number or press Enter to create new: "
        ).strip()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(names):
                return names[idx - 1], settings.profiles[names[idx - 1]]

    console.print("\nCreating new game profile.")
    while True:
        profile_name = console.input("Enter profile name: ").strip()
        if profile_name:
            break
    campaign = console.input("Enter name of the campaign: ").strip() or profile_name
    profile = GameProfile(campaign=campaign)
    settings.profiles[profile_name] = profile
    settings.save(settings_path)
    return profile_name, profile


def main() -> None:
    # ``root`` already points to repository root via module-level setup.
    console = Console()
    config_path = root / CONFIG_FILE
    recordings_dir = ensure_recording_dir(config_path, console)
    session_dir = choose_session(recordings_dir, console)

    profile_name, profile = choose_game_profile(root / GAME_FILE, console)

    # Prepare output directory ``<session>/transcript``
    transcript_dir = session_dir / "transcript"
    transcript_dir.mkdir(exist_ok=True)
    out_base = transcript_dir / f"{profile.campaign}Transcript"

    if "--input" not in sys.argv:
        sys.argv.extend(["--input", str(session_dir)])
    if "--out" not in sys.argv:
        sys.argv.extend(["--out", str(out_base)])

    # Remember last used session and profile
    last_path = root / LAST_SESSION_FILE
    last = LastSession.load(last_path)
    last.recording_session = session_dir.name
    last.game_profile = profile_name
    last.save(last_path)

    from cli.app import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
