#!/usr/bin/env python3
"""Convenience launcher for the audio transcription CLI.

Running this script adds the ``src`` directory to ``sys.path`` so that the
CLI can be executed without manual environment setup. This allows the
application to be started with a single double-click or by running
``python run.py`` from the repository root.

On first launch the script prompts for the location of the recordings
directory and stores it in ``appsettings.json``. Session folders inside the
recordings directory are listed in order of most recent activity and the user
is asked to choose one. Pressing :kbd:`Enter` without a choice selects the
most recent session. Afterward the user selects or creates a game profile and
the transcript is written to ``<session>/transcript/<CampaignName>Transcript``.

When explicit command-line arguments are supplied, the script behaves like a
thin CLI launcher and forwards directly to :mod:`cli.app` without running the
interactive session/profile selection flow.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

# Ensure application modules can be imported without installing the package.
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "src"))

from rich.console import Console

from cli.app import main as cli_main
from domain.config import AppSettings, GameProfile, GameSettings, LastSession

CONFIG_FILE = "appsettings.json"
GAME_FILE = "gamesettings.json"
LAST_SESSION_FILE = "lastsession.json"


def should_launch_cli_directly(argv: list[str]) -> bool:
    """Return ``True`` when ``run.py`` should act as a CLI pass-through."""

    return len(argv) > 1


def ensure_recording_dir(config_path: Path, console: Console) -> Path:
    """Load the saved recordings directory or prompt until a valid one exists."""

    settings = AppSettings.load(config_path)
    recording_dir = settings.recording_directory

    while not recording_dir.is_dir():
        reply = console.input("Enter full path to recordings directory: ").strip()
        if not reply:
            continue

        candidate = Path(reply)
        if candidate.is_dir():
            settings.recording_directory = candidate
            settings.save(config_path)
            break

        console.print("Directory not found. Please try again.")
        recording_dir = Path()

    return settings.recording_directory


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
    """Select an existing game profile or create a new one."""

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
    """Launch the interactive helper or forward directly to the CLI."""

    console = Console()

    if should_launch_cli_directly(sys.argv):
        _ = cli_main()
        return

    config_path = root / CONFIG_FILE
    recordings_dir = ensure_recording_dir(config_path, console)
    session_dir = choose_session(recordings_dir, console)

    profile_name, profile = choose_game_profile(root / GAME_FILE, console)

    # Prepare the default output location for the selected session/profile.
    transcript_dir = session_dir / "transcript"
    transcript_dir.mkdir(exist_ok=True)
    out_base = transcript_dir / f"{profile.campaign}Transcript"

    if "--input" not in sys.argv:
        sys.argv.extend(["--input", str(session_dir)])
    if "--out" not in sys.argv:
        sys.argv.extend(["--out", str(out_base)])

    # Persist the most recent interactive selection for the next launch.
    last_path = root / LAST_SESSION_FILE
    last = LastSession.load(last_path)
    last.recording_session = session_dir.name
    last.game_profile = profile_name
    last.save(last_path)

    _ = cli_main(profile)

    # Persist any updated character mappings to the game profile.
    settings_path = root / GAME_FILE
    settings = GameSettings.load(settings_path)
    settings.profiles[profile_name] = profile
    settings.save(settings_path)


if __name__ == "__main__":
    main()
