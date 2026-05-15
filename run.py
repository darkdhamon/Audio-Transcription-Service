#!/usr/bin/env python3
"""Convenience launcher for the audio transcription CLI.

Running this script adds the ``src`` directory to ``sys.path`` so that the
CLI can be executed without manual environment setup. This allows the
application to be started with a single double-click or by running
``python run.py`` from the repository root.

On first launch the script prompts for the location of the recordings
directory and stores it in ``appsettings.json``. If a previous session was
used, the launcher first asks whether that folder should be reused. Otherwise
session folders inside the recordings directory are listed in order of most
recent activity and the user is asked to choose one. Pressing :kbd:`Enter`
without a choice selects the most recent session. Afterward the user selects
or creates a game profile and the transcript is written to
``<session>/transcript/<CampaignName>Transcript``.

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
from domain.config import AppSettings, GameProfile, GameSettings, JsonConfigStore, LastSession

CONFIG_FILE = "appsettings.json"
GAME_FILE = "gamesettings.json"
LAST_SESSION_FILE = "lastsession.json"
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


def should_launch_cli_directly(argv: list[str]) -> bool:
    """Return ``True`` when ``run.py`` should act as a CLI pass-through."""

    return len(argv) > 1


def has_saved_recording_directory(config_path: Path) -> bool:
    """Return ``True`` when ``appsettings.json`` contains a stored path."""

    data = JsonConfigStore.load_dict(config_path)
    return bool(data.get("recording_directory") or data.get("RecordingDirectory"))


def is_recording_session_dir(path: Path) -> bool:
    """Return ``True`` when ``path`` already looks like a single session folder.

    A session folder typically contains one or more audio recordings directly
    inside the directory. Detecting that shape lets the launcher use the
    folder as-is instead of incorrectly listing helper subdirectories such as
    ``transcript`` or ``.qodo`` as if they were separate sessions.
    """

    if not path.is_dir():
        return False

    try:
        return any(
            child.is_file() and child.suffix.lower() in AUDIO_EXTENSIONS
            for child in path.iterdir()
        )
    except OSError:
        return False


def confirm_saved_recording_directory(recording_dir: Path, console: Console) -> bool:
    """Ask whether the previously saved folder should be reused."""

    folder_kind = "session folder" if is_recording_session_dir(recording_dir) else "recordings directory"
    while True:
        reply = console.input(
            f'Use saved {folder_kind} "{recording_dir}"? [Y/n]: '
        ).strip().lower()
        if reply in {"", "y", "yes"}:
            return True
        if reply in {"n", "no"}:
            return False
        console.print("Please answer Y or N.")


def ensure_recording_dir(config_path: Path, console: Console) -> Path:
    """Load or prompt for the recordings directory or a direct session folder."""

    settings = AppSettings.load(config_path)
    recording_dir = settings.recording_directory

    if has_saved_recording_directory(config_path) and recording_dir.is_dir():
        if confirm_saved_recording_directory(recording_dir, console):
            return recording_dir

    while True:
        reply = console.input(
            "Enter full path to recordings directory or session folder: "
        ).strip()
        if not reply:
            continue

        candidate = Path(reply)
        if candidate.is_dir():
            settings.recording_directory = candidate
            settings.save(config_path)
            return candidate

        console.print("Directory not found. Please try again.")


def choose_session(
    recordings_dir: Path,
    console: Console,
    last_session_name: str | None = None,
) -> Path:
    """Prompt the user to choose a recording session.

    Sessions are ordered by modification time with the most recent first. When
    a previously used session still exists, the user is first asked whether it
    should be reused. If that session is declined, pressing :kbd:`Enter`
    without typing a number selects the newest session automatically.
    """

    sessions = [p for p in recordings_dir.iterdir() if p.is_dir()]
    if not sessions:
        console.print("No session directories found.")
        raise SystemExit(1)

    sessions.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    # Reuse the last successful session only when the user explicitly confirms
    # it, which avoids silently pointing a new transcript at the wrong folder.
    last_session = next(
        (session for session in sessions if session.name == last_session_name),
        None,
    )
    if last_session is not None:
        while True:
            reply = console.input(
                f'Use last recording session "{last_session.name}"? [Y/n]: '
            ).strip().lower()
            if reply in {"", "y", "yes"}:
                return last_session
            if reply in {"n", "no"}:
                break
            console.print("Please answer Y or N.")

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
    last_path = root / LAST_SESSION_FILE
    last = LastSession.load(last_path)
    if is_recording_session_dir(recordings_dir):
        session_dir = recordings_dir
    else:
        session_dir = choose_session(recordings_dir, console, last.recording_session)

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
