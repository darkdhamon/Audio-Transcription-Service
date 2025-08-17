#!/usr/bin/env python3
"""Convenience launcher for the audio transcription CLI.

Running this script adds the ``src`` directory to ``sys.path`` so that the
CLI can be executed without manual environment setup.  This allows the
application to be started with a single double‑click or by running
``python run.py`` from the repository root.

On first launch the script prompts for the location of the recordings
directory and stores it in ``appsettings.json``.  Session folders inside the
recordings directory are listed and the user is asked to choose one.  The
selected session becomes the CLI's ``--input`` argument.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys
from typing import Any

from rich.console import Console

CONFIG_FILE = "appsettings.json"


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
    sessions = [p for p in recordings_dir.iterdir() if p.is_dir()]
    if not sessions:
        console.print("No session directories found.")
        raise SystemExit(1)
    console.print("\nSessions:")
    for idx, sess in enumerate(sessions, 1):
        console.print(f"  {idx}. {sess.name}")
    while True:
        choice = console.input("Select session number: ").strip()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(sessions):
                return sessions[idx - 1]
        console.print("Invalid selection.")


def main() -> None:
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "src"))

    console = Console()
    config_path = root / CONFIG_FILE
    recordings_dir = ensure_recording_dir(config_path, console)
    session_dir = choose_session(recordings_dir, console)

    if "--input" not in sys.argv:
        sys.argv.extend(["--input", str(session_dir)])

    from cli.app import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
