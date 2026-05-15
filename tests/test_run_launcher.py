"""Regression tests for the top-level ``run.py`` launcher."""

import os
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))

import run  # noqa: E402
from domain.config import AppSettings, LastSession  # noqa: E402


class StubConsole:
    """Small console double that records prompts for launcher tests."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = iter(replies)
        self.messages: list[str] = []

    def input(self, prompt: str) -> str:
        self.messages.append(prompt)
        return next(self._replies)

    def print(self, message: object = "") -> None:
        self.messages.append(str(message))


def create_session(path: Path, name: str, modified_at: int) -> Path:
    """Create a fake session directory with a controlled modification time."""

    session_path = path / name
    session_path.mkdir()
    os.utime(session_path, (modified_at, modified_at))
    return session_path


def test_app_settings_load_supports_legacy_recording_directory_key(tmp_path: Path) -> None:
    """Existing ``appsettings.json`` files should keep working after the refactor."""

    config_path = tmp_path / "appsettings.json"
    expected_dir = tmp_path / "recordings"
    config_path.write_text(
        '{\n  "RecordingDirectory": "%s"\n}\n'
        % str(expected_dir).replace("\\", "\\\\"),
        encoding="utf-8",
    )

    settings = AppSettings.load(config_path)
    assert settings.recording_directory == expected_dir


def test_last_session_load_ignores_invalid_json(tmp_path: Path) -> None:
    """Malformed local session files should not prevent the launcher from starting."""

    session_path = tmp_path / "lastsession.json"
    session_path.write_text(
        "{\n<<<<<<< Updated upstream\n  \"recording_session\": \"old\",\n=======\n"
        "  \"recording_session\": \"new\",\n>>>>>>> Stashed changes\n"
        "  \"game_profile\": \"Campaign\"\n}\n",
        encoding="utf-8",
    )

    session = LastSession.load(session_path)
    assert session.recording_session is None
    assert session.game_profile is None


def test_main_with_cli_args_bypasses_interactive_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit CLI usage should not trigger the interactive launcher prompts."""

    cli_calls: list[object] = []

    def fake_cli_main(profile=None):
        cli_calls.append(profile)

    def fail(*_args, **_kwargs):
        raise AssertionError("interactive setup should not run when CLI args are supplied")

    monkeypatch.setattr(run, "cli_main", fake_cli_main)
    monkeypatch.setattr(run, "ensure_recording_dir", fail)
    monkeypatch.setattr(run, "choose_session", fail)
    monkeypatch.setattr(run, "choose_game_profile", fail)
    monkeypatch.setattr(run.sys, "argv", ["run.py", "--help"])

    run.main()

    assert cli_calls == [None]


def test_choose_session_reuses_confirmed_last_session(tmp_path: Path) -> None:
    """The launcher should reuse the saved session when the user confirms it."""

    latest_session = create_session(tmp_path, "latest", 200)
    _ = create_session(tmp_path, "previous", 100)
    console = StubConsole([""])

    chosen = run.choose_session(tmp_path, console, last_session_name=latest_session.name)

    assert chosen == latest_session
    assert console.messages == [f'Use last recording session "{latest_session.name}"? [Y/n]: ']


def test_choose_session_lists_sessions_after_rejecting_last_session(tmp_path: Path) -> None:
    """Rejecting the saved session should fall back to the numbered session list."""

    latest_session = create_session(tmp_path, "latest", 200)
    previous_session = create_session(tmp_path, "previous", 100)
    console = StubConsole(["n", "2"])

    chosen = run.choose_session(tmp_path, console, last_session_name=latest_session.name)

    assert chosen == previous_session
    assert any(message == "\nSessions:" for message in console.messages)
