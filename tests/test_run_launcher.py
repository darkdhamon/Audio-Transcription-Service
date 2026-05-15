"""Regression tests for the top-level ``run.py`` launcher."""

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))

import run  # noqa: E402
from domain.config import AppSettings, LastSession  # noqa: E402


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
