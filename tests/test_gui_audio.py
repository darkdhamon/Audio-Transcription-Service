"""Tests for the GUI audio playback helpers."""

from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from gui.audio import (
    WindowsMciAudioPlayer,
    build_mci_staging_path,
    is_mci_safe_filename,
)


def test_is_mci_safe_filename_accepts_short_83_name() -> None:
    """Short 8.3-style names should not require MCI staging."""

    assert is_mci_safe_filename(Path("ats12345.wav")) is True


def test_is_mci_safe_filename_rejects_long_name() -> None:
    """Descriptive transcript filenames should be staged before MCI opens them."""

    assert is_mci_safe_filename(Path("Shadows of DraxisTranscript.synced.wav")) is False


def test_build_mci_staging_path_returns_short_temp_name(tmp_path: Path) -> None:
    """The staged MCI filename should stay within a conservative 8.3-style shape."""

    staging_path = build_mci_staging_path(
        Path(r"C:\Users\Bronze\Desktop\Shadows of DraxisTranscript.synced.wav"),
        temp_dir=tmp_path,
    )

    assert staging_path.parent == tmp_path
    assert len(staging_path.stem) == 8
    assert staging_path.suffix == ".wav"


def test_prepare_device_path_copies_long_name_to_staging(monkeypatch, tmp_path: Path) -> None:
    """Long filenames should be copied to an MCI-safe staging path."""

    source_path = tmp_path / "Shadows of DraxisTranscript.synced.wav"
    source_path.write_bytes(b"RIFF")
    staged_path = tmp_path / "ats12345.wav"

    player = WindowsMciAudioPlayer.__new__(WindowsMciAudioPlayer)
    player._staged_path = None
    monkeypatch.setattr("gui.audio.build_mci_staging_path", lambda _path: staged_path)

    copied = {}

    def fake_copyfile(src: Path, dst: Path) -> None:
        copied["src"] = Path(src)
        copied["dst"] = Path(dst)
        Path(dst).write_bytes(Path(src).read_bytes())

    monkeypatch.setattr("gui.audio.shutil.copyfile", fake_copyfile)

    device_path = player._prepare_device_path(source_path)

    assert device_path == staged_path
    assert copied == {"src": source_path, "dst": staged_path}
    assert staged_path.exists()


def test_open_device_falls_back_to_auto_detect_for_wav() -> None:
    """WAV playback should retry without an explicit device type when needed."""

    player = WindowsMciAudioPlayer.__new__(WindowsMciAudioPlayer)
    player.alias = "test_alias"
    commands = []

    def fake_send(command: str, allow_failure: bool = False) -> str:
        commands.append(command)
        if len(commands) == 1:
            raise RuntimeError("waveaudio failed")
        return ""

    player._send = fake_send

    player._open_device(Path(r"C:\Temp\ats12345.wav"))

    assert commands == [
        'open "C:\\Temp\\ats12345.wav" type waveaudio alias test_alias',
        'open "C:\\Temp\\ats12345.wav" alias test_alias',
    ]
