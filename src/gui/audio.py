from __future__ import annotations

"""Small audio playback wrapper for the desktop GUI."""

from pathlib import Path
import ctypes
import hashlib
import shutil
import sys
import tempfile
from typing import Optional


def is_mci_safe_filename(audio_path: Path) -> bool:
    """Return whether ``audio_path`` already follows a conservative 8.3-style name."""

    stem = audio_path.stem
    suffix = audio_path.suffix
    if not (1 <= len(stem) <= 8) or len(suffix) > 4:
        return False
    allowed_stem_characters = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    allowed_suffix_characters = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    return all(character in allowed_stem_characters for character in stem) and all(
        character in allowed_suffix_characters for character in suffix.lstrip(".")
    )


def build_mci_staging_path(audio_path: Path, temp_dir: Optional[Path] = None) -> Path:
    """Return a short local filename that MCI can open reliably."""

    safe_temp_dir = temp_dir or Path(tempfile.gettempdir())
    extension = audio_path.suffix.lower()[:4] or ".wav"
    digest = hashlib.sha1(str(audio_path.resolve()).encode("utf-8")).hexdigest()[:5]
    return safe_temp_dir / f"ats{digest}{extension}"


class UnsupportedAudioPlayer:
    """No-op player returned on platforms without the Windows MCI API."""

    supported = False
    availability_message = "In-app audio playback is only available on Windows."

    def __init__(self) -> None:
        self.source_path: Optional[Path] = None

    def load(self, audio_path: Path) -> None:
        raise RuntimeError(self.availability_message)

    def play(self, start_ms: Optional[int] = None) -> None:
        raise RuntimeError(self.availability_message)

    def pause(self) -> None:
        raise RuntimeError(self.availability_message)

    def resume(self) -> None:
        raise RuntimeError(self.availability_message)

    def stop(self) -> None:
        self.source_path = None

    def close(self) -> None:
        self.source_path = None

    def is_loaded(self) -> bool:
        return False

    def is_playing(self) -> bool:
        return False

    def is_paused(self) -> bool:
        return False

    def get_position_ms(self) -> int:
        return 0

    def get_length_ms(self) -> int:
        return 0


class WindowsMciAudioPlayer:
    """Plays WAV audio through the Windows MCI API so the GUI can query position."""

    supported = True
    availability_message = "In-app audio playback is ready."

    def __init__(self, alias: str = "audio_transcription_service_player") -> None:
        self.alias = alias
        self.source_path: Optional[Path] = None
        self._device_path: Optional[Path] = None
        self._staged_path: Optional[Path] = None
        self._paused = False
        self._length_ms = 0
        self._mci_send = ctypes.windll.winmm.mciSendStringW
        self._mci_error_string = ctypes.windll.winmm.mciGetErrorStringW

    def _send(self, command: str, allow_failure: bool = False) -> str:
        """Execute an MCI command and return its text response."""

        response = ctypes.create_unicode_buffer(260)
        error_code = self._mci_send(command, response, len(response), 0)
        if error_code == 0:
            return response.value
        if allow_failure:
            return ""

        error_message = ctypes.create_unicode_buffer(260)
        if self._mci_error_string(error_code, error_message, len(error_message)):
            details = error_message.value
        else:
            details = f"MCI error {error_code}"
        raise RuntimeError(details)

    def _close_alias_if_open(self) -> None:
        self._send(f"close {self.alias}", allow_failure=True)
        self._cleanup_staged_path()
        self.source_path = None
        self._device_path = None
        self._paused = False
        self._length_ms = 0

    def load(self, audio_path: Path) -> None:
        """Open ``audio_path`` and prepare it for time-based playback controls."""

        resolved_path = audio_path.resolve()
        self._close_alias_if_open()
        device_path = self._prepare_device_path(resolved_path)
        self._open_device(device_path)
        self._send(f"set {self.alias} time format milliseconds")
        self.source_path = resolved_path
        self._device_path = device_path
        self._paused = False
        self._length_ms = int(self._send(f"status {self.alias} length") or "0")

    def play(self, start_ms: Optional[int] = None) -> None:
        """Start playback from the current position or from ``start_ms``."""

        if not self.is_loaded():
            raise RuntimeError("No session audio has been loaded yet.")

        if start_ms is not None:
            bounded_start = max(0, int(start_ms))
            self._send(f"play {self.alias} from {bounded_start}")
            self._paused = False
            return

        self._send(f"play {self.alias}")
        self._paused = False

    def pause(self) -> None:
        """Pause playback while keeping the current position available."""

        if not self.is_loaded():
            return
        self._send(f"pause {self.alias}")
        self._paused = True

    def resume(self) -> None:
        """Resume playback from the paused position."""

        if not self.is_loaded():
            raise RuntimeError("No session audio has been loaded yet.")
        self._send(f"resume {self.alias}")
        self._paused = False

    def stop(self) -> None:
        """Stop playback and seek back to the start of the track."""

        if not self.is_loaded():
            return
        self._send(f"stop {self.alias}", allow_failure=True)
        self._send(f"seek {self.alias} to start", allow_failure=True)
        self._paused = False

    def close(self) -> None:
        """Release the currently loaded audio track."""

        self._close_alias_if_open()

    def is_loaded(self) -> bool:
        return self.source_path is not None

    def is_playing(self) -> bool:
        if not self.is_loaded():
            return False
        return self._send(f"status {self.alias} mode", allow_failure=True).lower() == "playing"

    def is_paused(self) -> bool:
        return self.is_loaded() and self._paused

    def get_position_ms(self) -> int:
        if not self.is_loaded():
            return 0
        value = self._send(f"status {self.alias} position", allow_failure=True).strip() or "0"
        return int(value)

    def get_length_ms(self) -> int:
        return self._length_ms

    def _prepare_device_path(self, resolved_path: Path) -> Path:
        """Return a path that the MCI API can open reliably."""

        if is_mci_safe_filename(resolved_path):
            return resolved_path
        staged_path = build_mci_staging_path(resolved_path)
        staged_path.parent.mkdir(parents=True, exist_ok=True)
        if staged_path.exists():
            staged_path.unlink()
        # MCI accepts a copied short-name WAV here, but it fails to open a
        # hard link that targets the same file contents.
        shutil.copyfile(resolved_path, staged_path)
        self._staged_path = staged_path
        return staged_path

    def _open_device(self, device_path: Path) -> None:
        """Open ``device_path`` using the most reliable MCI device type."""

        attempts = [f'open "{device_path}" type waveaudio alias {self.alias}']
        if device_path.suffix.lower() != ".wav":
            attempts.insert(0, f'open "{device_path}" alias {self.alias}')
        elif f'open "{device_path}" alias {self.alias}' not in attempts:
            attempts.append(f'open "{device_path}" alias {self.alias}')

        last_error: Optional[RuntimeError] = None
        for command in attempts:
            try:
                self._send(command)
                return
            except RuntimeError as exc:
                last_error = exc

        assert last_error is not None
        raise last_error

    def _cleanup_staged_path(self) -> None:
        """Remove any temporary staging file created for MCI playback."""

        if self._staged_path is not None:
            try:
                self._staged_path.unlink()
            except OSError:
                pass
        self._staged_path = None


def create_audio_player() -> UnsupportedAudioPlayer | WindowsMciAudioPlayer:
    """Return the best audio player supported by the current platform."""

    if sys.platform.startswith("win"):
        return WindowsMciAudioPlayer()
    return UnsupportedAudioPlayer()
