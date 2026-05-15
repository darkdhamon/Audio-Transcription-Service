from __future__ import annotations

"""Core transcription logic for the application.

The original project started out as a single monolithic script.  This
module extracts the functionality into easily testable units that are
independent from any particular user interface.  A CLI, GUI or web
front-end can drive the functions here without modification.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from multiprocessing import Process, Queue, cpu_count
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Literal
import json
import logging
import os
import queue
import re
import subprocess
import time
import unicodedata

from domain.runtime import HardwareProfile, RuntimeSelection, resolve_runtime_selection

AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac")
TS_RE = re.compile(r"(20\d{2}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})(?:\.(\d{3,6}))?")

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionOptions:
    """Options that control the transcription process."""

    model: str = "auto"
    lang: str = "en"
    beam: int = 5
    temperature: float = 0.2
    vad: bool = True
    vad_threshold: Optional[float] = None
    min_speech_ms: Optional[int] = None
    min_silence_ms: Optional[int] = None
    speech_pad_ms: Optional[int] = None
    cpu_threads: int = 0
    squelch: bool = False
    squelch_max_dur: float = 1.2
    junk_words: Optional[List[str]] = None
    skip_existing: bool = False
    device: Literal["auto", "cpu", "cuda"] = "auto"
    compute_type: Literal["auto", "int8", "int8_float16", "float16", "float32"] = "auto"
    engine: Literal["faster-whisper", "whisperx"] = "faster-whisper"
    # These fields are filled in once runtime detection selects the most
    # compatible execution path for the current machine.
    resolved_model: Optional[str] = None
    resolved_device: Optional[Literal["cpu", "cuda"]] = None
    resolved_compute_type: Optional[str] = None
    runtime_notes: List[str] = field(default_factory=list)


def parse_ts_from_name(name: str) -> Optional[float]:
    """Extract an absolute timestamp (seconds) from a filename if present."""

    m = TS_RE.search(name)
    if not m:
        return None
    date_s, time_s, frac = m.groups()
    micro = int((frac or "0").ljust(6, "0")[:6])
    dt = datetime.strptime(f"{date_s} {time_s}", "%Y-%m-%d %H-%M-%S").replace(
        tzinfo=timezone.utc
    )
    return dt.timestamp() + micro / 1_000_000.0


def fmt(ts: float) -> str:
    h = int(ts // 3600)
    m = int((ts % 3600) // 60)
    s = ts % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def write_outputs(segments: List[Dict], outbase: str) -> None:
    segments = dedupe_consecutive(segments)
    segments = collapse_nearby_repeats(segments, window_s=12.0, max_len=80)

    segments.sort(key=lambda x: (x["start"], x["end"]))
    compact: List[Dict] = []
    for seg in segments:
        if (
            compact
            and seg["speaker"] == compact[-1]["speaker"]
            and seg["start"] - compact[-1]["end"] < 0.5
        ):
            compact[-1]["end"] = max(compact[-1]["end"], seg["end"])
            compact[-1]["text"] += " " + seg["text"]
        else:
            compact.append(seg)

    with open(outbase + ".txt", "w", encoding="utf-8") as f:
        for s in compact:
            f.write(f"[{fmt(s['start']).replace(',',':')[:-3]}] {s['speaker']}: {s['text']}\n")

    with open(outbase + ".srt", "w", encoding="utf-8") as f:
        for i, s in enumerate(compact, 1):
            f.write(
                f"{i}\n{fmt(s['start'])} --> {fmt(s['end'])}\n{s['speaker']}: {s['text']}\n\n"
            )

    with open(outbase + ".json", "w", encoding="utf-8") as f:
        json.dump(compact, f, ensure_ascii=False, indent=2)


def list_audio_files(folder: str) -> List[str]:
    files: List[str] = []
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith(AUDIO_EXTS):
            continue
        if f.lower().startswith("capture_"):
            continue
        files.append(os.path.join(folder, f))
    return files


def get_duration_seconds(path: str) -> float:
    """Return the duration of ``path`` in seconds using ``ffprobe``.

    A warning is logged and ``0.0`` is returned if ``ffprobe`` is missing or
    fails to probe the file.
    """

    try:
        res = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nw=1:nk=1",
                path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(res.stdout.strip())
    except FileNotFoundError:
        logger.warning("ffprobe is not installed or not found in PATH")
    except subprocess.CalledProcessError as exc:
        logger.warning("ffprobe failed for %s: %s", path, exc)
    except Exception as exc:
        logger.warning("unexpected error running ffprobe on %s: %s", path, exc)
    return 0.0


def _norm_text(t: str) -> str:
    t = unicodedata.normalize("NFKC", t or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t


def dedupe_consecutive(segments: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    last_key: Optional[Tuple[str, str]] = None
    for s in segments:
        key = (s.get("speaker"), _norm_text(s.get("text", "")))
        if key == last_key:
            continue
        out.append(s)
        last_key = key
    return out


def collapse_nearby_repeats(
    segments: List[Dict], window_s: float = 12.0, max_len: int = 80
) -> List[Dict]:
    last_seen: Dict[Tuple[str, str], float] = {}
    out: List[Dict] = []
    for s in segments:
        spk = s.get("speaker")
        txt_norm = _norm_text(s.get("text", ""))
        if len(txt_norm) <= max_len:
            key = (spk, txt_norm)
            t0 = float(s.get("start", 0.0))
            prev = last_seen.get(key)
            if prev is not None and (t0 - prev) <= window_s:
                continue
            last_seen[key] = t0
        out.append(s)
    return out


def build_vad_parameters(opts: TranscriptionOptions) -> Optional[dict]:
    if not opts.vad:
        return None
    params: dict = {}
    if opts.vad_threshold is not None:
        params["threshold"] = opts.vad_threshold
    if opts.min_speech_ms is not None:
        params["min_speech_duration_ms"] = opts.min_speech_ms
    if opts.min_silence_ms is not None:
        params["min_silence_duration_ms"] = opts.min_silence_ms
    if opts.speech_pad_ms is not None:
        params["speech_pad_ms"] = opts.speech_pad_ms
    # Return an empty dict when VAD is enabled without custom thresholds so
    # backends can still turn on their default voice activity detection.
    return params


def is_junk(seg: Dict, max_dur: float, junk_words: set) -> bool:
    dur = float(seg["end"]) - float(seg["start"])
    if dur > max_dur:
        return False
    txt = _norm_text(seg.get("text", ""))
    return txt in junk_words


def worker_transcribe(
    file_path: str,
    options: TranscriptionOptions,
    vad_params: Optional[dict],
    offset: float,
    cpu_threads: int,
    prog_q: Queue,
    speaker_label: str,
    transcribe_fn: Callable[[str, TranscriptionOptions, Optional[dict], float, str], List[Dict]],
) -> None:
    part_json = file_path + ".json.part"
    if options.skip_existing and os.path.exists(part_json):
        try:
            with open(part_json, "r", encoding="utf-8") as f:
                _ = json.load(f)
            prog_q.put({"type": "skipped", "file": file_path, "part": part_json})
            return
        except Exception:
            pass

    duration = get_duration_seconds(file_path)
    prog_q.put(
        {"type": "start", "file": file_path, "speaker": speaker_label, "duration": duration}
    )
    segments = transcribe_fn(
        audio_path=file_path,
        options=options,
        vad_params=vad_params,
        offset=offset,
        speaker_label=speaker_label,
    )

    out: List[Dict] = []
    last_emit = 0.0
    junk = set(options.junk_words or [])

    for seg in segments:
        if options.squelch and is_junk(seg, options.squelch_max_dur, junk):
            continue

        out.append(seg)

        now = time.time()
        if duration > 0 and now - last_emit > 0.25:
            prog_q.put(
                {
                    "type": "progress",
                    "file": file_path,
                    "pos": max(0.0, seg["end"]),
                    "duration": duration,
                }
            )
            last_emit = now

    with open(part_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    prog_q.put({"type": "done", "file": file_path, "part": part_json})


def run_parallel(
    files: List[str],
    workers: int,
    speakers: Dict[str, str],
    offsets: Dict[str, float],
    options: TranscriptionOptions,
    progress_callback: Optional[Callable[[Dict], None]] = None,
) -> List[str]:
    """Spawn worker processes to transcribe ``files`` in parallel."""

    prog_q: Queue = Queue()
    active: List[Tuple[str, Process]] = []
    pending = list(files)
    finished_parts: List[str] = []
    # Compute the VAD parameters once and reuse for all workers to avoid
    # redundant calculations in each spawned process.
    vad_params = build_vad_parameters(options)

    # Resolve the backend transcription function based on the selected engine.
    if options.engine == "whisperx":
        from domain.WhisperX.transcribe import transcribe as backend_transcribe
    elif options.engine == "faster-whisper":
        from domain.FasterWhisper.transcribe import transcribe as backend_transcribe
    else:  # pragma: no cover - defensive programming
        raise ValueError(f"Unsupported engine: {options.engine}")

    def spawn_one(path: str, vad_params: Optional[dict]) -> Process:
        """Start a worker process for ``path`` using cached VAD settings."""
        label = speakers.get(
            os.path.basename(path).lower(),
            os.path.splitext(os.path.basename(path))[0],
        )
        off = float(offsets.get(os.path.basename(path).lower(), 0.0))
        p = Process(
            target=worker_transcribe,
            args=(
                path,
                options,
                vad_params,
                off,
                options.cpu_threads,
                prog_q,
                label,
                backend_transcribe,
            ),
        )
        p.start()
        return p

    while pending and len(active) < workers:
        f = pending.pop(0)
        active.append((f, spawn_one(f, vad_params)))

    alive = True
    while alive:
        try:
            msg = prog_q.get(timeout=0.5)
        except queue.Empty:
            msg = None

        if msg and progress_callback:
            progress_callback(msg)

        if msg and msg.get("type") in ("done", "skipped"):
            if msg["type"] in ("done", "skipped"):
                finished_parts.append(msg["part"])
            file = msg["file"]
            for i, (f, proc) in enumerate(active):
                if f == file:
                    try:
                        proc.join()
                    finally:
                        active.pop(i)
                    break
            if pending:
                nxt = pending.pop(0)
                active.append((nxt, spawn_one(nxt, vad_params)))

        alive = bool(active) or bool(pending)

    return finished_parts


def derive_suggested_label(filename: str) -> str:
    """Derive a human friendly speaker label from an audio ``filename``.

    The recording files follow the convention
    ``playback_<display name>_<user id>_<timestamp>.wav``.  Character name
    suggestions should therefore be based on the TeamSpeak display name and
    must ignore both the numeric user id and timestamp portion.

    Parameters
    ----------
    filename:
        Name of the audio file including extension.

    Returns
    -------
    str
        Suggested label derived from the TeamSpeak display name or
        ``"speaker"`` if no name could be parsed.
    """

    # Strip the file extension and known prefix
    base = os.path.splitext(filename)[0]
    base = re.sub(r"^playback_", "", base, flags=re.IGNORECASE)

    # Remove timestamp (e.g. 2024-01-01_12-00-00.000)
    base = TS_RE.sub("", base)

    # Clean up trailing separators left behind after removing the timestamp
    base = re.sub(r"[_\- ]+$", "", base)

    # Drop the trailing user id if present.  The id is numeric and separated
    # from the display name by an underscore.
    base = re.sub(r"_[0-9]+$", "", base)

    # Final cleanup in case stripping the user id introduced a trailing
    # separator again.
    base = re.sub(r"[_\- ]+$", "", base)

    return base or "speaker"


def filename_offsets(files: List[str], input_dir: str, baseline: str) -> Dict[str, float]:
    all_candidates: List[Tuple[str, float]] = []
    for f in sorted(os.listdir(input_dir)):
        if f.lower().endswith(AUDIO_EXTS):
            ts = parse_ts_from_name(f)
            if ts is not None:
                all_candidates.append((f, ts))
    if not all_candidates:
        return {}

    if baseline == "capture":
        base_ts = next((ts for (f, ts) in all_candidates if f.lower().startswith("capture_")), None)
        if base_ts is None:
            base_ts = min(ts for _, ts in all_candidates)
    else:
        base_ts = min(ts for _, ts in all_candidates)

    offsets: Dict[str, float] = {}
    for path in files:
        name = os.path.basename(path)
        ts = parse_ts_from_name(name)
        if ts is not None:
            offsets[name.lower()] = float(ts - base_ts)
    return offsets


class TranscriptionService:
    """Facade used by front-ends to transcribe recording sessions."""

    def __init__(self, options: TranscriptionOptions) -> None:
        self.options = options
        self._runtime: Optional[RuntimeSelection] = None

    def resolve_runtime(self) -> RuntimeSelection:
        """Resolve hardware-aware execution settings for the current options."""

        if self._runtime is None:
            runtime = resolve_runtime_selection(self.options)
            self.options.resolved_model = runtime.model
            self.options.resolved_device = runtime.device
            self.options.resolved_compute_type = runtime.compute_type
            self.options.runtime_notes = list(runtime.notes)
            self._runtime = runtime
        return self._runtime

    def detect_hardware(self) -> Optional[HardwareProfile]:
        """Return the detected hardware profile after runtime resolution."""

        return self.resolve_runtime().hardware

    def describe_runtime(self) -> str:
        """Return a compact human-readable summary of the resolved runtime."""

        return self.resolve_runtime().describe()

    def validate_dependencies(self) -> None:
        """Ensure required external tools and libraries are available.

        Raises
        ------
        RuntimeError
            If ``faster-whisper`` or ``ffprobe`` is missing from the
            environment.
        """

        _ = self.resolve_runtime()

        try:  # Verify Python package is installed
            if self.options.engine == "whisperx":
                import whisperx  # type: ignore  # noqa: F401
            else:
                import faster_whisper  # type: ignore  # noqa: F401
        except ImportError as exc:
            pkg = "whisperx" if self.options.engine == "whisperx" else "faster-whisper"
            raise RuntimeError(
                f"The '{pkg}' package is required. Install it with 'pip install {pkg}'."
            ) from exc

        try:  # Verify ffprobe command is available
            subprocess.run(["ffprobe", "-version"], capture_output=True, check=True)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "ffprobe was not found. Install FFmpeg and ensure 'ffprobe' is on your PATH."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError("ffprobe is installed but failed to run properly") from exc

    def transcribe(
        self,
        input_dir: str,
        out_base: str,
        speakers: Dict[str, str],
        offsets: Optional[Dict[str, float]] = None,
        workers: int = 0,
        progress_callback: Optional[Callable[[Dict], None]] = None,
        only: Optional[str] = None,
        skip_filename_ts: bool = False,
        baseline: str = "earliest",
    ) -> None:
        # Ensure required tools and libraries exist before processing
        self.validate_dependencies()

        if only:
            files = [os.path.join(input_dir, only)]
            if os.path.basename(files[0]).lower().startswith("capture_"):
                raise ValueError("'capture_' files are ignored for transcription.")
        else:
            files = list_audio_files(input_dir)

        if not files:
            raise ValueError("No audio files found (capture_* are ignored by design).")

        if workers <= 0:
            workers = min(len(files), max(1, (cpu_count() or 4) // 2))

        offsets = offsets or {}
        if not skip_filename_ts:
            offsets.update(filename_offsets(files, input_dir, baseline))

        finished_parts = run_parallel(
            files,
            workers,
            speakers,
            offsets,
            self.options,
            progress_callback=progress_callback,
        )

        merged: List[Dict] = []
        for p in sorted(finished_parts):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    merged.extend(json.load(f))
            except Exception:
                pass
        write_outputs(merged, out_base)

    def merge_parts(self, input_dir: str, out_base: str) -> int:
        parts = sorted(Path(input_dir).glob("*.json.part"))
        if not parts:
            raise FileNotFoundError("No .json.part files found to merge.")
        merged: List[Dict] = []
        for p in parts:
            with p.open("r", encoding="utf-8") as f:
                merged.extend(json.load(f))
        write_outputs(merged, out_base)
        return len(parts)
