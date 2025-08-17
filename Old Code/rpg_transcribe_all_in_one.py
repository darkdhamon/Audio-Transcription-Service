import argparse
import os
import json
import glob
import time
import queue
import subprocess
import re
from multiprocessing import Process, Queue, cpu_count
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timezone

# Progress bars
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    SpinnerColumn,
)
from rich.console import Console

console = Console()

# ------------- Filename timestamp parsing -------------
AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac")
TS_RE = re.compile(r"(20\d{2}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})(?:\.(\d{3,6}))?")


def parse_ts_from_name(name: str) -> Optional[float]:
    """Extract an absolute timestamp (seconds) from a filename if present."""
    m = TS_RE.search(name)
    if not m:
        return None
    date_s, time_s, frac = m.groups()
    micro = int((frac or "0").ljust(6, "0")[:6])
    dt = datetime.strptime(f"{date_s} {time_s}", "%Y-%m-%d %H-%M-%S").replace(tzinfo=timezone.utc)
    return dt.timestamp() + micro / 1_000_000.0


# ------------- Utility helpers -------------

def fmt(ts: float) -> str:
    h = int(ts // 3600)
    m = int((ts % 3600) // 60)
    s = ts % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def write_outputs(segments: List[Dict], outbase: str) -> None:
    # De-dup obvious repeats first
    segments = dedupe_consecutive(segments)
    # Then collapse same short phrase repeated in a short window
    segments = collapse_nearby_repeats(segments, window_s=12.0, max_len=80)

    # Sort and lightly merge adjacent lines by same speaker
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

    # TXT
    with open(outbase + ".txt", "w", encoding="utf-8") as f:
        for s in compact:
            f.write(f"[{fmt(s['start']).replace(',',':')[:-3]}] {s['speaker']}: {s['text']}\n")

    # SRT
    with open(outbase + ".srt", "w", encoding="utf-8") as f:
        for i, s in enumerate(compact, 1):
            f.write(
                f"{i}\n{fmt(s['start'])} --> {fmt(s['end'])}\n{s['speaker']}: {s['text']}\n\n"
            )

    # JSON
    with open(outbase + ".json", "w", encoding="utf-8") as f:
        json.dump(compact, f, ensure_ascii=False, indent=2)


def list_audio_files(folder: str) -> List[str]:
    files: List[str] = []
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith(AUDIO_EXTS):
            continue
        if f.lower().startswith("capture_"):
            # capture_* used only for baseline; not transcribed
            continue
        files.append(os.path.join(folder, f))
    return files


def get_duration_seconds(path: str) -> float:
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
    except Exception:
        return 0.0


# ---- De-dup helpers ----
import unicodedata


def _norm_text(t: str) -> str:
    t = unicodedata.normalize("NFKC", t or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t


def dedupe_consecutive(segments: List[Dict]) -> List[Dict]:
    """Remove immediately repeated identical lines from the same speaker."""
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
    """If the same speaker says the exact same short phrase within a time window, keep the first."""
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
                # skip repeat inside window
                continue
            last_seen[key] = t0
        out.append(s)
    return out


# ------------- VAD + Squelch knobs -------------

def build_vad_parameters(args: argparse.Namespace) -> Optional[dict]:
    if not args.vad:
        return None
    params: dict = {}
    if args.vad_threshold is not None:
        params["threshold"] = args.vad_threshold
    if args.min_speech_ms is not None:
        params["min_speech_duration_ms"] = args.min_speech_ms
    if args.min_silence_ms is not None:
        params["min_silence_duration_ms"] = args.min_silence_ms
    if args.speech_pad_ms is not None:
        params["speech_pad_ms"] = args.speech_pad_ms
    return params or None


def is_junk(seg: Dict, max_dur: float, junk_words: set) -> bool:
    dur = float(seg["end"]) - float(seg["start"])
    if dur > max_dur:
        return False
    txt = _norm_text(seg.get("text", ""))
    return txt in junk_words


# ------------- Worker -------------

def worker_transcribe(
    file_path: str,
    model_name: str,
    lang: str,
    beam: int,
    temperature: float,
    vad_params: Optional[dict],
    offset: float,
    cpu_threads: int,
    prog_q: Queue,
    speaker_label: str,
    squelch: bool,
    squelch_max_dur: float,
    junk_words: List[str],
    skip_existing: bool,
):
    from faster_whisper import WhisperModel  # import inside process

    part_json = file_path + ".json.part"
    if skip_existing and os.path.exists(part_json):
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

    model = WhisperModel(
        model_name, device="cpu", compute_type="int8", cpu_threads=cpu_threads or 0
    )

    transcribe_kwargs = dict(
        language=lang if model_name not in ("large", "large-v2", "large-v3") else None,
        vad_filter=bool(vad_params),
        beam_size=beam,
        temperature=temperature,
    )
    if vad_params:
        transcribe_kwargs["vad_parameters"] = vad_params

    segments, info = model.transcribe(file_path, **transcribe_kwargs)

    out: List[Dict] = []
    last_emit = 0.0
    junk = set(junk_words or [])

    for seg in segments:
        start = float(seg.start) + offset
        end = float(seg.end) + offset
        item = {"speaker": speaker_label, "start": start, "end": end, "text": seg.text.strip()}

        if squelch and is_junk(item, squelch_max_dur, junk):
            continue

        out.append(item)

        now = time.time()
        if duration > 0 and now - last_emit > 0.25:
            prog_q.put({"type": "progress", "file": file_path, "pos": max(0.0, end), "duration": duration})
            last_emit = now

    with open(part_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    prog_q.put({"type": "done", "file": file_path, "part": part_json})


# ------------- Parallel Scheduler -------------

def run_parallel(
    files: List[str],
    workers: int,
    speakers: Dict[str, str],
    offsets: Dict[str, float],
    **kw,
) -> List[str]:
    prog_q: Queue = Queue()
    active: List[Tuple[str, Process]] = []
    pending = list(files)
    finished_parts: List[str] = []

    def spawn_one(path: str) -> Process:
        label = speakers.get(os.path.basename(path).lower(), os.path.splitext(os.path.basename(path))[0])
        off = float(offsets.get(os.path.basename(path).lower(), 0.0))
        p = Process(
            target=worker_transcribe,
            args=(
                path,
                kw["model"],
                kw["lang"],
                kw["beam"],
                kw["temperature"],
                kw["vad_params"],
                off,
                kw["cpu_threads"],
                prog_q,
                label,
                kw["squelch"],
                kw["squelch_max_dur"],
                kw["junk_words"],
                kw["skip_existing"],
            ),
        )
        p.start()
        return p

    task_by_file: Dict[str, int] = {}
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.fields[filename]}"),
        BarColumn(),
        TextColumn("{task.percentage:>5.1f}%"),
        TimeElapsedColumn(),
        TextColumn("• ETA:"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        # start initial batch
        while pending and len(active) < workers:
            f = pending.pop(0)
            active.append((f, spawn_one(f)))

        alive = True
        while alive:
            try:
                msg = prog_q.get(timeout=0.5)
            except queue.Empty:
                msg = None

            if msg:
                t = msg.get("type")
                if t == "start":
                    file_short = os.path.basename(msg["file"])
                    duration = msg.get("duration", 0.0)
                    task_id = progress.add_task(
                        "transcribe", total=max(duration, 1.0), filename=file_short
                    )
                    task_by_file[msg["file"]] = task_id
                elif t == "progress":
                    file = msg["file"]
                    pos = float(msg.get("pos", 0.0))
                    task = task_by_file.get(file)
                    if task is not None:
                        progress.update(task, completed=pos)
                elif t in ("done", "skipped"):
                    if t == "done":
                        finished_parts.append(msg["part"])  # skipped has no part
                    file = msg["file"]
                    task = task_by_file.get(file)
                    if task is not None:
                        progress.update(task, advance=0)
                        progress.stop_task(task)
                    # retire process and spawn next
                    for i, (f, proc) in enumerate(active):
                        if f == file:
                            try:
                                proc.join()
                            finally:
                                active.pop(i)
                            break
                    if pending:
                        nxt = pending.pop(0)
                        active.append((nxt, spawn_one(nxt)))

            alive = bool(active) or bool(pending)

    return finished_parts


# ------------- Speaker names & offsets -------------

def derive_suggested_label(filename: str) -> str:
    base = os.path.splitext(filename)[0]
    base = re.sub(r"^playback_", "", base, flags=re.IGNORECASE)
    # strip any timestamp pattern from the remaining name
    base = TS_RE.sub("", base)
    base = re.sub(r"[_\- ]+$", "", base)  # tidy trailing separators
    return base or "speaker"


def prompt_speaker_names(files: List[str]) -> Dict[str, str]:
    console.print("\nAssign character names to files. Press Enter to accept the suggested name in [brackets].")
    mapping: Dict[str, str] = {}
    for path in files:
        base = os.path.basename(path)
        suggested = derive_suggested_label(base)
        reply = console.input(f"  {base} -> [bold][{suggested}][/bold]: ")
        name = reply.strip() or suggested
        mapping[base.lower()] = name
    console.print("")
    return mapping


def filename_offsets(files: List[str], input_dir: str, baseline: str) -> Dict[str, float]:
    # consider ALL audio files (including capture_*) to compute baseline if needed
    all_candidates: List[Tuple[str, float]] = []
    for f in sorted(os.listdir(input_dir)):
        if f.lower().endswith(AUDIO_EXTS):
            ts = parse_ts_from_name(f)
            if ts is not None:
                all_candidates.append((f, ts))
    if not all_candidates:
        return {}

    if baseline == "capture":
        base = next((ts for (f, ts) in all_candidates if f.lower().startswith("capture_")), None)
        if base is None:
            base = min(ts for _, ts in all_candidates)
    else:
        base = min(ts for _, ts in all_candidates)

    offsets: Dict[str, float] = {}
    for path in files:
        name = os.path.basename(path)
        ts = parse_ts_from_name(name)
        if ts is not None:
            offsets[name.lower()] = float(ts - base)
    return offsets


# ------------- Main -------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Parallel multitrack transcription with Rich progress bars, filename timestamp alignment, and speaker prompts."
    )
    ap.add_argument("--model", default="small")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--input", default=".")
    ap.add_argument("--out", default="rpg_transcript")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.2, help="Decoding temperature (0.0=deterministic)")
    ap.add_argument("--vad", action="store_true")
    ap.add_argument("--vad-threshold", type=float, default=None)
    ap.add_argument("--min-speech-ms", type=int, default=None)
    ap.add_argument("--min-silence-ms", type=int, default=None)
    ap.add_argument("--speech-pad-ms", type=int, default=None)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--cpu-threads", type=int, default=0)
    ap.add_argument("--only")
    ap.add_argument("--merge-parts", action="store_true")
    ap.add_argument("--no-filename-ts", action="store_true")
    ap.add_argument("--baseline", choices=["earliest", "capture"], default="earliest")
    ap.add_argument("--no-prompt", action="store_true")
    ap.add_argument("--squelch", action="store_true")
    ap.add_argument("--squelch-max-dur", type=float, default=1.2)
    ap.add_argument("--junk-words", default="you,ya,yeah,uh,huh,mm,hmm")
    ap.add_argument("--skip-existing", action="store_true")

    args = ap.parse_args()

    if args.merge_parts:
        parts = sorted(glob.glob(os.path.join(args.input, "*.json.part")))
        if not parts:
            raise SystemExit("No .json.part files found to merge.")
        merged: List[Dict] = []
        for p in parts:
            with open(p, "r", encoding="utf-8") as f:
                merged.extend(json.load(f))
        write_outputs(merged, args.out)
        console.print(f"Merged {len(parts)} part files -> {args.out}.txt / .srt / .json")
        return

    # Choose files (ignoring capture_*)
    if args.only:
        files = [os.path.join(args.input, args.only)]
        if os.path.basename(files[0]).lower().startswith("capture_"):
            raise SystemExit("'capture_' files are ignored for transcription.")
    else:
        files = list_audio_files(args.input)

    if not files:
        raise SystemExit("No audio files found (capture_* are ignored by design).")

    # Offsets: filename-based unless disabled
    offsets: Dict[str, float] = {}
    if not args.no_filename_ts:
        auto = filename_offsets(files, args.input, args.baseline)
        offsets.update(auto)
        if auto:
            console.print("[dim]Auto-offsets from filenames:[/dim]")
            for name, off in sorted(auto.items()):
                console.print(f"  {name} -> +{off:.3f}s")

    # Speaker names
    if args.no_prompt:
        speakers = {os.path.basename(p).lower(): derive_suggested_label(os.path.basename(p)) for p in files}
    else:
        speakers = prompt_speaker_names(files)

    # Workers
    if args.workers <= 0:
        workers = min(len(files), max(1, (cpu_count() or 4) // 2))
    else:
        workers = args.workers

    console.print(f"Processing {len(files)} file(s) with up to {workers} parallel worker(s)…")

    vad_params = build_vad_parameters(args)

    finished_parts = run_parallel(
        files,
        workers,
        speakers,
        offsets,
        model=args.model,
        lang=args.lang,
        beam=args.beam,
        temperature=args.temperature,
        vad_params=vad_params,
        cpu_threads=args.cpu_threads,
        squelch=args.squelch,
        squelch_max_dur=args.squelch_max_dur,
        junk_words=[w.strip().lower() for w in args.junk_words.split(',') if w.strip()],
        skip_existing=args.skip_existing,
    )

    # Merge all parts to final outputs
    merged: List[Dict] = []
    for p in sorted(glob.glob(os.path.join(args.input, "*.json.part"))):
        try:
            with open(p, "r", encoding="utf-8") as f:
                merged.extend(json.load(f))
        except Exception:
            console.print(f"[yellow]Warning: couldn't read {p}; skipping.[/yellow]")
    write_outputs(merged, args.out)
    console.print(f"\nAll done. Outputs -> {args.out}.txt / .srt / {args.out}.json")


if __name__ == "__main__":
    main()
