from __future__ import annotations

"""Command line interface for the transcription service.

The CLI is intentionally thin and delegates the heavy lifting to the
:mod:`domain.transcription` module.  This makes it easy to swap out the
user interface in the future (for example with a GUI) while preserving
all business logic.
"""

from pathlib import Path
import argparse
from typing import Dict, List, Optional

from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    SpinnerColumn,
)

from domain.transcription import (
    TranscriptionOptions,
    TranscriptionService,
    derive_suggested_label,
    filename_offsets,
    list_audio_files,
)
from domain.config import GameProfile

console = Console()


class RichProgressHandler:
    """Render progress information using rich's progress bars."""

    def __init__(self) -> None:
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.fields[filename]}"),
            BarColumn(),
            TextColumn("{task.percentage:>5.1f}%"),
            TimeElapsedColumn(),
            TextColumn("• ETA:"),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        )
        # Track tasks by the absolute file path they relate to for consistent updates
        self.task_by_file: Dict[Path, int] = {}

    def __enter__(self) -> "RichProgressHandler":
        self.progress.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.progress.__exit__(exc_type, exc, tb)

    def handle(self, msg: Dict) -> None:
        t = msg.get("type")
        if t == "start":
            path = Path(msg["file"])
            duration = msg.get("duration", 0.0)
            task_id = self.progress.add_task(
                "transcribe", total=max(duration, 1.0), filename=path.name
            )
            self.task_by_file[path] = task_id
        elif t == "progress":
            path = Path(msg["file"])
            pos = float(msg.get("pos", 0.0))
            task = self.task_by_file.get(path)
            if task is not None:
                self.progress.update(task, completed=pos)
        elif t in ("done", "skipped"):
            path = Path(msg["file"])
            task = self.task_by_file.get(path)
            if task is not None:
                self.progress.update(task, advance=0)
                self.progress.stop_task(task)


def prompt_speaker_names(
    files: List[Path], profile: Optional[GameProfile] = None
) -> Dict[str, str]:
    """Prompt the user to assign character names to each audio file.

    Parameters
    ----------
    files:
        List of ``Path`` objects pointing to audio recordings.
    profile:
        Optional profile storing previously used character mappings.

    Returns
    -------
    Dict[str, str]
        Mapping of lowercase filenames to chosen speaker labels.
    """

    console.print(
        "\nAssign character names to files. Press Enter to accept the suggested name in [brackets]."
    )
    mapping: Dict[str, str] = {}
    for path in files:
        base = path.name

        # Derive TeamSpeak display name from the file name. This is used for
        # lookups in the profile and as the default suggestion.
        display = derive_suggested_label(base)

        suggested: Optional[str] = None
        if profile is not None:
            suggested = profile.get_character(display)
        if not suggested:
            suggested = display

        reply = console.input(f"  {base} -> [bold][{suggested}][/bold]: ")
        name = reply.strip() or suggested
        mapping[base.lower()] = name
        if profile is not None:
            profile.set_player(display, name)
    console.print("")
    return mapping


def build_options(args: argparse.Namespace) -> TranscriptionOptions:
    return TranscriptionOptions(
        model=args.model,
        lang=args.lang,
        beam=args.beam,
        temperature=args.temperature,
        vad=args.vad,
        vad_threshold=args.vad_threshold,
        min_speech_ms=args.min_speech_ms,
        min_silence_ms=args.min_silence_ms,
        speech_pad_ms=args.speech_pad_ms,
        cpu_threads=args.cpu_threads,
        squelch=args.squelch,
        squelch_max_dur=args.squelch_max_dur,
        junk_words=[w.strip().lower() for w in args.junk_words.split(',') if w.strip()],
        skip_existing=args.skip_existing,
    )


def main(profile: Optional[GameProfile] = None) -> Dict[str, str]:
    ap = argparse.ArgumentParser(description="Parallel multitrack transcription with Rich progress bars, filename timestamp alignment, and speaker prompts.")
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

    options = build_options(args)
    service = TranscriptionService(options)

    # Use ``Path`` objects to manage filesystem locations in an OS-agnostic way
    input_dir = Path(args.input)
    out_base = Path(args.out)

    if args.merge_parts:
        count = service.merge_parts(str(input_dir), str(out_base))
        console.print(f"Merged {count} part files -> {out_base}.txt / .srt / .json")
        return

    if args.only:
        files = [input_dir / args.only]
        if files[0].name.lower().startswith("capture_"):
            raise SystemExit("'capture_' files are ignored for transcription.")
    else:
        files = [Path(p) for p in list_audio_files(str(input_dir))]

    if not files:
        raise SystemExit("No audio files found (capture_* are ignored by design).")

    offsets: Dict[str, float] = {}
    if not args.no_filename_ts:
        offsets.update(
            filename_offsets([str(f) for f in files], str(input_dir), args.baseline)
        )
        if offsets:
            console.print("[dim]Auto-offsets from filenames:[/dim]")
            for name, off in sorted(offsets.items()):
                console.print(f"  {name} -> +{off:.3f}s")

    if args.no_prompt:
        speakers = {p.name.lower(): derive_suggested_label(p.name) for p in files}
    else:
        speakers = prompt_speaker_names(files, profile)

    console.print(
        f"Processing {len(files)} file(s) with up to {args.workers or 'auto'} parallel worker(s)…"
    )

    with RichProgressHandler() as progress:
        service.transcribe(
            str(input_dir),
            str(out_base),
            speakers,
            offsets=offsets,
            workers=args.workers,
            progress_callback=progress.handle,
            only=args.only,
            skip_filename_ts=args.no_filename_ts,
            baseline=args.baseline,
        )

    console.print(
        f"\nAll done. Outputs -> {out_base}.txt / .srt / {out_base}.json"
    )
    return speakers


if __name__ == "__main__":
    main()
