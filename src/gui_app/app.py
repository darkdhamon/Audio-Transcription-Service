from __future__ import annotations

"""Tkinter-based GUI for the audio transcription service.

This module rewrites the previous CLI experience as a desktop-style
application that guides users through selecting folders, choosing
transcription options, and monitoring progress. The GUI is intentionally
thin and delegates all transcription logic to :mod:`domain.transcription`.
"""

from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import Dict, List, Optional
import queue
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from domain.transcription import (
    TranscriptionOptions,
    TranscriptionService,
    derive_suggested_label,
    filename_offsets,
    list_audio_files,
)


@dataclass
class FileProgress:
    """Track the progress of an individual file."""

    name: str
    duration: float
    completed: bool = False


class ProgressAdapter:
    """Bridge transcription callbacks to the Tkinter event loop."""

    def __init__(self, ui_queue: "queue.Queue[tuple[str, object]]") -> None:
        self.ui_queue = ui_queue

    def handle(self, msg: Dict) -> None:
        """Forward worker updates to the UI thread."""

        self.ui_queue.put(("progress", msg))


class TranscriptionGUI:
    """Graphical front end for running audio transcriptions."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Audio Transcription Service")
        self.root.geometry("900x650")

        # Tkinter variables make it easy to bind inputs to state.
        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar(value="rpg_transcript")
        self.model_var = tk.StringVar(value="small")
        self.lang_var = tk.StringVar(value="en")
        self.engine_var = tk.StringVar(value="faster-whisper")
        self.beam_var = tk.StringVar(value="5")
        self.temp_var = tk.StringVar(value="0.2")
        self.vad_var = tk.BooleanVar(value=False)
        self.vad_threshold_var = tk.StringVar()
        self.workers_var = tk.StringVar(value="0")
        self.cpu_threads_var = tk.StringVar(value="0")
        self.skip_filename_ts_var = tk.BooleanVar(value=False)
        self.baseline_var = tk.StringVar(value="earliest")
        self.squelch_var = tk.BooleanVar(value=False)
        self.squelch_max_dur_var = tk.StringVar(value="1.2")
        self.junk_words_var = tk.StringVar(value="you,ya,yeah,uh,huh,mm,hmm")
        self.skip_existing_var = tk.BooleanVar(value=False)

        # UI plumbing
        self.ui_queue: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self.file_progress: Dict[str, FileProgress] = {}

        self._build_layout()
        self.root.after(150, self._drain_queue)

    # -------------------------- UI construction -------------------------
    def _build_layout(self) -> None:
        """Create the Tkinter layout."""

        main_frame = ttk.Frame(self.root, padding=12)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self._build_path_section(main_frame)
        self._build_options_section(main_frame)
        self._build_progress_section(main_frame)

    def _build_path_section(self, parent: tk.Widget) -> None:
        path_frame = ttk.LabelFrame(parent, text="Project Folders", padding=12)
        path_frame.pack(fill=tk.X, expand=False, pady=(0, 12))

        # Input folder selection
        ttk.Label(path_frame, text="Input folder:").grid(row=0, column=0, sticky=tk.W)
        input_entry = ttk.Entry(path_frame, textvariable=self.input_var, width=80)
        input_entry.grid(row=0, column=1, sticky=tk.W, padx=6)
        ttk.Button(path_frame, text="Browse", command=self._choose_input).grid(
            row=0, column=2, padx=4
        )

        # Output base path selection
        ttk.Label(path_frame, text="Output base:").grid(row=1, column=0, sticky=tk.W, pady=(8, 0))
        output_entry = ttk.Entry(path_frame, textvariable=self.output_var, width=80)
        output_entry.grid(row=1, column=1, sticky=tk.W, padx=6, pady=(8, 0))
        ttk.Button(path_frame, text="Browse", command=self._choose_output).grid(
            row=1, column=2, padx=4, pady=(8, 0)
        )

    def _build_options_section(self, parent: tk.Widget) -> None:
        opts = ttk.LabelFrame(parent, text="Transcription Options", padding=12)
        opts.pack(fill=tk.X, expand=False, pady=(0, 12))

        # Row 0: model, language, engine
        ttk.Label(opts, text="Model:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(opts, textvariable=self.model_var, width=18).grid(
            row=0, column=1, sticky=tk.W, padx=(4, 12)
        )

        ttk.Label(opts, text="Language:").grid(row=0, column=2, sticky=tk.W)
        ttk.Entry(opts, textvariable=self.lang_var, width=10).grid(
            row=0, column=3, sticky=tk.W, padx=(4, 12)
        )

        ttk.Label(opts, text="Engine:").grid(row=0, column=4, sticky=tk.W)
        ttk.Combobox(
            opts,
            textvariable=self.engine_var,
            values=("faster-whisper", "whisperx"),
            state="readonly",
            width=15,
        ).grid(row=0, column=5, sticky=tk.W)

        # Row 1: beam, temperature, workers
        ttk.Label(opts, text="Beam width:").grid(row=1, column=0, sticky=tk.W, pady=(8, 0))
        ttk.Entry(opts, textvariable=self.beam_var, width=8).grid(
            row=1, column=1, sticky=tk.W, padx=(4, 12), pady=(8, 0)
        )

        ttk.Label(opts, text="Temperature:").grid(row=1, column=2, sticky=tk.W, pady=(8, 0))
        ttk.Entry(opts, textvariable=self.temp_var, width=8).grid(
            row=1, column=3, sticky=tk.W, padx=(4, 12), pady=(8, 0)
        )

        ttk.Label(opts, text="Workers:").grid(row=1, column=4, sticky=tk.W, pady=(8, 0))
        ttk.Entry(opts, textvariable=self.workers_var, width=6).grid(
            row=1, column=5, sticky=tk.W, pady=(8, 0)
        )

        # Row 2: CPU threads, VAD options
        ttk.Label(opts, text="CPU threads:").grid(row=2, column=0, sticky=tk.W, pady=(8, 0))
        ttk.Entry(opts, textvariable=self.cpu_threads_var, width=8).grid(
            row=2, column=1, sticky=tk.W, padx=(4, 12), pady=(8, 0)
        )

        ttk.Checkbutton(opts, text="Enable VAD", variable=self.vad_var).grid(
            row=2, column=2, sticky=tk.W, pady=(8, 0)
        )
        ttk.Label(opts, text="Threshold:").grid(row=2, column=3, sticky=tk.W, pady=(8, 0))
        ttk.Entry(opts, textvariable=self.vad_threshold_var, width=6).grid(
            row=2, column=4, sticky=tk.W, padx=(4, 12), pady=(8, 0)
        )

        # Row 3: squelch + filename timestamp
        ttk.Checkbutton(opts, text="Squelch", variable=self.squelch_var).grid(
            row=3, column=0, sticky=tk.W, pady=(8, 0)
        )
        ttk.Label(opts, text="Max squelch (s):").grid(row=3, column=1, sticky=tk.W, pady=(8, 0))
        ttk.Entry(opts, textvariable=self.squelch_max_dur_var, width=8).grid(
            row=3, column=2, sticky=tk.W, padx=(4, 12), pady=(8, 0)
        )

        ttk.Checkbutton(
            opts,
            text="Ignore filename timestamps",
            variable=self.skip_filename_ts_var,
        ).grid(row=3, column=3, columnspan=2, sticky=tk.W, pady=(8, 0))

        ttk.Checkbutton(
            opts, text="Skip existing outputs", variable=self.skip_existing_var
        ).grid(row=3, column=5, sticky=tk.W, pady=(8, 0))

        # Row 4: junk words and baseline
        ttk.Label(opts, text="Junk words (comma-separated):").grid(
            row=4, column=0, sticky=tk.W, pady=(8, 0)
        )
        ttk.Entry(opts, textvariable=self.junk_words_var, width=40).grid(
            row=4, column=1, columnspan=3, sticky=tk.W, padx=(4, 12), pady=(8, 0)
        )

        ttk.Label(opts, text="Baseline:").grid(row=4, column=4, sticky=tk.W, pady=(8, 0))
        ttk.Combobox(
            opts,
            textvariable=self.baseline_var,
            values=("earliest", "capture"),
            state="readonly",
            width=12,
        ).grid(row=4, column=5, sticky=tk.W, pady=(8, 0))

        # Action buttons
        btn_frame = ttk.Frame(opts)
        btn_frame.grid(row=5, column=0, columnspan=6, sticky=tk.E, pady=(16, 0))
        self.start_button = ttk.Button(btn_frame, text="Start transcription", command=self.start_transcription)
        self.start_button.pack(side=tk.RIGHT)

    def _build_progress_section(self, parent: tk.Widget) -> None:
        progress = ttk.LabelFrame(parent, text="Progress", padding=12)
        progress.pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="Select an input folder to begin.")
        ttk.Label(progress, textvariable=self.status_var).pack(anchor=tk.W)

        self.progress_bar = ttk.Progressbar(progress, length=400, mode="determinate")
        self.progress_bar.pack(fill=tk.X, pady=8)

        self.log = tk.Text(progress, height=18, wrap="word", state=tk.DISABLED)
        self.log.pack(fill=tk.BOTH, expand=True)

    # ------------------------------ UI events ---------------------------
    def _choose_input(self) -> None:
        path = filedialog.askdirectory(title="Select input folder")
        if path:
            self.input_var.set(path)
            # Default output base into the selected folder if relative
            out_base = self.output_var.get().strip()
            if out_base and not Path(out_base).is_absolute():
                self.output_var.set(str(Path(path) / out_base))

    def _choose_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Select base name for transcript (without extension)",
            defaultextension=".txt",
        )
        if path:
            # Remove extension because the transcription pipeline appends it.
            self.output_var.set(str(Path(path).with_suffix("")))

    def start_transcription(self) -> None:
        """Kick off a background transcription run."""

        input_dir = Path(self.input_var.get().strip())
        if not input_dir.is_dir():
            messagebox.showerror("Invalid input", "Please select a valid input folder containing audio files.")
            return

        out_base = Path(self.output_var.get().strip())
        if not out_base.is_absolute():
            out_base = input_dir / out_base

        options = self._build_options()
        self._log(f"Input: {input_dir}")
        self._log(f"Output base: {out_base}")
        self._log(f"Model: {options.model} ({self.engine_var.get()}) | Language: {options.lang}")

        self.start_button.state(["disabled"])
        self.progress_bar.configure(value=0, maximum=1)
        self.status_var.set("Preparing files…")

        worker = Thread(
            target=self._run_worker,
            args=(input_dir, out_base, options),
            daemon=True,
        )
        worker.start()

    # ---------------------------- Worker logic --------------------------
    def _run_worker(
        self, input_dir: Path, out_base: Path, options: TranscriptionOptions
    ) -> None:
        """Run transcription in a background thread and forward updates."""

        try:
            files = [Path(p) for p in list_audio_files(str(input_dir))]
            files = [p for p in files if not p.name.lower().startswith("capture_")]
            if not files:
                self.ui_queue.put(("error", "No audio files found (capture_* are ignored)."))
                return

            offsets: Dict[str, float] = {}
            if not self.skip_filename_ts_var.get():
                offsets = filename_offsets([str(f) for f in files], str(input_dir), self.baseline_var.get())
                if offsets:
                    self.ui_queue.put(("log", f"Auto-offsets from filenames: {offsets}"))

            speakers = {p.name.lower(): derive_suggested_label(p.name) for p in files}
            self.ui_queue.put(("setup", files))

            service = TranscriptionService(options)
            service.transcribe(
                str(input_dir),
                str(out_base),
                speakers,
                offsets=offsets,
                workers=self._parse_int(self.workers_var.get(), default=0),
                progress_callback=ProgressAdapter(self.ui_queue).handle,
                skip_filename_ts=self.skip_filename_ts_var.get(),
                baseline=self.baseline_var.get(),
            )

            self.ui_queue.put(("log", f"Completed transcription -> {out_base}.txt / .srt / .json"))
        except Exception as exc:  # pylint: disable=broad-except
            self.ui_queue.put(("error", str(exc)))
        finally:
            self.ui_queue.put(("finished", None))

    # ------------------------------ Helpers -----------------------------
    def _build_options(self) -> TranscriptionOptions:
        """Create :class:`TranscriptionOptions` from UI selections."""

        return TranscriptionOptions(
            model=self.model_var.get().strip(),
            lang=self.lang_var.get().strip() or "en",
            beam=self._parse_int(self.beam_var.get(), default=5),
            temperature=self._parse_float(self.temp_var.get(), default=0.2),
            vad=self.vad_var.get(),
            vad_threshold=self._parse_optional_float(self.vad_threshold_var.get()),
            cpu_threads=self._parse_int(self.cpu_threads_var.get(), default=0),
            squelch=self.squelch_var.get(),
            squelch_max_dur=self._parse_float(self.squelch_max_dur_var.get(), default=1.2),
            junk_words=[w.strip().lower() for w in self.junk_words_var.get().split(",") if w.strip()],
            skip_existing=self.skip_existing_var.get(),
            engine=self.engine_var.get(),
        )

    def _parse_int(self, raw: str, default: int = 0) -> int:
        try:
            return int(raw)
        except (TypeError, ValueError):
            return default

    def _parse_float(self, raw: str, default: float) -> float:
        try:
            return float(raw)
        except (TypeError, ValueError):
            return default

    def _parse_optional_float(self, raw: str) -> Optional[float]:
        raw = raw.strip()
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    def _log(self, message: str) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, message + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _drain_queue(self) -> None:
        """Handle updates generated by the background worker."""

        while not self.ui_queue.empty():
            kind, payload = self.ui_queue.get()
            if kind == "log":
                self._log(str(payload))
            elif kind == "setup":
                self._prepare_progress(payload)  # type: ignore[arg-type]
            elif kind == "progress":
                self._handle_progress(payload)  # type: ignore[arg-type]
            elif kind == "error":
                messagebox.showerror("Transcription failed", str(payload))
                self._log(f"Error: {payload}")
            elif kind == "finished":
                self.status_var.set("Idle")
                self.start_button.state(["!disabled"])
        self.root.after(150, self._drain_queue)

    def _prepare_progress(self, files: List[Path]) -> None:
        """Initialize progress trackers for a new run."""

        self.file_progress = {
            f.name: FileProgress(name=f.name, duration=0.0) for f in files
        }
        self.progress_bar.configure(maximum=max(len(files), 1), value=0)
        self.status_var.set(f"Processing {len(files)} file(s)…")
        self._log(f"Processing {len(files)} file(s)…")

    def _handle_progress(self, msg: Dict) -> None:
        """Update progress bar and log based on worker callbacks."""

        msg_type = msg.get("type")
        filename = Path(msg.get("file", "")).name

        if msg_type == "start":
            duration = float(msg.get("duration", 0.0))
            self.file_progress[filename] = FileProgress(name=filename, duration=duration)
            self._log(f"Started: {filename}")
        elif msg_type == "progress":
            # Individual progress events are currently only logged for context.
            pos = float(msg.get("pos", 0.0))
            duration = self.file_progress.get(filename, FileProgress(filename, 0.0)).duration
            if duration:
                pct = (pos / duration) * 100
                self.status_var.set(f"{filename}: {pct:0.1f}%")
        elif msg_type in ("done", "skipped"):
            if filename in self.file_progress:
                self.file_progress[filename].completed = True
            completed = sum(1 for fp in self.file_progress.values() if fp.completed)
            self.progress_bar.configure(value=completed)
            self.status_var.set(f"Completed {completed}/{len(self.file_progress)} files")
            self._log(f"{msg_type.title()}: {filename}")

    def run(self) -> None:
        """Start the Tkinter main loop."""

        self.root.mainloop()


if __name__ == "__main__":
    TranscriptionGUI().run()
