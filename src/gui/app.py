from __future__ import annotations

"""Tkinter GUI front end for the audio transcription service."""

from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional

from domain.config import AppSettings, GameProfile, GameSettings, LastSession
from domain.launcher import (
    SessionCatalog,
    SpeakerAssignment,
    build_speaker_assignments,
    list_session_audio_files,
)
from domain.transcription import TranscriptionOptions, TranscriptionService

CONFIG_FILE = "appsettings.json"
GAME_FILE = "gamesettings.json"
LAST_SESSION_FILE = "lastsession.json"
ENGINE_CHOICES = ("faster-whisper", "whisperx")
MODEL_CHOICES = ("auto", "distil-large-v3", "small", "turbo", "large-v3")
LANGUAGE_DEFAULT = "en"


@dataclass
class ProgressRow:
    """Tracks the visible state of a single file row in the GUI."""

    item_id: str
    total: float = 0.0
    completed: float = 0.0
    status: str = "Waiting"


class TranscriptionGuiApp:
    """Desktop GUI that drives the existing transcription service."""

    def __init__(self, root: tk.Tk, repo_root: Path) -> None:
        self.root = root
        self.repo_root = repo_root
        self.config_path = repo_root / CONFIG_FILE
        self.game_settings_path = repo_root / GAME_FILE
        self.last_session_path = repo_root / LAST_SESSION_FILE
        self.event_queue: Queue[dict] = Queue()
        self.worker_thread: Optional[threading.Thread] = None
        self.current_catalog: Optional[SessionCatalog] = None
        self.current_assignments: List[SpeakerAssignment] = []
        self.current_profiles: Dict[str, GameProfile] = {}
        self.progress_rows: Dict[str, ProgressRow] = {}

        self.recordings_path_var = tk.StringVar()
        self.session_name_var = tk.StringVar()
        self.profile_name_var = tk.StringVar()
        self.campaign_var = tk.StringVar()
        self.engine_var = tk.StringVar(value="faster-whisper")
        self.model_var = tk.StringVar(value="auto")
        self.language_var = tk.StringVar(value=LANGUAGE_DEFAULT)
        self.workers_var = tk.IntVar(value=0)
        self.runtime_summary_var = tk.StringVar(value="Runtime: not loaded yet.")
        self.runtime_notes_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Choose a recordings path or session folder to begin.")
        self.session_hint_var = tk.StringVar(value="")
        self.speaker_vars: Dict[str, tk.StringVar] = {}

        self.root.title("Audio Transcription Service")
        self.root.minsize(1100, 760)
        self._build_layout()
        self._bind_variable_updates()
        self._load_initial_state()
        self.root.after(150, self._pump_events)

    def _build_layout(self) -> None:
        """Create and arrange the main GUI widgets."""

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main = ttk.Frame(self.root, padding=12)
        main.grid(sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(3, weight=1)
        main.rowconfigure(4, weight=1)

        path_frame = ttk.LabelFrame(main, text="Recordings")
        path_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        path_frame.columnconfigure(1, weight=1)
        ttk.Label(path_frame, text="Folder").grid(row=0, column=0, padx=8, pady=8, sticky="w")
        self.path_entry = ttk.Entry(path_frame, textvariable=self.recordings_path_var)
        self.path_entry.grid(row=0, column=1, padx=(0, 8), pady=8, sticky="ew")
        ttk.Button(path_frame, text="Browse...", command=self._browse_for_path).grid(
            row=0, column=2, padx=(0, 8), pady=8
        )
        ttk.Button(path_frame, text="Refresh", command=self._refresh_catalog).grid(
            row=0, column=3, padx=(0, 8), pady=8
        )

        selection_frame = ttk.Frame(main)
        selection_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        selection_frame.columnconfigure(1, weight=1)
        selection_frame.columnconfigure(3, weight=1)
        selection_frame.columnconfigure(5, weight=1)

        ttk.Label(selection_frame, text="Session").grid(row=0, column=0, sticky="w")
        self.session_combo = ttk.Combobox(
            selection_frame,
            textvariable=self.session_name_var,
            state="readonly",
        )
        self.session_combo.grid(row=0, column=1, sticky="ew", padx=(6, 16))
        self.session_combo.bind("<<ComboboxSelected>>", lambda _event: self._on_session_changed())

        ttk.Label(selection_frame, text="Profile").grid(row=0, column=2, sticky="w")
        self.profile_combo = ttk.Combobox(
            selection_frame,
            textvariable=self.profile_name_var,
            state="normal",
        )
        self.profile_combo.grid(row=0, column=3, sticky="ew", padx=(6, 16))
        self.profile_combo.bind("<<ComboboxSelected>>", lambda _event: self._on_profile_changed())

        ttk.Label(selection_frame, text="Campaign").grid(row=0, column=4, sticky="w")
        ttk.Entry(selection_frame, textvariable=self.campaign_var).grid(
            row=0, column=5, sticky="ew", padx=(6, 0)
        )

        options_frame = ttk.LabelFrame(main, text="Options")
        options_frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        options_frame.columnconfigure(1, weight=1)
        options_frame.columnconfigure(3, weight=1)
        options_frame.columnconfigure(5, weight=1)
        options_frame.columnconfigure(7, weight=1)

        ttk.Label(options_frame, text="Engine").grid(row=0, column=0, padx=8, pady=8, sticky="w")
        self.engine_combo = ttk.Combobox(
            options_frame,
            textvariable=self.engine_var,
            values=ENGINE_CHOICES,
            state="readonly",
        )
        self.engine_combo.grid(row=0, column=1, padx=(0, 12), pady=8, sticky="ew")

        ttk.Label(options_frame, text="Model").grid(row=0, column=2, padx=(0, 8), pady=8, sticky="w")
        self.model_combo = ttk.Combobox(
            options_frame,
            textvariable=self.model_var,
            values=MODEL_CHOICES,
            state="readonly",
        )
        self.model_combo.grid(row=0, column=3, padx=(0, 12), pady=8, sticky="ew")

        ttk.Label(options_frame, text="Language").grid(row=0, column=4, padx=(0, 8), pady=8, sticky="w")
        ttk.Entry(options_frame, textvariable=self.language_var, width=8).grid(
            row=0, column=5, padx=(0, 12), pady=8, sticky="ew"
        )

        ttk.Label(options_frame, text="Workers").grid(row=0, column=6, padx=(0, 8), pady=8, sticky="w")
        ttk.Spinbox(options_frame, from_=0, to=32, textvariable=self.workers_var, width=6).grid(
            row=0, column=7, padx=(0, 8), pady=8, sticky="w"
        )

        runtime_frame = ttk.LabelFrame(main, text="Runtime Preview")
        runtime_frame.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        runtime_frame.columnconfigure(0, weight=1)
        ttk.Label(
            runtime_frame,
            textvariable=self.runtime_summary_var,
            wraplength=1040,
            justify="left",
        ).grid(row=0, column=0, sticky="w", padx=8, pady=(8, 4))
        ttk.Label(
            runtime_frame,
            textvariable=self.runtime_notes_var,
            wraplength=1040,
            justify="left",
        ).grid(row=1, column=0, sticky="w", padx=8, pady=(0, 8))

        content_frame = ttk.Frame(main)
        content_frame.grid(row=4, column=0, sticky="nsew")
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        content_frame.rowconfigure(0, weight=1)

        speakers_frame = ttk.LabelFrame(content_frame, text="Speaker Names")
        speakers_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        speakers_frame.columnconfigure(0, weight=1)
        speakers_frame.rowconfigure(0, weight=1)

        self.speakers_canvas = tk.Canvas(speakers_frame, highlightthickness=0)
        self.speakers_canvas.grid(row=0, column=0, sticky="nsew")
        speakers_scrollbar = ttk.Scrollbar(
            speakers_frame, orient="vertical", command=self.speakers_canvas.yview
        )
        speakers_scrollbar.grid(row=0, column=1, sticky="ns")
        self.speakers_canvas.configure(yscrollcommand=speakers_scrollbar.set)
        self.speakers_inner = ttk.Frame(self.speakers_canvas)
        self.speakers_inner.columnconfigure(1, weight=1)
        self.speakers_canvas.create_window((0, 0), window=self.speakers_inner, anchor="nw")
        self.speakers_inner.bind(
            "<Configure>",
            lambda _event: self.speakers_canvas.configure(scrollregion=self.speakers_canvas.bbox("all")),
        )

        progress_frame = ttk.LabelFrame(content_frame, text="Progress")
        progress_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        progress_frame.columnconfigure(0, weight=1)
        progress_frame.rowconfigure(0, weight=1)
        self.progress_tree = ttk.Treeview(
            progress_frame,
            columns=("file", "status", "progress"),
            show="headings",
            height=12,
        )
        self.progress_tree.heading("file", text="File")
        self.progress_tree.heading("status", text="Status")
        self.progress_tree.heading("progress", text="Progress")
        self.progress_tree.column("file", width=320, anchor="w")
        self.progress_tree.column("status", width=140, anchor="w")
        self.progress_tree.column("progress", width=160, anchor="w")
        self.progress_tree.grid(row=0, column=0, sticky="nsew")
        progress_scrollbar = ttk.Scrollbar(
            progress_frame, orient="vertical", command=self.progress_tree.yview
        )
        progress_scrollbar.grid(row=0, column=1, sticky="ns")
        self.progress_tree.configure(yscrollcommand=progress_scrollbar.set)

        log_frame = ttk.LabelFrame(main, text="Status")
        log_frame.grid(row=5, column=0, sticky="ew", pady=(8, 0))
        log_frame.columnconfigure(0, weight=1)
        ttk.Label(log_frame, textvariable=self.session_hint_var, wraplength=1040).grid(
            row=0, column=0, sticky="w", padx=8, pady=(8, 4)
        )
        ttk.Label(log_frame, textvariable=self.status_var, wraplength=1040).grid(
            row=1, column=0, sticky="w", padx=8, pady=(0, 8)
        )

        button_frame = ttk.Frame(main)
        button_frame.grid(row=6, column=0, sticky="e", pady=(8, 0))
        self.start_button = ttk.Button(button_frame, text="Start Transcription", command=self._start_transcription)
        self.start_button.grid(row=0, column=0, padx=(0, 8))
        ttk.Button(button_frame, text="Refresh Runtime", command=self._refresh_runtime_preview).grid(
            row=0, column=1
        )

    def _bind_variable_updates(self) -> None:
        """Refresh the runtime preview when core option fields change."""

        for variable in (self.engine_var, self.model_var, self.language_var):
            variable.trace_add("write", lambda *_args: self._refresh_runtime_preview())
        self.profile_name_var.trace_add("write", lambda *_args: self._on_profile_name_edited())

    def _load_initial_state(self) -> None:
        """Load saved recordings path, profile, and last session state."""

        self._load_profiles()
        settings = AppSettings.load(self.config_path)
        last_session = LastSession.load(self.last_session_path)
        if settings.recording_directory:
            self.recordings_path_var.set(str(settings.recording_directory))
        if last_session.game_profile:
            self.profile_name_var.set(last_session.game_profile)
        self._on_profile_changed()
        if self.recordings_path_var.get():
            self._refresh_catalog(select_last_session=True)
        self._refresh_runtime_preview()

    def _load_profiles(self) -> None:
        """Load saved game profiles into the editable combobox."""

        settings = GameSettings.load(self.game_settings_path)
        self.current_profiles = dict(sorted(settings.profiles.items()))
        self.profile_combo["values"] = list(self.current_profiles.keys())

    def _browse_for_path(self) -> None:
        """Let the user choose either a recordings directory or a session folder."""

        initial_dir = self.recordings_path_var.get() or str(self.repo_root)
        chosen = filedialog.askdirectory(initialdir=initial_dir, title="Choose recordings or session folder")
        if not chosen:
            return
        self.recordings_path_var.set(chosen)
        self._refresh_catalog(select_last_session=True)

    def _refresh_catalog(self, select_last_session: bool = False) -> None:
        """Reload session information from the currently selected path."""

        source_path = Path(self.recordings_path_var.get()).expanduser()
        if not source_path.is_dir():
            self.current_catalog = None
            self.session_combo["values"] = []
            self.session_combo.configure(state="readonly")
            self.session_name_var.set("")
            self.session_hint_var.set("")
            self.status_var.set("Choose a valid recordings directory or session folder.")
            self._rebuild_speaker_editor([])
            return

        AppSettings(recording_directory=source_path).save(self.config_path)
        self.current_catalog = SessionCatalog.discover(source_path)
        last_session = LastSession.load(self.last_session_path)

        if self.current_catalog.is_direct_session:
            direct_session = self.current_catalog.direct_session_path
            assert direct_session is not None  # Help static analysis and future readers.
            self.session_combo["values"] = [direct_session.name]
            self.session_name_var.set(direct_session.name)
            self.session_combo.configure(state="disabled")
            self.session_hint_var.set("The selected folder already looks like a single recording session.")
        else:
            sessions = self.current_catalog.sessions
            session_names = [session.name for session in sessions]
            self.session_combo["values"] = session_names
            self.session_combo.configure(state="readonly")
            if not session_names:
                self.session_name_var.set("")
                self.session_combo.configure(state="readonly")
                self.session_hint_var.set("No session folders were found in the selected directory.")
                self._rebuild_speaker_editor([])
                self.status_var.set("Select a different folder or add session directories.")
                return

            preferred_session = None
            if select_last_session:
                preferred_session = self.current_catalog.find_by_name(last_session.recording_session)
            current_name = self.session_name_var.get()
            if current_name and current_name in session_names:
                self.session_name_var.set(current_name)
            elif preferred_session is not None:
                self.session_name_var.set(preferred_session.name)
            else:
                self.session_name_var.set(session_names[0])
            self.session_hint_var.set("Choose the session folder you want to transcribe.")

        self.status_var.set("Session information refreshed.")
        self._on_session_changed()

    def _get_selected_session_path(self) -> Optional[Path]:
        """Return the currently selected session path."""

        if self.current_catalog is None:
            return None
        if self.current_catalog.is_direct_session:
            return self.current_catalog.direct_session_path
        selected_name = self.session_name_var.get()
        session = self.current_catalog.find_by_name(selected_name)
        return session.path if session is not None else None

    def _on_profile_name_edited(self) -> None:
        """Refresh campaign defaults when the typed profile name changes."""

        profile = self.current_profiles.get(self.profile_name_var.get().strip())
        if profile is not None:
            self.campaign_var.set(profile.campaign)

    def _on_profile_changed(self) -> None:
        """Refresh speaker suggestions after the active profile changes."""

        profile = self.current_profiles.get(self.profile_name_var.get().strip())
        if profile is not None:
            self.campaign_var.set(profile.campaign)
        elif not self.campaign_var.get().strip() and self.profile_name_var.get().strip():
            self.campaign_var.set(self.profile_name_var.get().strip())
        self._on_session_changed()

    def _on_session_changed(self) -> None:
        """Rebuild speaker suggestions for the newly selected session."""

        session_path = self._get_selected_session_path()
        if session_path is None or not session_path.is_dir():
            self._rebuild_speaker_editor([])
            return

        files = list_session_audio_files(session_path)
        profile = self.current_profiles.get(self.profile_name_var.get().strip())
        assignments = build_speaker_assignments(files, profile)
        self.current_assignments = assignments
        self._rebuild_speaker_editor(assignments)

        if assignments:
            self.status_var.set(f"Loaded {len(assignments)} audio file(s) from {session_path.name}.")
        else:
            self.status_var.set("No audio files were found in the selected session.")

    def _rebuild_speaker_editor(self, assignments: List[SpeakerAssignment]) -> None:
        """Render editable speaker-name rows for each audio file."""

        for child in self.speakers_inner.winfo_children():
            child.destroy()

        self.speaker_vars.clear()

        if not assignments:
            ttk.Label(
                self.speakers_inner,
                text="Select a session folder to load speaker-name suggestions.",
            ).grid(row=0, column=0, padx=8, pady=8, sticky="w")
            return

        for row_index, assignment in enumerate(assignments):
            key = assignment.file_path.name.lower()
            self.speaker_vars[key] = tk.StringVar(value=assignment.suggested_name)

            ttk.Label(
                self.speakers_inner,
                text=assignment.file_path.name,
                wraplength=320,
                justify="left",
            ).grid(row=row_index, column=0, padx=8, pady=6, sticky="w")
            ttk.Entry(
                self.speakers_inner,
                textvariable=self.speaker_vars[key],
            ).grid(row=row_index, column=1, padx=(0, 8), pady=6, sticky="ew")

    def _build_transcription_options(self) -> TranscriptionOptions:
        """Create the service options represented by the current GUI selections."""

        language = self.language_var.get().strip() or LANGUAGE_DEFAULT
        return TranscriptionOptions(
            model=self.model_var.get().strip() or "auto",
            lang=language,
            vad=True,
            device="auto",
            compute_type="auto",
            engine=self.engine_var.get().strip() or "faster-whisper",
        )

    def _refresh_runtime_preview(self) -> None:
        """Show the runtime choice that the current options would resolve to."""

        try:
            service = TranscriptionService(self._build_transcription_options())
            runtime = service.resolve_runtime()
        except Exception as exc:
            self.runtime_summary_var.set(f"Runtime: unavailable ({exc})")
            self.runtime_notes_var.set("")
            return

        self.runtime_summary_var.set(f"Runtime: {runtime.describe()}")
        self.runtime_notes_var.set("\n".join(runtime.notes) if runtime.notes else "No additional runtime notes.")

    def _start_transcription(self) -> None:
        """Validate inputs, persist selections, and launch a background run."""

        if self.worker_thread is not None and self.worker_thread.is_alive():
            messagebox.showinfo("Transcription Running", "A transcription job is already running.")
            return

        session_path = self._get_selected_session_path()
        if session_path is None or not session_path.is_dir():
            messagebox.showerror("Missing Session", "Choose a valid recordings directory or session folder first.")
            return

        profile_name = self.profile_name_var.get().strip()
        if not profile_name:
            messagebox.showerror("Missing Profile", "Enter a profile name before starting transcription.")
            return

        campaign_name = self.campaign_var.get().strip() or profile_name
        if not self.current_assignments:
            messagebox.showerror("No Audio Files", "The selected session does not contain transcribable audio files.")
            return

        profile = self.current_profiles.get(profile_name, GameProfile(campaign=campaign_name))
        profile.campaign = campaign_name

        speakers: Dict[str, str] = {}
        for assignment in self.current_assignments:
            key = assignment.file_path.name.lower()
            chosen_name = self.speaker_vars[key].get().strip() or assignment.suggested_name
            speakers[key] = chosen_name
            profile.set_player(assignment.display_name, chosen_name)

        settings = GameSettings.load(self.game_settings_path)
        settings.profiles[profile_name] = profile
        settings.save(self.game_settings_path)
        self.current_profiles = settings.profiles
        self._load_profiles()

        source_path = Path(self.recordings_path_var.get()).expanduser()
        AppSettings(recording_directory=source_path).save(self.config_path)
        last_session = LastSession(recording_session=session_path.name, game_profile=profile_name)
        last_session.save(self.last_session_path)

        transcript_dir = session_path / "transcript"
        transcript_dir.mkdir(exist_ok=True)
        out_base = transcript_dir / f"{profile.campaign}Transcript"

        self.progress_tree.delete(*self.progress_tree.get_children())
        self.progress_rows.clear()
        self.status_var.set(f"Starting transcription for {session_path.name}...")
        self.start_button.configure(state="disabled")

        options = self._build_transcription_options()
        workers = max(0, int(self.workers_var.get()))
        self.worker_thread = threading.Thread(
            target=self._run_transcription,
            args=(session_path, out_base, profile_name, profile, speakers, options, workers),
            daemon=True,
        )
        self.worker_thread.start()

    def _run_transcription(
        self,
        session_path: Path,
        out_base: Path,
        profile_name: str,
        profile: GameProfile,
        speakers: Dict[str, str],
        options: TranscriptionOptions,
        workers: int,
    ) -> None:
        """Run the transcription service on a worker thread."""

        try:
            service = TranscriptionService(options)
            runtime = service.resolve_runtime()
            self.event_queue.put(
                {
                    "type": "runtime",
                    "summary": f"Runtime: {runtime.describe()}",
                    "notes": "\n".join(runtime.notes) if runtime.notes else "No additional runtime notes.",
                }
            )

            def progress_callback(message: dict) -> None:
                self.event_queue.put({"type": "progress", "message": message})

            service.transcribe(
                str(session_path),
                str(out_base),
                speakers,
                workers=workers,
                progress_callback=progress_callback,
            )

            settings = GameSettings.load(self.game_settings_path)
            settings.profiles[profile_name] = profile
            settings.save(self.game_settings_path)

            self.event_queue.put(
                {
                    "type": "completed",
                    "message": f"Transcription completed: {out_base}.txt",
                    "output_dir": str(transcript_dir_from_out_base(out_base)),
                }
            )
        except Exception as exc:  # pragma: no cover - exercised through the GUI manually
            self.event_queue.put(
                {
                    "type": "error",
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    def _pump_events(self) -> None:
        """Drain worker-thread events and apply them on the Tk main thread."""

        while True:
            try:
                event = self.event_queue.get_nowait()
            except Empty:
                break

            event_type = event.get("type")
            if event_type == "progress":
                self._handle_progress_event(event["message"])
            elif event_type == "runtime":
                self.runtime_summary_var.set(event["summary"])
                self.runtime_notes_var.set(event["notes"])
            elif event_type == "completed":
                self.status_var.set(event["message"])
                self.start_button.configure(state="normal")
                messagebox.showinfo("Transcription Complete", event["message"])
            elif event_type == "error":
                self.status_var.set(f"Transcription failed: {event['message']}")
                self.start_button.configure(state="normal")
                messagebox.showerror(
                    "Transcription Failed",
                    f"{event['message']}\n\n{event['traceback']}",
                )

        if self.worker_thread is not None and not self.worker_thread.is_alive():
            self.start_button.configure(state="normal")
        self.root.after(150, self._pump_events)

    def _handle_progress_event(self, message: dict) -> None:
        """Update the progress table from a service progress message."""

        path = Path(message["file"])
        key = path.name.lower()
        progress_type = message.get("type")

        if progress_type == "start":
            total = float(message.get("duration", 0.0))
            item_id = self.progress_tree.insert(
                "",
                "end",
                values=(path.name, "Running", format_progress(0.0, total)),
            )
            self.progress_rows[key] = ProgressRow(item_id=item_id, total=total, completed=0.0, status="Running")
        elif progress_type == "progress":
            row = self.progress_rows.get(key)
            if row is None:
                return
            row.completed = float(message.get("pos", row.completed))
            row.total = float(message.get("duration", row.total))
            row.status = "Running"
            self.progress_tree.set(row.item_id, "file", path.name)
            self.progress_tree.set(row.item_id, "status", row.status)
            self.progress_tree.set(row.item_id, "progress", format_progress(row.completed, row.total))
        elif progress_type in {"done", "skipped"}:
            row = self.progress_rows.get(key)
            if row is None:
                total = float(message.get("duration", 0.0))
                item_id = self.progress_tree.insert(
                    "",
                    "end",
                    values=(path.name, "Done", format_progress(total, total)),
                )
                row = ProgressRow(item_id=item_id, total=total, completed=total, status="Done")
                self.progress_rows[key] = row
            row.status = "Skipped" if progress_type == "skipped" else "Done"
            row.completed = row.total or row.completed
            self.progress_tree.set(row.item_id, "file", path.name)
            self.progress_tree.set(row.item_id, "status", row.status)
            self.progress_tree.set(row.item_id, "progress", format_progress(row.completed, row.total))


def format_progress(completed: float, total: float) -> str:
    """Return a compact progress string for the GUI progress table."""

    if total <= 0:
        return f"{completed:.1f}s"
    percentage = min(100.0, max(0.0, (completed / total) * 100.0))
    return f"{percentage:5.1f}% ({completed:.1f}s / {total:.1f}s)"


def transcript_dir_from_out_base(out_base: Path) -> Path:
    """Return the transcript directory that contains ``out_base`` outputs."""

    return out_base.parent


def main() -> None:
    """Launch the Tkinter GUI front end."""

    repo_root = Path(__file__).resolve().parents[2]
    root = tk.Tk()
    _ = TranscriptionGuiApp(root, repo_root)
    root.mainloop()


if __name__ == "__main__":
    main()
