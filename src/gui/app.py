from __future__ import annotations

"""Tkinter GUI front end for the audio transcription service."""

from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
import json
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
from domain.session_media import (
    CombinedSessionAudioBuilder,
    SessionMediaCatalog,
    TranscriptBundle,
    TranscriptSegment,
    choose_preferred_transcript_bundle,
    transcript_output_base_name,
)
from domain.transcription import TranscriptionOptions, TranscriptionService
from gui.audio import create_audio_player

CONFIG_FILE = "appsettings.json"
GAME_FILE = "gamesettings.json"
LAST_SESSION_FILE = "lastsession.json"
ENGINE_CHOICES = ("faster-whisper", "whisperx")
MODEL_CHOICES = ("auto", "distil-large-v3", "small", "turbo", "large-v3")
LANGUAGE_DEFAULT = "en"
UI_POLL_INTERVAL_MS = 150


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
        self.media_catalog = SessionMediaCatalog()
        self.audio_builder = CombinedSessionAudioBuilder(self.media_catalog)
        self.audio_player = create_audio_player()
        self.current_transcript_bundles: List[TranscriptBundle] = []
        self.current_transcript_bundle: Optional[TranscriptBundle] = None
        self.current_transcript_segments: List[TranscriptSegment] = []
        self.transcript_item_ids: List[str] = []
        self.active_segment_index = -1
        self.is_busy = False

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
        self.transcript_name_var = tk.StringVar()
        self.transcript_status_var = tk.StringVar(value="No transcript loaded.")
        self.audio_status_var = tk.StringVar(value=self.audio_player.availability_message)
        self.audio_position_var = tk.StringVar(value="00:00:00 / 00:00:00")
        self.pause_button_text_var = tk.StringVar(value="Pause")
        self.speaker_vars: Dict[str, tk.StringVar] = {}

        self.root.title("Audio Transcription Service")
        self.root.minsize(1180, 820)
        self.root.protocol("WM_DELETE_WINDOW", self._on_window_close)
        self._build_layout()
        self._bind_variable_updates()
        self._load_initial_state()
        self.root.after(UI_POLL_INTERVAL_MS, self._pump_events)

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
            wraplength=1120,
            justify="left",
        ).grid(row=0, column=0, sticky="w", padx=8, pady=(8, 4))
        ttk.Label(
            runtime_frame,
            textvariable=self.runtime_notes_var,
            wraplength=1120,
            justify="left",
        ).grid(row=1, column=0, sticky="w", padx=8, pady=(0, 8))

        content_frame = ttk.Frame(main)
        content_frame.grid(row=4, column=0, sticky="nsew")
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=2)
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

        details_notebook = ttk.Notebook(content_frame)
        details_notebook.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        progress_frame = ttk.Frame(details_notebook, padding=6)
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
        self.progress_tree.column("file", width=360, anchor="w")
        self.progress_tree.column("status", width=140, anchor="w")
        self.progress_tree.column("progress", width=180, anchor="w")
        self.progress_tree.grid(row=0, column=0, sticky="nsew")
        progress_scrollbar = ttk.Scrollbar(
            progress_frame, orient="vertical", command=self.progress_tree.yview
        )
        progress_scrollbar.grid(row=0, column=1, sticky="ns")
        self.progress_tree.configure(yscrollcommand=progress_scrollbar.set)
        details_notebook.add(progress_frame, text="Progress")

        transcript_frame = ttk.Frame(details_notebook, padding=6)
        transcript_frame.columnconfigure(1, weight=1)
        transcript_frame.rowconfigure(3, weight=1)

        ttk.Label(transcript_frame, text="Transcript").grid(row=0, column=0, sticky="w")
        self.transcript_combo = ttk.Combobox(
            transcript_frame,
            textvariable=self.transcript_name_var,
            state="readonly",
        )
        self.transcript_combo.grid(row=0, column=1, sticky="ew", padx=(6, 8))
        self.transcript_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._on_transcript_selection_changed(),
        )
        self.refresh_transcript_button = ttk.Button(
            transcript_frame,
            text="Refresh Transcript",
            command=self._refresh_transcript_view,
        )
        self.refresh_transcript_button.grid(row=0, column=2, padx=(0, 8))
        self.build_audio_button = ttk.Button(
            transcript_frame,
            text="Build Synced Audio",
            command=self._start_session_audio_build,
        )
        self.build_audio_button.grid(row=0, column=3)

        status_frame = ttk.Frame(transcript_frame)
        status_frame.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(8, 6))
        status_frame.columnconfigure(0, weight=1)
        ttk.Label(
            status_frame,
            textvariable=self.transcript_status_var,
            wraplength=760,
            justify="left",
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            status_frame,
            textvariable=self.audio_status_var,
            wraplength=760,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        playback_frame = ttk.Frame(transcript_frame)
        playback_frame.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(0, 6))
        self.play_audio_button = ttk.Button(
            playback_frame,
            text="Play",
            command=self._play_session_audio,
        )
        self.play_audio_button.grid(row=0, column=0, padx=(0, 8))
        self.pause_audio_button = ttk.Button(
            playback_frame,
            textvariable=self.pause_button_text_var,
            command=self._toggle_audio_pause,
        )
        self.pause_audio_button.grid(row=0, column=1, padx=(0, 8))
        self.stop_audio_button = ttk.Button(
            playback_frame,
            text="Stop",
            command=self._stop_audio,
        )
        self.stop_audio_button.grid(row=0, column=2, padx=(0, 16))
        ttk.Label(playback_frame, textvariable=self.audio_position_var).grid(row=0, column=3, sticky="w")

        transcript_tree_frame = ttk.Frame(transcript_frame)
        transcript_tree_frame.grid(row=3, column=0, columnspan=4, sticky="nsew")
        transcript_tree_frame.columnconfigure(0, weight=1)
        transcript_tree_frame.rowconfigure(0, weight=1)
        self.transcript_tree = ttk.Treeview(
            transcript_tree_frame,
            columns=("time", "speaker", "text"),
            show="headings",
            height=14,
        )
        self.transcript_tree.heading("time", text="Time")
        self.transcript_tree.heading("speaker", text="Speaker")
        self.transcript_tree.heading("text", text="Transcript")
        self.transcript_tree.column("time", width=110, anchor="w", stretch=False)
        self.transcript_tree.column("speaker", width=140, anchor="w", stretch=False)
        self.transcript_tree.column("text", width=540, anchor="w")
        self.transcript_tree.grid(row=0, column=0, sticky="nsew")
        self.transcript_tree.tag_configure("active", background="#d8f0d0")
        self.transcript_tree.bind("<Double-1>", lambda _event: self._play_selected_segment())
        transcript_v_scroll = ttk.Scrollbar(
            transcript_tree_frame, orient="vertical", command=self.transcript_tree.yview
        )
        transcript_v_scroll.grid(row=0, column=1, sticky="ns")
        transcript_h_scroll = ttk.Scrollbar(
            transcript_tree_frame, orient="horizontal", command=self.transcript_tree.xview
        )
        transcript_h_scroll.grid(row=1, column=0, sticky="ew")
        self.transcript_tree.configure(
            yscrollcommand=transcript_v_scroll.set,
            xscrollcommand=transcript_h_scroll.set,
        )
        details_notebook.add(transcript_frame, text="Transcript Viewer")

        log_frame = ttk.LabelFrame(main, text="Status")
        log_frame.grid(row=5, column=0, sticky="ew", pady=(8, 0))
        log_frame.columnconfigure(0, weight=1)
        ttk.Label(log_frame, textvariable=self.session_hint_var, wraplength=1120).grid(
            row=0, column=0, sticky="w", padx=8, pady=(8, 4)
        )
        ttk.Label(log_frame, textvariable=self.status_var, wraplength=1120).grid(
            row=1, column=0, sticky="w", padx=8, pady=(0, 8)
        )

        button_frame = ttk.Frame(main)
        button_frame.grid(row=6, column=0, sticky="e", pady=(8, 0))
        self.start_button = ttk.Button(
            button_frame,
            text="Start Transcription",
            command=self._start_transcription,
        )
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
        self._update_action_states()

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
            self._reset_transcript_view(
                "Choose a valid session before loading transcript outputs.",
                session_path=None,
            )
            self._update_action_states()
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
                self.session_hint_var.set("No session folders were found in the selected directory.")
                self._rebuild_speaker_editor([])
                self._reset_transcript_view(
                    "No transcript outputs are available because no session is selected.",
                    session_path=None,
                )
                self.status_var.set("Select a different folder or add session directories.")
                self._update_action_states()
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
            self._reset_transcript_view(
                "Choose a valid session before loading transcript outputs.",
                session_path=None,
            )
            self._update_action_states()
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

        self._refresh_transcript_view()
        self._update_action_states()

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

        if self.is_busy:
            messagebox.showinfo("Background Job Running", "Wait for the current background job to finish first.")
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
        self._stop_audio()
        self.status_var.set(f"Starting transcription for {session_path.name}...")
        self.is_busy = True
        self._update_action_states()

        options = self._build_transcription_options()
        workers = max(0, int(self.workers_var.get()))
        self.worker_thread = threading.Thread(
            target=self._run_transcription,
            args=(
                session_path,
                out_base,
                profile_name,
                profile,
                speakers,
                options,
                workers,
            ),
            daemon=True,
        )
        self.worker_thread.start()

    def _start_session_audio_build(self) -> None:
        """Build transcript-synced session audio in the background."""

        if self.is_busy:
            messagebox.showinfo("Background Job Running", "Wait for the current background job to finish first.")
            return

        session_path = self._get_selected_session_path()
        if session_path is None or not session_path.is_dir():
            messagebox.showerror("Missing Session", "Choose a valid session before building session audio.")
            return

        if not list_session_audio_files(session_path):
            messagebox.showerror("No Audio Files", "The selected session does not contain transcribable audio files.")
            return
        if self.current_transcript_bundle is None:
            messagebox.showerror(
                "Missing Transcript",
                "Load a generated transcript first so synced audio can follow the transcript timing.",
            )
            return

        self._stop_audio()
        self.status_var.set(f"Building synced session audio for {session_path.name}...")
        self.is_busy = True
        self._update_action_states()
        self.worker_thread = threading.Thread(
            target=self._run_session_audio_build,
            args=(session_path, self.current_transcript_bundle),
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
            self.media_catalog.write_bundle_metadata(
                session_path=session_path,
                transcript_json_path=out_base.with_suffix(".json"),
                speakers=speakers,
            )

            self.event_queue.put(
                {
                    "type": "completed",
                    "message": f"Transcription completed: {out_base}.txt",
                    "output_dir": str(transcript_dir_from_out_base(out_base)),
                    "preferred_transcript": out_base.name,
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

    def _run_session_audio_build(
        self,
        session_path: Path,
        bundle: TranscriptBundle,
    ) -> None:
        """Create transcript-synced session audio on a worker thread."""

        try:
            output_path = self.audio_builder.build_for_session(session_path, bundle=bundle)
            self.event_queue.put(
                {
                    "type": "audio_built",
                    "message": f"Synced session audio created: {output_path.name}",
                    "audio_path": str(output_path),
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
                self.is_busy = False
                self._refresh_transcript_view(preferred_base_name=event.get("preferred_transcript"))
                self._update_action_states()
                messagebox.showinfo("Transcription Complete", event["message"])
            elif event_type == "audio_built":
                self.status_var.set(event["message"])
                self.is_busy = False
                self._refresh_transcript_view()
                self._update_action_states()
            elif event_type == "error":
                self.status_var.set(f"Background job failed: {event['message']}")
                self.is_busy = False
                self._update_action_states()
                messagebox.showerror(
                    "Background Job Failed",
                    f"{event['message']}\n\n{event['traceback']}",
                )

        if self.worker_thread is not None and not self.worker_thread.is_alive():
            self.is_busy = False

        self._sync_transcript_to_audio()
        self._update_action_states()
        self.root.after(UI_POLL_INTERVAL_MS, self._pump_events)

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

    def _refresh_transcript_view(self, preferred_base_name: Optional[str] = None) -> None:
        """Load transcript bundles for the selected session and render the preferred one."""

        session_path = self._get_selected_session_path()
        if session_path is None or not session_path.is_dir():
            self._reset_transcript_view(
                "Choose a valid session before loading transcript outputs.",
                session_path=None,
            )
            return

        bundles = self.media_catalog.list_transcript_bundles(session_path)
        self.current_transcript_bundles = bundles
        bundle_labels = [bundle.display_name for bundle in bundles]
        self.transcript_combo["values"] = bundle_labels

        if not bundles:
            self._reset_transcript_view(
                "No transcript has been generated for this session yet.",
                session_path=session_path,
            )
            return

        preferred_bundle = choose_preferred_transcript_bundle(
            bundles,
            preferred_base_name
            or transcript_output_base_name(self.campaign_var.get().strip() or self.profile_name_var.get().strip()),
        )
        if preferred_bundle is None:
            self._reset_transcript_view(
                "No transcript has been generated for this session yet.",
                session_path=session_path,
            )
            return

        self.transcript_name_var.set(preferred_bundle.display_name)
        self._load_transcript_bundle(preferred_bundle)

    def _reset_transcript_view(self, message: str, session_path: Optional[Path]) -> None:
        """Clear transcript viewer state and show the provided placeholder message."""

        self.transcript_combo["values"] = []
        self.transcript_name_var.set("")
        self.current_transcript_bundles = []
        self.current_transcript_bundle = None
        self.current_transcript_segments = []
        self.transcript_item_ids = []
        self.transcript_tree.delete(*self.transcript_tree.get_children())
        self.transcript_status_var.set(message)
        self.audio_position_var.set("00:00:00 / 00:00:00")
        self._set_active_transcript_index(-1)
        self.audio_player.close()
        self._update_audio_status(session_path=session_path, bundle=None)

    def _on_transcript_selection_changed(self) -> None:
        """Load the transcript bundle selected in the transcript picker."""

        selected_name = self.transcript_name_var.get().strip()
        for bundle in self.current_transcript_bundles:
            if bundle.display_name == selected_name:
                self._load_transcript_bundle(bundle)
                return

    def _load_transcript_bundle(self, bundle: TranscriptBundle) -> None:
        """Load transcript rows from ``bundle`` into the transcript viewer."""

        segments = self.media_catalog.load_segments(bundle)
        self.current_transcript_bundle = bundle
        self.current_transcript_segments = segments
        self.transcript_name_var.set(bundle.display_name)
        self._populate_transcript_tree(segments)

        transcript_line_label = "line" if len(segments) == 1 else "lines"
        status_message = f"Loaded {len(segments)} transcript {transcript_line_label} from {bundle.json_path.name}."
        if self._transcript_uses_legacy_offsets(bundle):
            status_message += " Re-run this transcript to remove legacy filename timing offsets."
        self.transcript_status_var.set(status_message)
        self._set_active_transcript_index(-1)
        self.audio_position_var.set("00:00:00 / 00:00:00")
        expected_audio_path = bundle.combined_audio_path.resolve() if bundle.combined_audio_path.exists() else None
        if self.audio_player.source_path != expected_audio_path:
            self.audio_player.close()
        self._update_audio_status(session_path=self._get_selected_session_path(), bundle=bundle)

    def _populate_transcript_tree(self, segments: List[TranscriptSegment]) -> None:
        """Render transcript rows into the tree view."""

        self.transcript_tree.delete(*self.transcript_tree.get_children())
        self.transcript_item_ids = []

        for segment in segments:
            item_id = self.transcript_tree.insert(
                "",
                "end",
                values=(
                    format_transcript_timestamp(segment.start),
                    segment.speaker,
                    segment.text,
                ),
            )
            self.transcript_item_ids.append(item_id)

    def _update_audio_status(
        self,
        session_path: Optional[Path],
        bundle: Optional[TranscriptBundle],
    ) -> None:
        """Refresh the audio helper text shown in the transcript viewer."""

        audio_status_lines: List[str] = [self.audio_player.availability_message]

        if session_path is None:
            self.audio_status_var.set(audio_status_lines[0])
            return

        if bundle is not None:
            audio_status_lines.append(f"Transcript JSON: {bundle.json_path.name}")
            if self._transcript_uses_legacy_offsets(bundle):
                audio_status_lines.append("Legacy offset transcript detected. Re-run transcription for best sync.")
            combined_audio_path = bundle.combined_audio_path
        else:
            combined_audio_path = self.media_catalog.combined_audio_path_for_session(session_path)

        if combined_audio_path.exists():
            audio_status_lines.append(f"Synced audio: {combined_audio_path.name}")
        else:
            audio_status_lines.append("Build synced audio to match the transcript timing.")

        self.audio_status_var.set("  ".join(audio_status_lines))

    def _update_action_states(self) -> None:
        """Enable or disable buttons to match the current application state."""

        session_selected = self._get_selected_session_path() is not None
        transcript_loaded = self.current_transcript_bundle is not None and bool(self.current_transcript_segments)
        audio_file_ready = (
            self.current_transcript_bundle is not None
            and self.current_transcript_bundle.combined_audio_path.exists()
        )
        player_loaded = self.audio_player.is_loaded()
        player_playing = self.audio_player.is_playing()
        player_paused = self.audio_player.is_paused()
        can_play_audio = audio_file_ready and self.audio_player.supported and transcript_loaded and not self.is_busy

        self.start_button.configure(state="disabled" if self.is_busy or not session_selected else "normal")
        self.refresh_transcript_button.configure(
            state="disabled" if self.is_busy or not session_selected else "normal"
        )
        self.build_audio_button.configure(
            state="disabled" if self.is_busy or not session_selected else "normal"
        )
        self.play_audio_button.configure(state="normal" if can_play_audio else "disabled")
        self.pause_audio_button.configure(
            state="normal" if player_loaded and (player_playing or player_paused) and not self.is_busy else "disabled"
        )
        self.stop_audio_button.configure(
            state="normal"
            if player_loaded and not self.is_busy and (player_playing or player_paused or self.audio_player.get_position_ms() > 0)
            else "disabled"
        )
        self.pause_button_text_var.set("Resume" if player_paused and not player_playing else "Pause")

    def _load_audio_if_needed(self, audio_path: Path) -> None:
        """Load the session audio file into the player if it is not active yet."""

        resolved_audio_path = audio_path.resolve()
        if self.audio_player.source_path != resolved_audio_path:
            self.audio_player.load(resolved_audio_path)

    def _transcript_uses_legacy_offsets(self, bundle: TranscriptBundle) -> bool:
        """Detect transcripts produced before the TeamSpeak timing fix."""

        try:
            return self.media_catalog.metadata_uses_legacy_offsets(bundle)
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
            return False

    def _play_session_audio(self) -> None:
        """Play the combined session audio and keep the transcript synced."""

        bundle = self.current_transcript_bundle
        if bundle is None:
            messagebox.showerror("Missing Transcript", "Load a generated transcript before starting playback.")
            return
        if not bundle.combined_audio_path.exists():
            messagebox.showinfo(
                "Session Audio Missing",
                "Build session audio first so the transcript can sync to a combined track.",
            )
            return

        try:
            self._load_audio_if_needed(bundle.combined_audio_path)
            selected_index = self._get_selected_transcript_index()
            start_ms: Optional[int] = None
            if selected_index is not None:
                start_ms = int(self.current_transcript_segments[selected_index].start * 1000.0)
            elif self.audio_player.is_paused():
                self.audio_player.resume()
                self.status_var.set(f"Resumed playback for {bundle.combined_audio_path.name}.")
                return
            elif self.audio_player.get_position_ms() >= max(0, self.audio_player.get_length_ms() - 250):
                start_ms = 0

            self.audio_player.play(start_ms=start_ms)
            self.status_var.set(f"Playing {bundle.combined_audio_path.name}.")
            if start_ms is not None:
                self._sync_transcript_to_audio(force_position_ms=start_ms)
        except RuntimeError as exc:
            messagebox.showerror("Audio Playback Failed", str(exc))

    def _toggle_audio_pause(self) -> None:
        """Pause or resume the active playback session."""

        if not self.audio_player.is_loaded():
            return

        try:
            if self.audio_player.is_paused() and not self.audio_player.is_playing():
                self.audio_player.resume()
                self.status_var.set("Resumed session audio.")
            else:
                self.audio_player.pause()
                self.status_var.set("Paused session audio.")
        except RuntimeError as exc:
            messagebox.showerror("Audio Playback Failed", str(exc))

    def _stop_audio(self) -> None:
        """Stop playback and clear transcript highlighting."""

        self.audio_player.stop()
        self.audio_position_var.set("00:00:00 / 00:00:00")
        self._set_active_transcript_index(-1)

    def _play_selected_segment(self) -> None:
        """Start playback from the transcript row selected by the user."""

        if self._get_selected_transcript_index() is None:
            return
        self._play_session_audio()

    def _sync_transcript_to_audio(self, force_position_ms: Optional[int] = None) -> None:
        """Update playback labels and transcript highlighting from the audio player."""

        if not self.audio_player.is_loaded() or not self.current_transcript_segments:
            return

        position_ms = force_position_ms if force_position_ms is not None else self.audio_player.get_position_ms()
        length_ms = self.audio_player.get_length_ms()
        self.audio_position_var.set(
            f"{format_playback_clock(position_ms)} / {format_playback_clock(length_ms)}"
        )

        if not self.audio_player.is_playing() and not self.audio_player.is_paused() and position_ms == 0:
            self._set_active_transcript_index(-1)
            return

        position_seconds = position_ms / 1000.0
        next_index = find_segment_index_at_position(
            self.current_transcript_segments,
            position_seconds,
            start_index=self.active_segment_index,
        )
        self._set_active_transcript_index(next_index)

    def _set_active_transcript_index(self, index: int) -> None:
        """Highlight the active transcript row and keep it visible."""

        if self.active_segment_index == index:
            return

        if 0 <= self.active_segment_index < len(self.transcript_item_ids):
            previous_item_id = self.transcript_item_ids[self.active_segment_index]
            self.transcript_tree.item(previous_item_id, tags=())

        self.active_segment_index = index

        if 0 <= index < len(self.transcript_item_ids):
            current_item_id = self.transcript_item_ids[index]
            self.transcript_tree.item(current_item_id, tags=("active",))
            self.transcript_tree.selection_set(current_item_id)
            self.transcript_tree.focus(current_item_id)
            self.transcript_tree.see(current_item_id)
        else:
            self.transcript_tree.selection_remove(self.transcript_tree.selection())

    def _get_selected_transcript_index(self) -> Optional[int]:
        """Return the currently selected transcript row index, if any."""

        selection = self.transcript_tree.selection()
        if not selection:
            return None
        try:
            return self.transcript_item_ids.index(selection[0])
        except ValueError:
            return None

    def _on_window_close(self) -> None:
        """Release audio resources before destroying the main window."""

        self.audio_player.close()
        self.root.destroy()


def format_progress(completed: float, total: float) -> str:
    """Return a compact progress string for the GUI progress table."""

    if total <= 0:
        return f"{completed:.1f}s"
    percentage = min(100.0, max(0.0, (completed / total) * 100.0))
    return f"{percentage:5.1f}% ({completed:.1f}s / {total:.1f}s)"


def format_transcript_timestamp(seconds: float) -> str:
    """Return a transcript-friendly timestamp label."""

    return format_playback_clock(int(round(max(0.0, seconds) * 1000.0)))


def format_playback_clock(milliseconds: int) -> str:
    """Return ``milliseconds`` as an ``HH:MM:SS`` label."""

    total_seconds = max(0, int(milliseconds // 1000))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def find_segment_index_at_position(
    segments: List[TranscriptSegment],
    position_seconds: float,
    start_index: int = -1,
) -> int:
    """Return the transcript segment index that best matches ``position_seconds``."""

    if not segments:
        return -1

    clamped_start = max(0, min(start_index, len(segments) - 1))
    search_order = list(range(clamped_start, len(segments))) + list(range(0, clamped_start))

    for index in search_order:
        segment = segments[index]
        if segment.start <= position_seconds <= segment.end:
            return index
        if index + 1 < len(segments):
            next_segment = segments[index + 1]
            if segment.end <= position_seconds < next_segment.start:
                return index

    if position_seconds < segments[0].start:
        return -1
    return len(segments) - 1


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
