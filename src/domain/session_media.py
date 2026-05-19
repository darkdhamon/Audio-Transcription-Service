from __future__ import annotations

"""Helpers for browsing transcript outputs and building session media assets."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import json
import subprocess
import tempfile

from domain.launcher import list_session_audio_files

TRANSCRIPT_DIR_NAME = "transcript"
SESSION_AUDIO_FILENAME = "session_mix.wav"
SESSION_MEDIA_METADATA_SUFFIX = ".session-media.json"
SYNCED_AUDIO_SUFFIX = ".synced.wav"


@dataclass(frozen=True)
class TranscriptSegment:
    """Represents one timed transcript entry ready for GUI display."""

    start: float
    end: float
    speaker: str
    text: str

    @classmethod
    def from_dict(cls, raw_segment: dict) -> "TranscriptSegment":
        """Create a normalized transcript segment from persisted JSON data."""

        start = float(raw_segment.get("start", 0.0))
        end = max(start, float(raw_segment.get("end", start)))
        speaker = str(raw_segment.get("speaker", "Unknown")).strip() or "Unknown"
        text = str(raw_segment.get("text", "")).strip()
        return cls(start=start, end=end, speaker=speaker, text=text)


@dataclass(frozen=True)
class TranscriptBundle:
    """Groups the files that belong to a saved transcript export."""

    json_path: Path

    @property
    def base_name(self) -> str:
        return self.json_path.stem

    @property
    def display_name(self) -> str:
        return self.base_name

    @property
    def transcript_dir(self) -> Path:
        return self.json_path.parent

    @property
    def metadata_path(self) -> Path:
        return self.json_path.with_name(f"{self.json_path.stem}{SESSION_MEDIA_METADATA_SUFFIX}")

    @property
    def combined_audio_path(self) -> Path:
        return self.json_path.with_name(f"{self.json_path.stem}{SYNCED_AUDIO_SUFFIX}")

    @property
    def txt_path(self) -> Path:
        return self.json_path.with_suffix(".txt")

    @property
    def srt_path(self) -> Path:
        return self.json_path.with_suffix(".srt")


@dataclass(frozen=True)
class TranscriptSourceTrack:
    """Describes how one speaker label maps back to the original session audio."""

    file_name: str
    speaker_label: str


@dataclass(frozen=True)
class TranscriptBundleMetadata:
    """Metadata required to rebuild transcript-synced session audio."""

    source_tracks: List[TranscriptSourceTrack]

    @classmethod
    def from_dict(cls, raw_data: dict) -> "TranscriptBundleMetadata":
        """Create typed metadata from persisted JSON data."""

        tracks = [
            TranscriptSourceTrack(
                file_name=str(track.get("file_name", "")),
                speaker_label=str(track.get("speaker_label", "")).strip(),
            )
            for track in raw_data.get("source_tracks", [])
            if isinstance(track, dict)
        ]
        return cls(source_tracks=tracks)


class SessionMediaCatalog:
    """Discovers transcript outputs and reads structured transcript data."""

    def __init__(
        self,
        transcript_dir_name: str = TRANSCRIPT_DIR_NAME,
        session_audio_filename: str = SESSION_AUDIO_FILENAME,
    ) -> None:
        self.transcript_dir_name = transcript_dir_name
        self.session_audio_filename = session_audio_filename

    def transcript_dir_for_session(self, session_path: Path) -> Path:
        """Return the transcript directory associated with ``session_path``."""

        return session_path / self.transcript_dir_name

    def combined_audio_path_for_session(self, session_path: Path) -> Path:
        """Return the expected mixed-audio output path for ``session_path``."""

        return self.transcript_dir_for_session(session_path) / self.session_audio_filename

    def list_transcript_bundles(self, session_path: Path) -> List[TranscriptBundle]:
        """Return saved transcript bundles for ``session_path`` ordered by recency."""

        transcript_dir = self.transcript_dir_for_session(session_path)
        if not transcript_dir.is_dir():
            return []

        json_files = sorted(
            (
                path
                for path in transcript_dir.glob("*.json")
                if not path.name.endswith(SESSION_MEDIA_METADATA_SUFFIX)
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return [TranscriptBundle(json_path=json_path) for json_path in json_files]

    def load_segments(self, bundle: TranscriptBundle) -> List[TranscriptSegment]:
        """Load structured transcript rows from the bundle's JSON output."""

        with bundle.json_path.open("r", encoding="utf-8") as handle:
            raw_segments = json.load(handle)

        if not isinstance(raw_segments, list):
            raise ValueError("Transcript JSON must contain a list of segments.")

        segments = [
            TranscriptSegment.from_dict(raw_segment)
            for raw_segment in raw_segments
            if isinstance(raw_segment, dict)
        ]
        segments.sort(key=lambda segment: (segment.start, segment.end, segment.speaker.lower(), segment.text))
        return segments

    def load_metadata(self, bundle: TranscriptBundle) -> TranscriptBundleMetadata:
        """Load transcript-to-audio mapping metadata for ``bundle``."""

        with bundle.metadata_path.open("r", encoding="utf-8") as handle:
            raw_metadata = json.load(handle)

        if not isinstance(raw_metadata, dict):
            raise ValueError("Transcript metadata must contain an object.")

        return TranscriptBundleMetadata.from_dict(raw_metadata)

    def metadata_uses_legacy_offsets(self, bundle: TranscriptBundle) -> bool:
        """Return ``True`` when ``bundle`` was generated with legacy filename offsets."""

        with bundle.metadata_path.open("r", encoding="utf-8") as handle:
            raw_metadata = json.load(handle)

        if not isinstance(raw_metadata, dict):
            return False
        if bool(raw_metadata.get("use_filename_offsets")):
            return True

        for track in raw_metadata.get("source_tracks", []):
            if not isinstance(track, dict):
                continue
            if abs(float(track.get("offset_seconds", 0.0))) > 0.001:
                return True
        return False

    def write_bundle_metadata(
        self,
        session_path: Path,
        transcript_json_path: Path,
        speakers: Dict[str, str],
    ) -> Path:
        """Persist transcript metadata that maps speakers back to audio files."""

        audio_files = list_session_audio_files(session_path)
        source_tracks = [
            TranscriptSourceTrack(
                file_name=audio_file.name,
                speaker_label=speakers.get(audio_file.name.lower(), audio_file.stem),
            )
            for audio_file in audio_files
        ]

        metadata_path = transcript_json_path.with_name(
            f"{transcript_json_path.stem}{SESSION_MEDIA_METADATA_SUFFIX}"
        )
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "version": 1,
                    "source_tracks": [
                        {
                            "file_name": track.file_name,
                            "speaker_label": track.speaker_label,
                        }
                        for track in source_tracks
                    ],
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
        return metadata_path


class CombinedSessionAudioBuilder:
    """Builds a mixed session audio track using the per-speaker recordings."""

    def __init__(
        self,
        catalog: Optional[SessionMediaCatalog] = None,
        ffmpeg_executable: str = "ffmpeg",
    ) -> None:
        self.catalog = catalog or SessionMediaCatalog()
        self.ffmpeg_executable = ffmpeg_executable

    def build_for_session(
        self,
        session_path: Path,
        output_path: Optional[Path] = None,
        bundle: Optional[TranscriptBundle] = None,
    ) -> Path:
        """Create a session audio file and return its path."""

        if bundle is not None:
            return self._build_for_transcript_bundle(session_path, bundle, output_path)

        return self._build_raw_session_mix(session_path, output_path)

    def _build_raw_session_mix(
        self,
        session_path: Path,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Mix the full session audio files into a single WAV file."""

        audio_files = list_session_audio_files(session_path)
        if not audio_files:
            raise ValueError("The selected session does not contain transcribable audio files.")

        resolved_output_path = output_path or self.catalog.combined_audio_path_for_session(session_path)
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

        self._run_ffmpeg_with_filter_script(
            input_paths=audio_files,
            filter_graph=self.build_mix_filter_graph(audio_files),
            output_path=resolved_output_path,
            error_prefix="combined session audio",
        )

        return resolved_output_path

    def _build_for_transcript_bundle(
        self,
        session_path: Path,
        bundle: TranscriptBundle,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Build session audio for a transcript bundle using the raw TeamSpeak timing."""

        audio_files = list_session_audio_files(session_path)
        if not audio_files:
            raise ValueError("The selected session does not contain transcribable audio files.")

        resolved_output_path = output_path or bundle.combined_audio_path
        self._run_ffmpeg_with_filter_script(
            input_paths=audio_files,
            filter_graph=self.build_mix_filter_graph(audio_files),
            output_path=resolved_output_path,
            error_prefix="synced session audio",
        )
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

        return resolved_output_path

    def build_command(
        self,
        audio_files: Sequence[Path],
        output_path: Path,
    ) -> List[str]:
        """Build the ``ffmpeg`` command used to create the mixed session track."""

        if not audio_files:
            raise ValueError("At least one audio file is required to build session audio.")

        command: List[str] = [self.ffmpeg_executable, "-y", "-hide_banner", "-loglevel", "error"]
        for audio_file in audio_files:
            command.extend(["-i", str(audio_file)])
        command.extend(
            [
                "-filter_complex",
                self.build_mix_filter_graph(audio_files),
                "-map",
                "[out]",
                "-ac",
                "2",
                "-c:a",
                "pcm_s16le",
                str(output_path),
            ]
        )
        return command

    def build_mix_filter_graph(
        self,
        audio_files: Sequence[Path],
    ) -> str:
        """Build the filter graph used to create the mixed session track."""

        if not audio_files:
            raise ValueError("At least one audio file is required to build session audio.")

        filter_parts: List[str] = []
        mix_inputs: List[str] = []

        for index, _audio_file in enumerate(audio_files):
            output_label = f"a{index}"
            filter_parts.append(
                f"[{index}:a]aresample=async=1:first_pts=0[{output_label}]"
            )
            mix_inputs.append(f"[{output_label}]")

        if len(audio_files) == 1:
            filter_parts.append(f"{mix_inputs[0]}anull[out]")
        else:
            filter_parts.append(
                f"{''.join(mix_inputs)}amix=inputs={len(audio_files)}:normalize=1:dropout_transition=0[out]"
            )

        return ";".join(filter_parts)

    def build_transcript_sync_command(
        self,
        session_path: Path,
        metadata: TranscriptBundleMetadata,
        segments: Sequence[TranscriptSegment],
        output_path: Path,
    ) -> List[str]:
        """Build the ``ffmpeg`` command for transcript-guided non-overlapping audio."""

        if not segments:
            raise ValueError("At least one transcript segment is required to build synced audio.")

        track_lookup = self._build_track_lookup(session_path, metadata)
        command: List[str] = [self.ffmpeg_executable, "-y", "-hide_banner", "-loglevel", "error"]
        for source_track in track_lookup.values():
            command.extend(["-i", str(source_track["path"])])
        command.extend(
            [
                "-filter_complex",
                self.build_transcript_sync_filter_graph(track_lookup, segments),
                "-map",
                "[out]",
                "-ac",
                "2",
                "-c:a",
                "pcm_s16le",
                str(output_path),
            ]
        )
        return command

    def build_transcript_sync_filter_graph(
        self,
        track_lookup: Dict[str, Dict[str, object]],
        segments: Sequence[TranscriptSegment],
    ) -> str:
        """Build the filter graph for transcript-guided non-overlapping audio."""

        filter_parts: List[str] = []
        mix_inputs: List[str] = []

        for segment_index, segment in enumerate(segments):
            source_track = track_lookup.get(segment.speaker.lower())
            if source_track is None:
                raise ValueError(
                    f"No source audio track was found for transcript speaker '{segment.speaker}'."
                )

            # TeamSpeak recordings already include leading silence, so
            # transcript timestamps line up with raw file time directly.
            source_start = max(0.0, segment.start)
            source_end = max(source_start, segment.end)
            if (source_end - source_start) < 0.01:
                continue

            delay_ms = int(round(segment.start * 1000.0))
            output_label = f"seg{segment_index}"
            filter_parts.append(
                f"[{source_track['input_index']}:a]"
                f"atrim=start={format_ffmpeg_seconds(source_start)}:end={format_ffmpeg_seconds(source_end)},"
                f"asetpts=PTS-STARTPTS,"
                f"adelay={delay_ms}|{delay_ms}"
                f"[{output_label}]"
            )
            mix_inputs.append(f"[{output_label}]")

        if not mix_inputs:
            raise ValueError("Transcript timing produced no playable audio segments.")

        if len(mix_inputs) == 1:
            filter_parts.append(f"{mix_inputs[0]}anull[out]")
        else:
            filter_parts.append(
                f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:normalize=0:dropout_transition=0[out]"
            )

        return ";".join(filter_parts)

    def _build_track_lookup(
        self,
        session_path: Path,
        metadata: TranscriptBundleMetadata,
    ) -> Dict[str, Dict[str, object]]:
        """Create a speaker-label lookup that points to the original session files."""

        track_lookup: Dict[str, Dict[str, object]] = {}
        for input_index, track in enumerate(metadata.source_tracks):
            key = track.speaker_label.strip().lower()
            if not key:
                continue
            if key in track_lookup:
                raise ValueError(
                    "Transcript-guided audio requires unique speaker labels for each source track."
                )

            file_path = session_path / track.file_name
            if not file_path.is_file():
                raise FileNotFoundError(
                    f"Transcript source audio file '{track.file_name}' could not be found in the session folder."
                )

            track_lookup[key] = {
                "input_index": input_index,
                "path": file_path,
            }
        return track_lookup

    def _run_ffmpeg_with_filter_script(
        self,
        input_paths: Sequence[Path],
        filter_graph: str,
        output_path: Path,
        error_prefix: str,
    ) -> None:
        """Run ``ffmpeg`` using a temporary filter script to avoid long command lines."""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        filter_script_path = self._write_filter_script(filter_graph)
        try:
            command = [self.ffmpeg_executable, "-y", "-hide_banner", "-loglevel", "error"]
            for input_path in input_paths:
                command.extend(["-i", str(input_path)])
            command.extend(
                [
                    "-filter_complex_script",
                    str(filter_script_path),
                    "-map",
                    "[out]",
                    "-ac",
                    "2",
                    "-c:a",
                    "pcm_s16le",
                    str(output_path),
                ]
            )
            self._run_ffmpeg_command(command, error_prefix)
        finally:
            filter_script_path.unlink(missing_ok=True)

    def _write_filter_script(self, filter_graph: str) -> Path:
        """Persist ``filter_graph`` to a temporary file for ``ffmpeg`` consumption."""

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".ffgraph",
            prefix="audio-transcription-",
            delete=False,
        ) as handle:
            handle.write(filter_graph)
            return Path(handle.name)

    def _run_ffmpeg_command(self, command: Sequence[str], error_prefix: str) -> None:
        """Execute ``ffmpeg`` and raise a focused error when the process cannot start."""

        try:
            subprocess.run(list(command), capture_output=True, text=True, check=True)
        except FileNotFoundError as exc:
            if getattr(exc, "winerror", None) == 206:
                raise RuntimeError(
                    "Windows could not start FFmpeg because the generated command line was too long."
                ) from exc
            raise RuntimeError(
                "ffmpeg was not found. Install FFmpeg and ensure 'ffmpeg' is on your PATH."
            ) from exc
        except subprocess.CalledProcessError as exc:
            details = (exc.stderr or exc.stdout or str(exc)).strip()
            raise RuntimeError(f"ffmpeg failed to build the {error_prefix}: {details}") from exc


def transcript_output_base_name(campaign_name: Optional[str]) -> Optional[str]:
    """Return the transcript output base name used by the transcription flow."""

    cleaned_name = (campaign_name or "").strip()
    if not cleaned_name:
        return None
    return f"{cleaned_name}Transcript"


def clip_transcript_segments_for_audio(
    segments: Sequence[TranscriptSegment],
    minimum_duration_seconds: float = 0.05,
) -> List[TranscriptSegment]:
    """Trim overlapping transcript segments so the synced audio stays single-speaker."""

    if not segments:
        return []

    ordered_segments = sorted(
        segments,
        key=lambda segment: (segment.start, segment.end, segment.speaker.lower(), segment.text),
    )
    clipped_segments: List[TranscriptSegment] = []
    for index, segment in enumerate(ordered_segments):
        effective_end = segment.end
        if index + 1 < len(ordered_segments):
            effective_end = min(effective_end, ordered_segments[index + 1].start)
        if (effective_end - segment.start) < minimum_duration_seconds:
            continue
        clipped_segments.append(
            TranscriptSegment(
                start=segment.start,
                end=effective_end,
                speaker=segment.speaker,
                text=segment.text,
            )
        )
    return clipped_segments


def format_ffmpeg_seconds(value: float) -> str:
    """Return a compact decimal representation suitable for ``ffmpeg`` filters."""

    return f"{max(0.0, value):.3f}"


def choose_preferred_transcript_bundle(
    bundles: Sequence[TranscriptBundle],
    preferred_base_name: Optional[str] = None,
) -> Optional[TranscriptBundle]:
    """Choose the transcript bundle that best matches the user's current context."""

    if not bundles:
        return None

    cleaned_preference = (preferred_base_name or "").strip().lower()
    if cleaned_preference:
        for bundle in bundles:
            if bundle.base_name.lower() == cleaned_preference:
                return bundle
        for bundle in bundles:
            if cleaned_preference in bundle.base_name.lower():
                return bundle

    return bundles[0]
