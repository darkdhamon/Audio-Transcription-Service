"""Tests for transcript browsing and session media helpers."""

from pathlib import Path
import json
import os
import sys
import types


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from domain.session_media import (
    CombinedSessionAudioBuilder,
    SessionMediaCatalog,
    TranscriptBundle,
    TranscriptBundleMetadata,
    TranscriptSegment,
    clip_transcript_segments_for_audio,
    choose_preferred_transcript_bundle,
    format_ffmpeg_seconds,
    transcript_output_base_name,
)


def create_audio_file(path: Path, name: str) -> Path:
    """Create a small placeholder audio file."""

    audio_path = path / name
    audio_path.write_bytes(b"RIFF")
    return audio_path


def test_session_media_catalog_lists_transcripts_in_recency_order(tmp_path: Path) -> None:
    """Newest transcript bundles should be listed first for the GUI picker."""

    session_path = tmp_path / "session"
    transcript_dir = session_path / "transcript"
    transcript_dir.mkdir(parents=True)

    older = transcript_dir / "OlderTranscript.json"
    newer = transcript_dir / "NewerTranscript.json"
    older.write_text("[]", encoding="utf-8")
    newer.write_text("[]", encoding="utf-8")
    os.utime(older, (100, 100))
    os.utime(newer, (200, 200))

    bundles = SessionMediaCatalog().list_transcript_bundles(session_path)

    assert [bundle.display_name for bundle in bundles] == ["NewerTranscript", "OlderTranscript"]


def test_session_media_catalog_loads_normalized_segments(tmp_path: Path) -> None:
    """Structured transcript rows should be loaded into typed segment objects."""

    session_path = tmp_path / "session"
    transcript_dir = session_path / "transcript"
    transcript_dir.mkdir(parents=True)
    transcript_json = transcript_dir / "CampaignTranscript.json"
    transcript_json.write_text(
        json.dumps(
            [
                {"speaker": "GM", "start": 4.0, "end": 5.0, "text": "Second"},
                {"speaker": "Hero", "start": 1.0, "end": 2.0, "text": " First "},
            ]
        ),
        encoding="utf-8",
    )

    bundle = SessionMediaCatalog().list_transcript_bundles(session_path)[0]
    segments = SessionMediaCatalog().load_segments(bundle)

    assert [(segment.speaker, segment.start, segment.text) for segment in segments] == [
        ("Hero", 1.0, "First"),
        ("GM", 4.0, "Second"),
    ]


def test_session_media_catalog_writes_and_loads_metadata(tmp_path: Path) -> None:
    """Transcript bundle metadata should preserve speaker-to-file mappings."""

    session_path = tmp_path / "session"
    transcript_dir = session_path / "transcript"
    transcript_dir.mkdir(parents=True)
    create_audio_file(session_path, "playback_alpha_001_2024-01-01_10-00-00.wav")
    create_audio_file(session_path, "playback_beta_002_2024-01-01_10-00-05.wav")
    transcript_json = transcript_dir / "CampaignTranscript.json"
    transcript_json.write_text("[]", encoding="utf-8")

    catalog = SessionMediaCatalog()
    metadata_path = catalog.write_bundle_metadata(
        session_path=session_path,
        transcript_json_path=transcript_json,
        speakers={
            "playback_alpha_001_2024-01-01_10-00-00.wav": "Alpha",
            "playback_beta_002_2024-01-01_10-00-05.wav": "Beta",
        },
    )

    metadata = catalog.load_metadata(TranscriptBundle(json_path=transcript_json))

    assert metadata_path.name == "CampaignTranscript.session-media.json"
    assert [(track.file_name, track.speaker_label) for track in metadata.source_tracks] == [
        ("playback_alpha_001_2024-01-01_10-00-00.wav", "Alpha"),
        ("playback_beta_002_2024-01-01_10-00-05.wav", "Beta"),
    ]


def test_session_media_catalog_detects_legacy_offset_metadata(tmp_path: Path) -> None:
    """Legacy transcript metadata should be detectable for sync warnings."""

    session_path = tmp_path / "session"
    transcript_dir = session_path / "transcript"
    transcript_dir.mkdir(parents=True)
    transcript_json = transcript_dir / "CampaignTranscript.json"
    transcript_json.write_text("[]", encoding="utf-8")
    metadata_path = transcript_dir / "CampaignTranscript.session-media.json"
    metadata_path.write_text(
        json.dumps(
            {
                "version": 1,
                "use_filename_offsets": True,
                "source_tracks": [
                    {
                        "file_name": "playback_beta_002_2024-01-01_10-00-05.wav",
                        "speaker_label": "Beta",
                        "offset_seconds": 5.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    catalog = SessionMediaCatalog()

    assert catalog.metadata_uses_legacy_offsets(TranscriptBundle(json_path=transcript_json)) is True


def test_choose_preferred_transcript_bundle_matches_campaign_name(tmp_path: Path) -> None:
    """The GUI should prefer the transcript matching the active campaign."""

    session_path = tmp_path / "session"
    transcript_dir = session_path / "transcript"
    transcript_dir.mkdir(parents=True)
    (transcript_dir / "AlphaTranscript.json").write_text("[]", encoding="utf-8")
    (transcript_dir / "BetaTranscript.json").write_text("[]", encoding="utf-8")

    bundles = SessionMediaCatalog().list_transcript_bundles(session_path)

    preferred = choose_preferred_transcript_bundle(
        bundles,
        transcript_output_base_name("Beta"),
    )

    assert preferred is not None
    assert preferred.display_name == "BetaTranscript"


def test_combined_session_audio_builder_build_command_uses_raw_track_mix(tmp_path: Path) -> None:
    """Mix commands should use the raw TeamSpeak tracks without filename delays."""

    audio_one = tmp_path / "speaker_one.wav"
    audio_two = tmp_path / "speaker_two.wav"
    audio_one.write_bytes(b"RIFF")
    audio_two.write_bytes(b"RIFF")

    builder = CombinedSessionAudioBuilder()
    command = builder.build_command(
        [audio_one, audio_two],
        output_path=tmp_path / "mix.wav",
    )

    assert command[0] == "ffmpeg"
    assert "adelay=" not in command[command.index("-filter_complex") + 1]
    assert "amix=inputs=2" in command[command.index("-filter_complex") + 1]
    assert command[-1] == str(tmp_path / "mix.wav")


def test_clip_transcript_segments_for_audio_removes_overlap() -> None:
    """Synced audio segments should end before the next transcript line starts."""

    clipped_segments = clip_transcript_segments_for_audio(
        [
            TranscriptSegment(start=2.0, end=5.0, speaker="Alpha", text="hello"),
            TranscriptSegment(start=4.0, end=6.0, speaker="Beta", text="there"),
        ]
    )

    assert [(segment.start, segment.end, segment.speaker) for segment in clipped_segments] == [
        (2.0, 4.0, "Alpha"),
        (4.0, 6.0, "Beta"),
    ]


def test_combined_session_audio_builder_builds_transcript_sync_command(tmp_path: Path) -> None:
    """Transcript-guided audio should trim the raw TeamSpeak tracks directly."""

    session_path = tmp_path / "session"
    session_path.mkdir()
    create_audio_file(session_path, "alpha.wav")
    create_audio_file(session_path, "beta.wav")

    builder = CombinedSessionAudioBuilder()
    command = builder.build_transcript_sync_command(
        session_path=session_path,
        metadata=TranscriptBundleMetadata(
            source_tracks=[
                types.SimpleNamespace(file_name="alpha.wav", speaker_label="Alpha"),
                types.SimpleNamespace(file_name="beta.wav", speaker_label="Beta"),
            ]
        ),
        segments=[
            TranscriptSegment(start=1.5, end=2.5, speaker="Alpha", text="one"),
            TranscriptSegment(start=6.0, end=7.5, speaker="Beta", text="two"),
        ],
        output_path=tmp_path / "sync.wav",
    )

    filter_graph = command[command.index("-filter_complex") + 1]
    assert "atrim=start=1.500:end=2.500" in filter_graph
    assert "atrim=start=6.000:end=7.500" in filter_graph
    assert "adelay=1500|1500" in filter_graph
    assert "adelay=6000|6000" in filter_graph
    assert command[-1] == str(tmp_path / "sync.wav")


def test_combined_session_audio_builder_runs_ffmpeg_for_session(monkeypatch, tmp_path: Path) -> None:
    """Building session audio should call ffmpeg and return the expected output path."""

    from domain import session_media

    session_path = tmp_path / "session"
    session_path.mkdir()
    create_audio_file(session_path, "playback_alpha_001.wav")

    captured_command = {}

    def fake_run(command, capture_output, text, check):
        captured_command["value"] = command
        Path(command[-1]).write_bytes(b"RIFF")
        return types.SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr(session_media.subprocess, "run", fake_run)

    output_path = CombinedSessionAudioBuilder().build_for_session(session_path)

    assert output_path == session_path / "transcript" / "session_mix.wav"
    assert captured_command["value"][0] == "ffmpeg"
    assert "-filter_complex_script" in captured_command["value"]
    assert "-filter_complex" not in captured_command["value"]


def test_build_for_transcript_bundle_uses_raw_track_mix(monkeypatch, tmp_path: Path) -> None:
    """Transcript bundle audio should be a full-track TeamSpeak mix, not trimmed segments."""

    session_path = tmp_path / "session"
    transcript_dir = session_path / "transcript"
    transcript_dir.mkdir(parents=True)
    create_audio_file(session_path, "playback_alpha_001_2024-01-01_10-00-00.wav")
    create_audio_file(session_path, "playback_beta_002_2024-01-01_10-00-05.wav")
    transcript_json = transcript_dir / "CampaignTranscript.json"
    transcript_json.write_text("[]", encoding="utf-8")

    captured = {}

    def fake_run_with_filter_script(input_paths, filter_graph, output_path, error_prefix):
        captured["input_paths"] = [Path(path) for path in input_paths]
        captured["filter_graph"] = filter_graph
        captured["output_path"] = Path(output_path)
        captured["error_prefix"] = error_prefix
        Path(output_path).write_bytes(b"RIFF")

    builder = CombinedSessionAudioBuilder()
    monkeypatch.setattr(builder, "_run_ffmpeg_with_filter_script", fake_run_with_filter_script)

    output_path = builder.build_for_session(
        session_path,
        bundle=TranscriptBundle(json_path=transcript_json),
    )

    assert output_path == transcript_dir / "CampaignTranscript.synced.wav"
    assert "adelay=" not in captured["filter_graph"]
    assert "atrim=" not in captured["filter_graph"]
    assert captured["error_prefix"] == "synced session audio"


def test_run_ffmpeg_command_reports_windows_command_length_limit(monkeypatch) -> None:
    """Windows path-length launch failures should not be reported as missing ffmpeg."""

    from domain import session_media

    def fake_run(*_args, **_kwargs):
        error = FileNotFoundError("too long")
        error.winerror = 206
        raise error

    monkeypatch.setattr(session_media.subprocess, "run", fake_run)

    builder = CombinedSessionAudioBuilder()
    try:
        builder._run_ffmpeg_command(["ffmpeg"], "synced session audio")
    except RuntimeError as exc:
        assert "too long" in str(exc).lower()
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected RuntimeError for WinError 206")


def test_format_ffmpeg_seconds_rounds_to_milliseconds() -> None:
    """FFmpeg timestamps should stay compact and stable in generated filters."""

    assert format_ffmpeg_seconds(1.23456) == "1.235"
