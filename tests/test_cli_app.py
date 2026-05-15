"""Tests for the command line interface utilities."""

from pathlib import Path
import sys
import argparse

from rich.console import Console

# Ensure the ``src`` package is importable when running tests directly.
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from cli.app import build_options, format_runtime_summary


def test_build_options_engine() -> None:
    """CLI arguments should populate runtime-related transcription options."""

    # Construct a minimal namespace resembling parsed CLI arguments.
    args = argparse.Namespace(
        model="small",
        lang="en",
        beam=5,
        temperature=0.2,
        vad=False,
        vad_threshold=None,
        min_speech_ms=None,
        min_silence_ms=None,
        speech_pad_ms=None,
        cpu_threads=0,
        device="cuda",
        compute_type="float16",
        squelch=False,
        squelch_max_dur=1.2,
        junk_words="",
        skip_existing=False,
        engine="whisperx",
    )

    opts = build_options(args)
    assert opts.engine == "whisperx"
    assert opts.device == "cuda"
    assert opts.compute_type == "float16"


def test_format_runtime_summary_renders_plain_text() -> None:
    """The runtime summary should render literal values in terminal output."""

    console = Console(record=True, width=200, color_system=None)
    console.print(
        format_runtime_summary(
            "engine=faster-whisper, model=small, device=cpu, compute_type=int8"
        ),
        highlight=False,
    )

    assert (
        console.export_text().strip()
        == "Runtime: engine=faster-whisper, model=small, device=cpu, compute_type=int8"
    )

