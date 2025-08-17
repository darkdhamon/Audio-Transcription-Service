"""Tests for the command line interface utilities."""

from pathlib import Path
import sys
import argparse

# Ensure the ``src`` package is importable when running tests directly.
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from cli.app import build_options


def test_build_options_engine() -> None:
    """Passing ``--engine`` should populate ``TranscriptionOptions.engine``."""

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
        squelch=False,
        squelch_max_dur=1.2,
        junk_words="",
        skip_existing=False,
        engine="whisperx",
    )

    opts = build_options(args)
    assert opts.engine == "whisperx"

