#!/usr/bin/env python3
"""Convenience launcher for the audio transcription CLI.

Running this script adds the ``src`` directory to ``sys.path`` so that the
CLI can be executed without manual environment setup.  This allows the
application to be started with a single double‑click or by running
``python run.py`` from the repository root.
"""

from __future__ import annotations

from pathlib import Path
import sys


def main() -> None:
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "src"))
    from cli.app import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
