#!/usr/bin/env python3
"""Convenience launcher for the desktop GUI front end."""

from __future__ import annotations

from pathlib import Path
import sys

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "src"))

from gui.app import main


if __name__ == "__main__":
    main()
