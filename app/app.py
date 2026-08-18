"""Entry point for the Streamlit testing dashboard."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.dashboard import render_dashboard  # noqa: E402


if __name__ == "__main__":
    render_dashboard()
