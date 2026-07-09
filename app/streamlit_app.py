"""Streamlit entry point for Compute Capacity Orchestrator."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
VIEWS_DIR = APP_DIR / "views"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def inject_navigation_styles() -> None:
    """Make the top navigation look more like visible application tabs."""

    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.0rem;
        }

        h1 {
            margin-top: 0rem;
            margin-bottom: 0.1rem;
        }

        /* Streamlit top navigation container */
        div[data-testid="stNavigation"] {
            border-bottom: 1px solid rgba(49, 51, 63, 0.18);
            margin-bottom: 1.2rem;
        }

        div[data-testid="stNavigation"] a {
            padding: 0.85rem 1.25rem;
            border: 1px solid transparent;
            border-bottom: none;
            border-radius: 0.45rem 0.45rem 0 0;
            font-weight: 600;
            text-decoration: none;
        }

        div[data-testid="stNavigation"] a:hover {
            background-color: rgba(255, 75, 75, 0.06);
        }

        div[data-testid="stNavigation"] a[aria-current="page"] {
            border-color: rgba(49, 51, 63, 0.18);
            background-color: white;
            color: #31333f;
            box-shadow: 0 -2px 0 #ff4b4b inset;
        }

        /* Fallback selector for some Streamlit versions */
        nav a[aria-current="page"] {
            border: 1px solid rgba(49, 51, 63, 0.18);
            border-bottom: none;
            border-radius: 0.45rem 0.45rem 0 0;
            box-shadow: 0 -2px 0 #ff4b4b inset;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.set_page_config(
    page_title="Compute Capacity Orchestrator",
    layout="wide",
)

inject_navigation_styles()

st.title("Compute Capacity Orchestrator")
st.caption("Simulation-optimization lab for GPU capacity orchestration")

snapshot_page = st.Page(
    str(VIEWS_DIR / "snapshot_page.py"),
    title="Snapshot",
    default=True,
)


simulation_page = st.Page(
    str(VIEWS_DIR / "simulation_page.py"),
    title="Simulation",
)

time_indexed_page = st.Page(
    str(VIEWS_DIR / "time_indexed_page.py"),
    title="Time-indexed",
)

offline_page = st.Page(
    str(VIEWS_DIR / "offline_page.py"),
    title="Offline experiments",
)

page = st.navigation(
    [
        snapshot_page,
        simulation_page,
        time_indexed_page,
        offline_page,
    ],
    position="top",
)

page.run()