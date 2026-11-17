"""The sole learning organization entry; never initializes the retired learning services."""

import streamlit as st

from src import __version__
from src.study_loop.ui import render
from src.workspace_ui import render_workspace

st.set_page_config(page_title=f"学习整理 · Nectivon v{__version__}", page_icon="📚", layout="wide")
render_workspace("pages/18_学习整理.py")
render()
