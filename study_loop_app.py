"""Isolated post-exam UI entry. Starting it does not initialize the legacy application DB."""

import streamlit as st

from src.study_loop.ui import render

st.set_page_config(page_title="Nectivon 学习整理开发分支", page_icon="📚", layout="wide")
render()
