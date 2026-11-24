import base64
from pathlib import Path

import pandas as pd
import streamlit as st


@st.cache_data
def header_image():
    path = Path(__file__).with_name("race-header.webp")
    return base64.b64encode(path.read_bytes()).decode("ascii")


def masthead(title, description):
    st.html("""
    <style>
    .stMainBlockContainer {max-width: 1220px; padding-top: 2.6rem;}
    .race-header {box-sizing:border-box; display:flex; flex-direction:column;
      justify-content:flex-end; min-height:clamp(180px,19vw,250px);
      padding:clamp(24px,3.6vw,50px); color:white; background-size:cover;
      background-position:center; border-radius:16px; overflow:hidden;
      border-bottom:5px solid #ed382e; margin-bottom:12px;}
    .race-header h1 {color:white; font-family:Barlow,Arial,sans-serif;
      font-weight:850; letter-spacing:-.055em; line-height:1;
      font-size:clamp(38px,5vw,72px); margin:0; padding:0;}
    .race-header p {color:#f3f3f3; max-width:31em; margin:12px 0 0;
      font:500 15px/1.45 Arial,sans-serif;}
    div[data-testid="stMetricValue"] {font-variant-numeric:tabular-nums;}
    button[kind="primary"] {background:#be251d; color:white; border-color:#be251d;}
    @media(max-width:600px){.race-header {background-position:62% center;}}
    </style>
    """)
    background = (
        "linear-gradient(90deg,rgba(13,16,20,.93),rgba(13,16,20,.56) 48%,"
        f"transparent 85%),url(data:image/webp;base64,{header_image()})"
    )
    st.html(f'<div class="race-header" style="background-image:{background}">'
            f'<h1>{title}</h1><p>{description}</p></div>')


def results_table(records):
    frame = pd.json_normalize(records)
    columns = {
        "driver_id": "Driver", "constructor_name": "Constructor", "grid": "Grid",
        "prediction": "Top 10", "probability": "Top-10 chance", "circuit_id": "Circuit",
        "season": "Season", "round": "Round", "created_at": "Created",
        "source": "Source", "model_version": "Model", "id": "ID",
    }
    if "probability" in frame:
        frame["probability"] = frame["probability"].map(
            lambda p: f"{p:.1%}" if pd.notna(p) else "—"
        )
    order = [key for key in columns if key in frame]
    st.dataframe(frame[order].rename(columns=columns), hide_index=True, width="stretch")
