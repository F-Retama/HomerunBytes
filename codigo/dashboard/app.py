"""Pitching Decision System — Diablos Rojos (Harp Helú). Ejecutar desde la raíz del repo:

    streamlit run dashboard/app.py

Requiere haber corrido 23_dashboard_data.py (y antes 17-22) para generar dashboard/data/.
"""

import streamlit as st

st.set_page_config(page_title="Pitching Decision System", page_icon="⚾", layout="wide", initial_sidebar_state="collapsed")

pages = [st.Page("views/roster.py", title="Roster", url_path="roster", default=True),
         st.Page("views/player.py", title="Player", url_path="player"),
         st.Page("views/simulator.py", title="Simulator", url_path="simulator"),
         st.Page("views/scouting.py", title="Scouting", url_path="scouting"),
         st.Page("views/game.py", title="Game", url_path="game")]
st.navigation(pages, position="hidden").run()
