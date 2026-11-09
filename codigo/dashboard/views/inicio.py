"""Inicio — la historia del tablero y el diferenciador: caída esperada contra caída real en Harp Helú."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dashboard.lib import data, ui  # noqa: E402

ui.inject_css()
ui.header("Inicio")
if not data.data_ready():
    st.error("Faltan las tablas del dashboard. Corre `python 23_dashboard_data.py` desde la raíz del repo.")
    st.stop()

year = data.seasons()[0]
P = data.load("pitchers"); P = P[P.year == year]
A = data.load("arsenal"); A = A[A.year == year]

ui.title("Nuestro diferenciador",
         "No basta con saber cuánto pierde un pitch en la altura. Lo que importa es si pierde más de lo que debería.",
         "Todos los pitches pierden algo en el Harp Helú. Comparamos la caída real de cada uno contra la caída esperada "
         "para su tipo y su nivel; la diferencia es la señal que usa todo el tablero.")

# ---------------- ejemplo: el pitch del roster por defecto que más cae de lo esperado ----------------
st_ = data.read_state()
opts = data.roster_options(year)
src = st_.get("roster_source") if st_.get("roster_source") in opts else next((o for o in opts if "Extreme" in o), opts[0])
ids = set(data.roster_ids(year, src)) or set(P.pitcher_anon_id)
cand = A[A.pitcher_anon_id.isin(ids) & A.exceso.notna() & A.delta.notna() & (A.action == "Adjust")]
if cand.empty:
    cand = A[A.pitcher_anon_id.isin(ids) & A.exceso.notna() & A.delta.notna()]
if not cand.empty:
    ex = cand.sort_values("exceso").iloc[0]
    esperada, real = ex.delta - ex.exceso, ex.delta
    c = st.columns([1, 1.6], gap="large")
    c[0].markdown(
        f'<div class="dr-card" style="height:100%"><div class="dr-label">Ejemplo real · temporada {year}</div>'
        f'<div class="dr-h2" style="margin:8px 0 10px">{ui.PITCH_ES.get(ex.AutoPitchType, ex.AutoPitchType)} del {ui.pid_label(ex.pitcher_anon_id)}</div>'
        f'<div class="dr-sub" style="font-size:15px">Un pitch de su tipo y nivel suele perder {abs(esperada):.1f} puntos de Stuff+ en el Harp Helú. '
        f'El suyo pierde {abs(real):.1f}. Esa diferencia es lo que lo convierte en alerta.</div>'
        f'<a class="dr-btn" target="_self" href="player?pid={ex.pitcher_anon_id}&season={year}&pitch={ex.AutoPitchType}">Ver al pitcher</a></div>',
        unsafe_allow_html=True)
    c[1].markdown(f'<div class="dr-card">{ui.brecha(esperada, real)}</div>', unsafe_allow_html=True)

# ---------------- la historia en tres pantallas ----------------
st.markdown('<div class="dr-h2" style="margin:36px 0 14px">La historia en tres pantallas</div>', unsafe_allow_html=True)
story = [("views/scouting.py", "Scouting", "Definimos el pitcher ideal para el Harp Helú y buscamos a quién se le parece",
          "Candidatos ordenados por cómo rendirían aquí y si caen más o menos de lo esperado."),
         ("views/game.py", "Game", "El mismo perfil, al revés: qué esperar del pitcher rival",
          "Un reporte para los bateadores: qué pitches del rival se debilitan aquí y cuál no cambia."),
         ("views/simulator.py", "Simulator", "Para los nuestros: qué ajuste recupera lo que se pierde",
          "Probar velocidad, giro o movimiento antes de trabajarlo con el pitcher.")]
cols = st.columns(3, gap="medium")
for col, (path, name, h, s) in zip(cols, story):
    with col:
        st.markdown(f'<div class="dr-card" style="min-height:190px"><div style="font:700 14px Space Grotesk;color:{ui.TEXT_RED}">{name}</div>'
                    f'<div style="font:700 20px/1.25 Space Grotesk;margin:8px 0">{h}</div><div class="dr-note" style="font-size:14px">{s}</div></div>',
                    unsafe_allow_html=True)
        st.page_link(path, label=f"Abrir {name}")

# ---------------- seguimiento y cifras base ----------------
ui.mas_detalle("Seguimiento semanal y cifras base")
c = st.columns(2, gap="medium")
with c[0]:
    st.markdown('<div class="dr-card"><div class="dr-h2" style="font-size:18px">Roster</div>'
                '<div class="dr-note" style="font-size:14px">Recommendation Inbox, Roster Map y tabla del equipo</div></div>', unsafe_allow_html=True)
    st.page_link("views/roster.py", label="Abrir Roster")
with c[1]:
    st.markdown('<div class="dr-card"><div class="dr-h2" style="font-size:18px">Player</div>'
                '<div class="dr-note" style="font-size:14px">Arsenal, fingerprint de movimiento y development targets</div></div>', unsafe_allow_html=True)
    st.page_link("views/player.py", label="Abrir Player")

kp = []
if data.has("delta_by_type"):
    D = data.load("delta_by_type"); D = D[(D.window == D.window.max()) & (D.get("variante", "fisica") == "fisica")].set_index("AutoPitchType")  # titular = variante física
    ff, si = D.delta_stuff.get("Four-Seam"), D.delta_stuff.get("Sinker")
    if ff is not None and si is not None:
        kp.append((f"{ff:+.1f} · {si:+.1f}", "puntos de Stuff+ que pierden la recta y el sinker en Harp Helú", ui.TEXT_RED))
kp.append(("77% · 71%", "del movimiento vertical y horizontal que conserva el mismo pitcher en altitud extrema", ui.TEXT))
Pa = P[P.n >= 100]
kp.append((f"{int((Pa.alert == 'red').sum())} · {int((Pa.alert == 'yellow').sum())}",
           f"pitchers con alerta roja y amarilla en {year}, de {len(Pa)} con 100+ pitches", ui.TEXT_RED))
st.write("")
for col, (v, s, colr) in zip(st.columns(len(kp)), kp):
    col.markdown(f'<div class="dr-card"><div class="dr-kpi-v" style="color:{colr}">{v}</div><div class="dr-kpi-s" style="margin-top:8px">{s}</div></div>',
                 unsafe_allow_html=True)
