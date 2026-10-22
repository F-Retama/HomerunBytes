"""Pantalla 1 — Pitching Staff (Roster): inbox, mapa Stuff+ x adaptación y tabla del roster."""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dashboard.lib import data, ui  # noqa: E402

ui.inject_css()
ui.header("Roster")
if not data.data_ready():
    st.error("Faltan las tablas del dashboard. Corre `python 23_dashboard_data.py` desde la raíz del repo.")
    st.stop()

top = st.columns([2.2, 1.6], vertical_alignment="bottom")
top[0].markdown('<div class="dr-h1">Pitching Staff</div><div class="dr-sub">Overview del roster y adaptación al Harp Helú</div>',
                unsafe_allow_html=True)
with top[1]:
    st.markdown(f'<div style="display:flex;justify-content:flex-end;margin-bottom:8px">{ui.env_badge()}</div>', unsafe_allow_html=True)
    year, source, ids = data.season_and_roster(st)

P = data.load("pitchers"); P = P[(P.year == year) & P.pitcher_anon_id.isin(ids)].copy()
A = data.load("arsenal"); A = A[A.year == year]
if P.empty:
    st.info("El roster seleccionado está vacío. Elige un equipo o agrega pitchers a la lista personalizada (abajo).")

# ---------------- filtros ----------------
f = st.columns([2.2, 1.3, 1.3, 2.2, 1.6], vertical_alignment="bottom")
q = f[0].text_input("Buscar", placeholder="Buscar pitcher…", label_visibility="collapsed")
role = f[1].segmented_control("Rol", ["All", "SP", "RP"], default="All")
hand = f[2].segmented_control("Mano", ["All", "RHP", "LHP"], default="All")
act = f[3].segmented_control("Acción", ["All", "Attention", "Review", "Maintain"], default="All")
fitf = f[4].selectbox("Harp Helú Fit", ["All", "Strong Fit", "Moderate Fit", "Low Fit", "Poor Fit"])
ak = {"Attention": "red", "Review": "yellow", "Maintain": "green"}
V = P.copy()
if q:
    V = V[V.pitcher_anon_id.str.contains(q.strip().lower().replace("pitcher", "").strip(), case=False)]
if role and role != "All":
    V = V[V.role == role]
if hand and hand != "All":
    V = V[V.hand == hand]
if act and act != "All":
    V = V[V.alert == ak[act]]
if fitf != "All":
    V = V[V.fit == fitf]

# ---------------- KPIs ----------------
k = st.columns(4)
n_sp, n_rp = int((P.role == "SP").sum()), int((P.role == "RP").sum())
fitc = P.fit.value_counts()
tot = max(len(P), 1)
bars = "".join(f'<div style="flex:{fitc.get(fl, 0)};height:6px;border-radius:3px;background:{c}"></div>'
               for fl, c in [("Strong Fit", "#e9e7e2"), ("Moderate Fit", "#9a9ca2"), ("Low Fit", "#5d5f65"), ("Poor Fit", "#3a3c42")] if fitc.get(fl, 0))
# Para comparar pitchers se usa el Stuff+ equivalente a nivel del mar: el de temporada "tal como se lanzó" baja con cada pitch
# lanzado en altitud (guía 36.3), así que penalizaría a los pitchers que lanzan de local en Extreme (regla de la guía 40.1).
Pm = P[P.stuff_mar.notna()]
avg_mar = np.average(Pm.stuff_mar, weights=Pm.n) if len(Pm) else float("nan")
avg_tal = np.average(P.stuff_plus, weights=P.n) if len(P) else float("nan")
cards = [
    f'<div class="dr-label">Roster size</div><div style="margin-top:10px"><span class="dr-kpi-v">{len(P)}</span> '
    f'<span class="dr-kpi-s">pitchers · {n_sp} SP · {n_rp} RP</span> <span class="dr-tag">ROL INFERIDO</span></div>',
    f'<div class="dr-label">Average Stuff+ · nivel del mar</div><div style="margin-top:10px"><span class="dr-kpi-v">{avg_mar:.1f}</span> '
    f'<span class="dr-kpi-s">liga = 100 · ponderado por pitches · temporada tal cual {avg_tal:.1f}</span></div>',
    f'<div class="dr-label">Harp Helú Fit</div><div style="display:flex;gap:4px;margin:14px 0 8px">{bars}</div>'
    f'<div class="dr-kpi-s">' + " ".join(f'<b style="color:{ui.TEXT}">{fitc.get(fl, 0)}</b> {fl.split()[0]}' for fl in ["Strong Fit", "Moderate Fit", "Low Fit", "Poor Fit"]) + '</div>',
    f'<div class="dr-label">Requieren atención</div><div style="margin-top:10px"><span class="dr-kpi-v" style="color:{ui.RED}">{int((P.alert == "red").sum())}</span> '
    f'<span class="dr-kpi-s"><span style="color:{ui.YEL}">●</span> {int((P.alert == "yellow").sum())} review · '
    f'<span style="color:{ui.GRN}">●</span> {int((P.alert == "green").sum())} maintain</span></div>',
]
for col, html in zip(k, cards):
    col.markdown(f'<div class="dr-card" style="height:104px">{html}</div>', unsafe_allow_html=True)

st.write("")
left, right = st.columns([1.25, 1])

# ---------------- Recommendation Inbox ----------------
with left:
    I = data.load("inbox"); I = I[(I.year == year) & I.pitcher_anon_id.isin(V.pitcher_anon_id)]
    I = I.assign(order=I.alert.map({"red": 0, "yellow": 1, "green": 2})).sort_values(["order", "delta"])
    pend = int((I.type != "Opportunity").sum())
    st.markdown(f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px">'
                f'<div style="display:flex;align-items:center;gap:10px"><span class="dr-h2">Recommendation Inbox</span>'
                f'<span class="dr-badge" style="background:{ui.ACCENT};color:#fff">{pend} pendientes</span></div>'
                f'<span class="dr-mono dr-note">Datos · Temporada {year}</span></div>', unsafe_allow_html=True)
    if I.empty:
        st.markdown('<div class="dr-card dr-note">Sin alertas ni oportunidades para los filtros actuales.</div>', unsafe_allow_html=True)
    fit_by = P.set_index("pitcher_anon_id")
    for _, r in I.head(4).iterrows():
        pp = fit_by.loc[r.pitcher_anon_id]
        st.markdown(
            f'<div class="dr-card" style="display:flex;gap:18px;align-items:center;margin-bottom:10px">'
            f'<div style="flex:1"><div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">{ui.type_badge(r.type)}'
            f'<b style="font:700 15px IBM Plex Sans">{ui.pid_label(r.pitcher_anon_id)}</b>{ui.fit_pill(pp.fit)}'
            f'<span class="dr-note">{pp.role} · {pp.hand}</span></div>'
            f'<div style="font:700 19px/1.25 IBM Plex Sans;color:{ui.TEXT};margin:10px 0">{ui.esc(r.headline)}</div>'
            f'<div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap"><span class="dr-pill"><span class="dr-code">{r.code}</span>'
            f'{r.AutoPitchType}</span><span class="dr-note">{ui.esc(r.evidence_text)}</span></div></div>'
            f'<div style="border-left:1px solid {ui.LINE};padding-left:16px;min-width:130px"><div class="dr-label">Confianza · altitud</div>'
            f'<div style="margin-top:8px">{ui.conf_bars(r.confidence)}</div></div>'
            f'<a class="dr-btn" target="_self" href="player?pid={r.pitcher_anon_id}&season={year}&pitch={r.AutoPitchType}">Review →</a></div>',
            unsafe_allow_html=True)
    if len(I) > 4:
        with st.expander(f"Ver {len(I) - 4} items más"):
            for _, r in I.iloc[4:].iterrows():
                st.markdown(f'{ui.type_badge(r.type)} <b>{ui.pid_label(r.pitcher_anon_id)}</b> · {ui.esc(r.headline)} '
                            f'<a target="_self" href="player?pid={r.pitcher_anon_id}&season={year}&pitch={r.AutoPitchType}">Review →</a>', unsafe_allow_html=True)

# ---------------- Harp Helú Roster Map ----------------
with right:
    st.markdown('<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:10px">'
                '<span class="dr-h2">Harp Helú Roster Map</span><span class="dr-note">Stuff+ × adaptación vs. lo esperado</span></div>', unsafe_allow_html=True)
    M = V[(V.fit != "Sin muestra") & V.stuff_mar.notna()]
    fig = go.Figure()
    for y0, y1, lab in [(75, 100, "Pierde menos<br>de lo esperado"), (50, 75, ""), (25, 50, ""), (0, 25, "Pierde más<br>de lo esperado")]:
        fig.add_hrect(y0=y0, y1=y1, fillcolor="rgba(255,255,255,0.025)" if lab in ("Strong Fit", "Low Fit") else "rgba(0,0,0,0)", line_width=0)
        fig.add_annotation(x=0, xref="paper", y=(y0 + y1) / 2, text=f"<b>{lab}</b>", showarrow=False, xanchor="right", xshift=-8,
                           font=dict(size=11, color="#c6c8cd"))
    fig.add_vline(x=100, line=dict(color="rgba(255,255,255,0.18)", dash="dot"))
    for al, (c, lab) in ui.ALERT.items():
        m = M[M.alert == al]
        fig.add_trace(go.Scatter(x=m.stuff_mar, y=100 * m.exceso_pct, mode="markers+text", name=lab,
                                 text=[ui.pid_label(x).replace("Pitcher ", "") for x in m.pitcher_anon_id], textposition="middle right",
                                 textfont=dict(size=10, color="#c6c8cd"),
                                 marker=dict(color=c, size=10 + np.sqrt(m.n) / 4, line=dict(color="#15161a", width=2)),
                                 customdata=np.stack([m.pitcher_anon_id, m.delta, m.n, m.stuff_harp, m.stuff_plus], axis=1) if len(m) else None,
                                 hovertemplate="%{customdata[0]}<br>Stuff+ nivel del mar %{x:.1f} · en Harp Helú %{customdata[3]:.1f}"
                                               "<br>temporada tal cual %{customdata[4]:.1f}<br>Δ altitud %{customdata[1]:+.1f}<br>n %{customdata[2]}<extra></extra>"))
    fig.update_layout(height=430, margin=dict(l=110, r=10, t=10, b=40), paper_bgcolor=ui.PANEL, plot_bgcolor=ui.PANEL,
                      font=dict(family="IBM Plex Sans", color=ui.DIM), legend=dict(orientation="h", y=1.08, x=0, font=dict(size=11)),
                      xaxis=dict(title="Stuff+ a nivel del mar →", gridcolor="rgba(255,255,255,0.05)", zeroline=False,
                                 range=[min(M.stuff_mar.min(), 95) - 1.5, max(M.stuff_mar.max(), 105) + 3] if len(M) else None),
                      yaxis=dict(title=None, range=[0, 100], showticklabels=False, gridcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('<div class="dr-note">Eje horizontal = Stuff+ del arsenal llevado a nivel del mar, para comparar pitchers sin importar en qué parques '
                'lanzaron. Eje vertical = percentil de cuánto pierde el arsenal en altitud comparado con lo esperado para su nivel de '
                'Stuff+ y sus tipos de pitch (los mejores pitchers pierden más puntos en términos absolutos; aquí eso se descuenta). '
                'Arriba = pierde menos de lo esperado.</div>', unsafe_allow_html=True)

# ---------------- Tabla del roster ----------------
st.write("")
T = data.load("trends"); T = T[T.year == year]
order = {"red": 0, "yellow": 1, "green": 2}
num = lambda v, f=".0f": "—" if v != v else format(v, f)  # sin proyección (sin altitud o sin movimiento medido)
V = V.assign(o=V.alert.map(order)).sort_values(["o", "stuff_mar"], ascending=[True, False])
rows = []
for _, r in V.iterrows():
    tr = T[T.pitcher_anon_id == r.pitcher_anon_id].sort_values("week").stuff_plus.tail(6).tolist()
    obs = "" if r.delta_observado != r.delta_observado else f'<div class="dr-note">obs. {r.delta_observado:+.1f}</div>'
    c, lab = ui.ALERT[r.alert]
    rows.append(
        f'<tr><td><a target="_self" href="player?pid={r.pitcher_anon_id}&season={year}"><b>{ui.pid_label(r.pitcher_anon_id)}</b></a>'
        f'<div class="dr-note">{r.role} · {r.hand} · n {int(r.n):,}</div></td>'
        f'<td class="dr-mono"><b>{num(r.stuff_mar)}</b><div class="dr-note">temp. {r.stuff_plus:.0f}</div></td>'
        f'<td class="dr-mono">{r.location_plus:.0f}</td><td class="dr-mono">{r.pitching_plus:.0f}</td>'
        f'<td>{ui.fit_pill(r.fit)} <span class="dr-mono dr-note">→ {num(r.stuff_harp)} en Harp</span><div style="margin-top:4px">{ui.conf_bars(r.confidence)}</div></td>'
        f'<td>{ui.delta_fmt(r.delta)} <span class="dr-note">Stuff+</span>{obs}</td>'
        f'<td><span style="color:{c};font-weight:600">● {lab}</span></td><td>{ui.sparkline(tr)}</td>'
        f'<td style="text-align:right"><a class="dr-btn-ghost" target="_self" href="player?pid={r.pitcher_anon_id}&season={year}">Player →</a></td></tr>')
st.markdown(
    f'<div class="dr-card" style="padding:0"><div style="display:flex;justify-content:space-between;align-items:center;padding:16px 18px">'
    f'<div><span class="dr-h2">Roster</span> <span class="dr-note">{len(V)} pitchers · {ui.esc(source)}</span></div>'
    f'<span class="dr-note">Stuff+ = mismo arsenal llevado a nivel del mar, comparable entre pitchers (temp. = temporada tal como se lanzó, '
    f'depende de en qué parques lanzó). Harp Helú Fit = Stuff+ proyectado en Harp Helú frente a la liga. Δ altitud = efecto del aire '
    f'sobre el mismo pitch (variante física). obs. = diferencia real Extreme − No Altitude del mismo pitcher. '
    f'Barras = confianza en el efecto de altitud (evidencia de sus tipos de pitch y muestra), no en el Stuff+.</span></div>'
    f'<table class="dr-tbl"><tr><th>Pitcher</th><th>Stuff+ (mar)</th><th>Location+</th><th>Pitching+</th><th>Harp Helú Fit</th><th>Δ altitud</th>'
    f'<th>Alert</th><th>Trend · 6 sem</th><th></th></tr>{"".join(rows) or "<tr><td colspan=9 class=dr-note>Ningún pitcher coincide con los filtros.</td></tr>"}</table></div>',
    unsafe_allow_html=True)

# ---------------- Comparar y lista personalizada ----------------
st.write("")
c1, c2 = st.columns([1.4, 1])
with c1:
    sel = st.multiselect("Comparar pitchers", V.pitcher_anon_id.tolist(), format_func=ui.pid_label, max_selections=4,
                         placeholder="Selecciona 2 a 4 pitchers")
    if len(sel) >= 2:
        C = P.set_index("pitcher_anon_id").loc[sel]
        head = "".join(f"<th>{ui.pid_label(i)}</th>" for i in sel)
        def row(lbl, fn):
            return f"<tr><td class='dr-note'>{lbl}</td>" + "".join(f"<td>{fn(C.loc[i])}</td>" for i in sel) + "</tr>"
        st.markdown(
            f'<div class="dr-card" style="padding:0"><table class="dr-tbl"><tr><th></th>{head}</tr>'
            + row("Rol · mano", lambda r: f"{r.role} · {r.hand}")
            + row("Stuff+ (nivel del mar → Harp Helú)", lambda r: f'<span class="dr-mono">{num(r.stuff_mar, ".1f")} → {num(r.stuff_harp, ".1f")}</span>')
            + row("Location+ / Pitching+", lambda r: f'<span class="dr-mono">{r.location_plus:.0f} / {r.pitching_plus:.0f}</span>')
            + row("Harp Helú Fit", lambda r: ui.fit_pill(r.fit))
            + row("Δ altitud (obs.)", lambda r: ui.delta_fmt(r.delta) + ("" if r.delta_observado != r.delta_observado else f' <span class="dr-note">({r.delta_observado:+.1f})</span>'))
            + row("Mejor en Harp · pierde más de lo esperado", lambda r: f"{r.best_pitch} · {r.worst_pitch}")
            + row("Confianza del efecto de altitud", lambda r: ui.conf_bars(r.confidence))
            + "</table></div>", unsafe_allow_html=True)
with c2:
    with st.expander("Lista personalizada (roster del club)"):
        allp = data.load("pitchers"); allp = allp[(allp.year == year) & (allp.n >= 50)]
        cur = data.read_state().get("custom_roster", [])
        new = st.multiselect("Pitchers", allp.pitcher_anon_id.tolist(), default=[x for x in cur if x in set(allp.pitcher_anon_id)],
                             format_func=ui.pid_label)
        if st.button("Guardar lista"):
            data.write_state(custom_roster=new)
            st.success(f"Lista guardada ({len(new)} pitchers). Selecciónala en 'Roster'.")
