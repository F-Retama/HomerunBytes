"""Pantalla 2 — Coach / Player Overview: diagnóstico, arsenal, adaptación a Harp Helú, why y targets."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dashboard.lib import data, ui  # noqa: E402

ui.inject_css()
ui.header("Player")
if not data.data_ready():
    st.error("Faltan las tablas del dashboard. Corre `python 23_dashboard_data.py` desde la raíz del repo.")
    st.stop()

P_all, A_all = data.load("pitchers"), data.load("arsenal")
qp = st.query_params
state = data.read_state()
year = int(qp.get("season", state.get("season", P_all.year.max())))
pid = qp.get("pid", state.get("last_pid"))
pool = P_all[(P_all.year == year) & (P_all.n >= 50)].sort_values("n", ascending=False)
if pid not in set(pool.pitcher_anon_id):
    roster = data.roster_ids(year, state.get("roster_source", "Liga completa"))
    cand = pool[pool.pitcher_anon_id.isin(roster)]
    pid = (cand if len(cand) else pool).pitcher_anon_id.iloc[0]

st.markdown('<a target="_self" href="./" class="dr-note" style="text-decoration:none">← Pitching Staff</a>', unsafe_allow_html=True)
sel = st.columns([2.2, 1, 1.4], vertical_alignment="bottom")
pid = sel[1].selectbox("Pitcher", pool.pitcher_anon_id.tolist(), index=pool.pitcher_anon_id.tolist().index(pid), format_func=ui.pid_label)
if pid != state.get("last_pid"):
    data.write_state(last_pid=pid)
p = pool.set_index("pitcher_anon_id").loc[pid]
A = A_all[(A_all.year == year) & (A_all.pitcher_anon_id == pid) & (A_all.n >= 20)].sort_values("usage", ascending=False)
I = data.load("inbox"); I = I[(I.year == year) & (I.pitcher_anon_id == pid)]
focus = qp.get("pitch")

sel[0].markdown(
    f'<div style="display:flex;align-items:center;gap:12px"><span class="dr-h1" style="margin:6px 0">{ui.pid_label(pid)}</span>'
    f'{ui.alert_badge(p.alert)}</div><div style="display:flex;gap:8px;align-items:center;margin-top:4px">'
    f'<span class="dr-pill">{p.hand} · {p.role} <span class="dr-tag">INFERIDO</span></span>'
    f'<span class="dr-pill">Temporada <b>{year}</b></span><span class="dr-pill">{ui.esc(p.team_label if isinstance(p.team_label, str) else "Sin equipo reconstruido")}</span></div>',
    unsafe_allow_html=True)
sel[2].markdown(f'<div style="display:flex;justify-content:flex-end;gap:8px;align-items:center">{ui.env_badge()}'
                f'<a class="dr-btn" target="_self" href="simulator?pid={pid}&season={year}">Open Simulator →</a></div>', unsafe_allow_html=True)

# ---------------- recomendación actual ----------------
rec = I[I.AutoPitchType == focus].head(1) if focus else I.head(0)
if rec.empty:
    rec = I.assign(o=I.alert.map({"red": 0, "yellow": 1, "green": 2})).sort_values(["o", "delta"]).head(1)
# Mejor pitch = mayor percentil dentro de su tipo en Harp Helú (la escala de Stuff+ no se recentra por tipo).
best = A.sort_values("harp_pct" if A.harp_pct.notna().any() else "stuff_plus", ascending=False).iloc[0]
core = A[A.n >= 50] if (A.n >= 50).any() else A
# Riesgo principal = el pitch (uso >= 10%) que pierde más DE LO ESPERADO para su nivel y tipo (exceso, guía 38.3), la misma
# lectura que alertas e inbox. El Δ bruto casi siempre señalaba a la recta, porque los mejores pitches pierden más puntos.
cand = A[A.usage >= 0.10] if (A.usage >= 0.10).any() else A
worst = (cand[cand.exceso.notna()].sort_values("exceso") if cand.exceso.notna().any() else cand.sort_values("delta")).iloc[0]
alt_core = core[core.AutoPitchType != worst.AutoPitchType]
res = (alt_core if len(alt_core) else core).sort_values("harp_pct" if core.harp_pct.notna().any() else "stuff_harp", ascending=False).iloc[0]
fac = data.load("fingerprint"); fac = fac[(fac.year == year) & (fac.pitcher_anon_id == pid)]
wfp = fac[fac.AutoPitchType == worst.AutoPitchType].set_index("env")
if focus:
    st.markdown(f'<div class="dr-card" style="padding:9px 14px;margin:12px 0 0"><span style="color:{ui.ACCENT}">◆</span> '
                f'<span class="dr-note">Reviewing:</span> <b>{ui.esc(focus)} · Altitude adaptation</b> &nbsp; '
                f'{ui.conf_bars(A.set_index("AutoPitchType").evidence.get(focus, "Uncertain"))}'
                f'<span class="dr-note" style="float:right">Desde Recommendation Inbox</span></div>', unsafe_allow_html=True)
if len(rec):
    r = rec.iloc[0]
    rec_txt = ui.esc(r.headline) + (". Revisar en el Simulator cómo cambia con su shape actual antes de ajustar el plan." if r.type != "Opportunity" else ".")
    rec_kind, rec_conf, rec_pitch = r.type, r.confidence, r.AutoPitchType
else:
    rec_txt = f"Sin alertas en {year}: el arsenal conserva su calidad en Harp Helú dentro de lo esperado para sus tipos de pitch."
    rec_kind, rec_conf, rec_pitch = "Opportunity", p.confidence, best.AutoPitchType
kind_badge = {"Needs attention": ("Adjust", ui.RED), "Review": ("Review", ui.YEL), "Opportunity": ("Maintain", ui.GRN)}[rec_kind]
ivb0, ivb1 = (wfp.loc[e].ivb if e in wfp.index else float("nan") for e in ("No Altitude", "Extreme Altitude"))
w_move = (f"el aire le quita movimiento (IVB {ivb0:.1f} → {ivb1:.1f} in)" if ivb0 == ivb0 and ivb1 == ivb1
          else "sin movimiento medido para mostrar el cambio")
wd = worst.delta
w_change = ("no tiene proyección" if wd != wd else "conserva su Stuff+" if abs(wd) < 0.05
            else f"pierde {abs(wd):.1f} de Stuff+" if wd < 0 else f"gana {wd:.1f} de Stuff+")
if worst.exceso == worst.exceso and worst.exceso < 0:
    risk_dot, risk_txt = ui.RED, (f"El {worst.AutoPitchType} ({worst.usage:.0%} de uso) {w_change} en Harp Helú, {abs(worst.exceso):.1f} más de "
                                  f"lo esperado para un {worst.AutoPitchType} de su nivel: {w_move}.")
elif worst.exceso == worst.exceso:
    risk_dot, risk_txt = ui.YEL, (f"Ningún pitch pierde más de lo esperado para su nivel. El más cercano es el {worst.AutoPitchType} "
                                  f"({worst.usage:.0%} de uso): {w_change} en Harp Helú, típico de su tipo; {w_move}.")
elif wd != wd:
    risk_dot, risk_txt = ui.YEL, (f"Sin proyección a Harp Helú para este arsenal (pitches sin movimiento medido o sin categoría de altitud): "
                                  f"no se puede estimar su riesgo en altitud.")
else:
    risk_dot, risk_txt = ui.RED, f"El {worst.AutoPitchType} ({worst.usage:.0%} de uso) {w_change} en Harp Helú: {w_move}."
c = st.columns([1.55, 1])
c[0].markdown(
    f'<div class="dr-card" style="height:100%;min-height:230px"><div style="display:flex;gap:10px;align-items:center"><span class="dr-label">Current recommendation</span>'
    f'<span class="dr-badge" style="background:{kind_badge[1]}24;color:{kind_badge[1]}">● {kind_badge[0]}</span></div>'
    f'<div style="font:700 23px/1.3 IBM Plex Sans;color:{ui.TEXT};margin:14px 0 16px">{rec_txt}</div>'
    f'<div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap"><span class="dr-pill"><span class="dr-code">{ui.CODE.get(rec_pitch, "")}</span>{rec_pitch}</span>'
    f'<span style="margin-left:12px"><span class="dr-label">Confianza · altitud</span><br>{ui.conf_bars(rec_conf)}</span></div></div>', unsafe_allow_html=True)
if res.stuff_harp == res.stuff_harp:
    pct_txt = f"percentil {100 * res.harp_pct:.0f} de los {res.AutoPitchType} de la liga; " if res.harp_pct == res.harp_pct else ""
    strength_txt = (f"Su pitch más destacado en Harp Helú frente a su tipo es el {res.AutoPitchType} "
                    f"({pct_txt}Stuff+ {res.stuff_mar:.0f} a nivel del mar → {res.stuff_harp:.0f} en altitud).")
else:
    sp = core.sort_values("stuff_plus", ascending=False).iloc[0]
    strength_txt = (f"Su pitch con mejor Stuff+ de temporada es el {sp.AutoPitchType} ({sp.stuff_plus:.0f}). Sin proyección a Harp Helú: "
                    f"sus pitches no tienen movimiento medido o categoría de altitud.")
c[1].markdown(
    f'<div class="dr-card" style="height:100%;min-height:230px"><div class="dr-label"><span style="color:{ui.GRN}">●</span> Primary strength</div>'
    f'<div style="font:600 14.5px/1.4 IBM Plex Sans;margin:8px 0 18px">{strength_txt}</div>'
    f'<div style="height:1px;background:{ui.LINE};margin-bottom:14px"></div>'
    f'<div class="dr-label"><span style="color:{risk_dot}">●</span> Primary risk</div>'
    f'<div style="font:600 14.5px/1.4 IBM Plex Sans;margin-top:8px">{risk_txt}</div></div>', unsafe_allow_html=True)
k = st.columns([1, 1, 1, 1.6])
# Stuff+ a nivel del mar, igual que el Roster (guía 40.1): el de temporada depende de en qué parques lanzó y queda como dato secundario.
num = lambda v, f=".0f": "—" if v != v else format(v, f)
for col, lab, v, sub in [(k[0], "Stuff+ · nivel del mar", p.stuff_mar, f"temp. {num(p.stuff_plus)}"), (k[1], "Location+", p.location_plus, ""),
                         (k[2], "Pitching+", p.pitching_plus, "temporada")]:
    col.markdown(f'<div class="dr-card" style="padding:12px 16px"><span class="dr-label">{lab}</span> <span class="dr-kpi-v" style="font-size:28px;margin-left:8px">{num(v)}</span>'
                 f' <span class="dr-note">{sub}</span></div>', unsafe_allow_html=True)
if p.stuff_harp == p.stuff_harp:
    fit_detail = (f'<span class="dr-mono">{p.stuff_mar:.0f}→{p.stuff_harp:.0f}</span>{ui.conf_bars(p.confidence)}<span class="dr-note">Δ altitud {p.delta:+.1f}'
                  + ("" if p.delta_observado != p.delta_observado else f' · obs. {p.delta_observado:+.1f}')
                  + ("" if p.exceso != p.exceso else f' · {"menos" if p.exceso >= 0 else "más"} de lo esperado ({p.exceso:+.1f})') + '</span>')
else:
    fit_detail = '<span class="dr-note">Sin proyección a Harp Helú (pitches sin movimiento medido o sin categoría de altitud)</span>'
k[3].markdown(f'<div class="dr-card" style="padding:12px 16px;display:flex;gap:12px;align-items:center"><span class="dr-label">Harp Helú Fit</span>{ui.fit_pill(p.fit)}'
              f'{fit_detail}</div>', unsafe_allow_html=True)

# ---------------- arsenal + fingerprint ----------------
st.write("")
c = st.columns([1.55, 1])
rows = []
for _, a in A.iterrows():
    hi = "background:rgba(255,255,255,0.035);" if focus == a.AutoPitchType else ""
    rows.append(f'<tr style="{hi}"><td style="white-space:nowrap"><span class="dr-code">{a.code}</span> <b>{a.AutoPitchType}</b></td>'
                f'<td style="white-space:nowrap"><span style="display:inline-block;width:{a.usage * 70:.0f}px;height:3px;background:#c6c8cd;border-radius:2px;vertical-align:middle"></span> '
                f'<span class="dr-mono">{a.usage:.0%}</span></td><td class="dr-mono"><b>{num(a.stuff_mar)}</b><div class="dr-note">temp. {num(a.stuff_plus)}</div></td><td class="dr-mono">{a.location_plus:.0f}</td>'
                f'<td>{ui.delta_fmt(a.delta)}</td><td>{ui.fit_pill(a.fit)}</td><td>{ui.action_dot(a.action)}</td><td>{ui.conf_bars(a.evidence)}</td></tr>')
c[0].markdown(
    f'<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:10px"><span class="dr-h2">Arsenal Overview</span>'
    f'<span class="dr-mono dr-note">Datos · Temporada {year}</span></div><div class="dr-card" style="padding:0"><table class="dr-tbl">'
    f'<tr><th>Pitch</th><th>Usage</th><th>Stuff+ (mar)</th><th>Loc+</th><th>Δ alt.</th><th>Fit</th><th>Action</th><th>Confianza · altitud</th></tr>'
    f'{"".join(rows)}</table><div class="dr-note" style="padding:10px 14px">Stuff+ (mar) = el mismo pitch llevado a nivel del mar, comparable entre pitchers; temp. = tal como se lanzó en sus parques. '
    f'Δ alt. = efecto del aire sobre el Stuff+ del mismo pitch (variante física). '
    f'Fit = Stuff+ proyectado en Harp Helú frente a pitches del mismo tipo. Action = pierde más de lo esperado para su nivel (Adjust/Review). '
    f'Confianza = evidencia del efecto de altitud para su tipo de pitch (guía 36), no del Stuff+. '
    f'Acciones con evidencia Conditional o Uncertain son sugerencias de revisión, no instrucciones.</div></div>',
    unsafe_allow_html=True)
with c[1]:
    hh = st.columns([0.9, 1.3], vertical_alignment="bottom")
    hh[0].markdown('<span class="dr-h2">Arsenal Fingerprint</span>', unsafe_allow_html=True)
    layer = hh[1].segmented_control("Capa", ["Both", "Nivel mar", "Extreme"], default="Both", label_visibility="collapsed")
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.15)")); fig.add_vline(x=0, line=dict(color="rgba(255,255,255,0.15)"))
    for _, a in A.iterrows():
        g = fac[fac.AutoPitchType == a.AutoPitchType].set_index("env")
        if not {"No Altitude", "Extreme Altitude"} <= set(g.index):
            continue
        na, ea = g.loc["No Altitude"], g.loc["Extreme Altitude"]
        size = 10 + a.usage * 40
        if layer == "Both":
            fig.add_trace(go.Scatter(x=[na.hb, ea.hb], y=[na.ivb, ea.ivb], mode="lines", line=dict(color="rgba(255,255,255,0.25)", dash="dot"), showlegend=False, hoverinfo="skip"))
        if layer in ("Both", "Nivel mar"):
            fig.add_trace(go.Scatter(x=[na.hb], y=[na.ivb], mode="markers", marker=dict(size=size, color="rgba(0,0,0,0)", line=dict(color="#9b9da4", width=2, dash=None)),
                                     showlegend=False, hovertemplate=f"{a.AutoPitchType} · No Altitude ({na.fuente})<br>HB %{{x:.1f}} in · IVB %{{y:.1f}} in<extra></extra>"))
        if layer in ("Both", "Extreme"):
            fig.add_trace(go.Scatter(x=[ea.hb], y=[ea.ivb], mode="markers+text", text=[a.code], textposition="middle right",
                                     textfont=dict(color="#c6c8cd", size=11), marker=dict(size=size, color="#eeede9", line=dict(color="#15161a", width=2)),
                                     showlegend=False, hovertemplate=f"{a.AutoPitchType} · Extreme ({ea.fuente})<br>HB %{{x:.1f}} in · IVB %{{y:.1f}} in<extra></extra>"))
    fig.update_layout(height=360, margin=dict(l=40, r=10, t=10, b=40), paper_bgcolor=ui.PANEL, plot_bgcolor=ui.PANEL,
                      font=dict(family="IBM Plex Sans", color=ui.DIM, size=11),
                      xaxis=dict(title="HB (in) → · convención de diestro", range=[-22, 22], gridcolor="rgba(255,255,255,0.05)", zeroline=False),
                      yaxis=dict(title="IVB (in) ↑", range=[-22, 24], gridcolor="rgba(255,255,255,0.05)", zeroline=False))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    v = "".join(f'<div><b class="dr-mono">{a.code}</b><br><span class="dr-mono">{a.velo:.1f}</span> <span class="dr-note">mph</span><br>'
                f'<span class="dr-note dr-mono">rel {a.rel_height:.1f} ft</span></div>' for _, a in A.iterrows())
    st.markdown(f'<div style="display:flex;gap:22px;flex-wrap:wrap">{v}</div><div class="dr-note" style="margin-top:6px">○ No Altitude · ● Extreme. '
                f'Observado si hay ≥ 20 pitches en ese entorno; si no, proyectado con la transformación validada (IVB × 0.77, HB × 0.71).</div>',
                unsafe_allow_html=True)

# ---------------- adaptación ----------------
st.write("")
st.markdown('<span class="dr-h2">Harp Helú Adaptation</span> <span class="dr-note">&nbsp;Nivel del mar vs. Extreme Altitude (mismo pitch)</span>', unsafe_allow_html=True)

def has_projection(a):
    """El pitch tiene Δ de altitud y movimiento en ambos entornos (fingerprint); si no, no hay lectura de adaptación."""
    envs = set(fac[fac.AutoPitchType == a.AutoPitchType].env)
    return a.delta == a.delta and {"No Altitude", "Extreme Altitude"} <= envs


top3 = A[(A.n >= 50) & A.apply(has_projection, axis=1)].head(3) if len(A) else A
if top3.empty:
    st.markdown('<div class="dr-card dr-note">Sin proyección a Harp Helú para este pitcher: sus pitches no tienen movimiento medido '
                'o categoría de altitud.</div>', unsafe_allow_html=True)
cols = st.columns(max(len(top3), 1))
for col, (_, a) in zip(cols, top3.iterrows()):
    g = fac[fac.AutoPitchType == a.AutoPitchType].set_index("env")
    na, ea = g.loc["No Altitude"], g.loc["Extreme Altitude"]

    def arrow(d, tol):
        return "↓" if d < -tol else "↑" if d > tol else "≈"
    dw, dc = a.whiff_harp - a.whiff_mar, a.chase_harp - a.chase_mar
    # Efecto neto (guia 40): Neutral si |Δ| < 1 (umbral de relevancia de 21). Con |Δ| >= 1 es Mixed cuando hay
    # señales en conflicto: (a) el modelo de arsenal da el signo contrario con al menos 1 punto, o (b) whiff o chase
    # se mueven >= 0.5 pp en contra del Stuff+. Si no, Favorable / Unfavorable.
    sgn = 1 if a.delta > 0 else -1
    c_var = (a.delta_oficial * sgn <= -1)
    c_comp = (dw * sgn <= -0.005) or (dc * sgn <= -0.005)
    net = "Neutral" if abs(a.delta) < 1 else "Mixed" if (c_var or c_comp) else "Favorable" if a.delta > 0 else "Unfavorable"
    net_note = {"Neutral": "El cambio de Stuff+ es menor a 1 punto: sin efecto práctico, aunque algún componente se mueva.",
                "Mixed": (f"Por sí solo el pitch {'gana' if sgn > 0 else 'pierde'} {abs(a.delta):.1f}, pero frente a su recta el modelo de arsenal da "
                          f"{a.delta_oficial:+.1f}: en altitud se encoge la separación entre ambos. No leerlo como {'ganancia' if sgn > 0 else 'pérdida'} limpia."
                          if c_var else "Whiff y chase se mueven en sentidos opuestos: el neto esconde un componente que va en contra."),
                "Favorable": "Stuff+, whiff y chase van en la misma dirección.", "Unfavorable": "Stuff+, whiff y chase van en la misma dirección."}[net]
    note = {"Robust": "Modelo y Stuff+ observado del mismo pitcher coinciden en su tipo de pitch.",
            "Conditional": "El Stuff+ observado del mismo pitcher en altitud va en la misma dirección, con más incertidumbre.",
            "Uncertain": "En su tipo de pitch el efecto es pequeño o hay pocas comparaciones del mismo pitcher: tomar como orientación."}[a.evidence]
    if abs(a.delta_oficial - a.delta) >= 0.7 and not (net == "Mixed" and c_var):
        note += f" Con el modelo de arsenal (sensibilidad): {a.delta_oficial:+.1f}."

    if a.delta_observado == a.delta_observado:
        note += f" Este pitcher: observado {a.delta_observado:+.1f}."
    col.markdown(
        f'<div class="dr-card"><div style="display:flex;justify-content:space-between"><span><span class="dr-code">{a.code}</span> <b>{a.AutoPitchType}</b></span>'
        f'{ui.action_dot(a.action)}</div><table class="dr-tbl" style="margin-top:8px"><tr><th></th><th style="text-align:right">Nivel del mar</th><th style="text-align:right">Extreme</th></tr>'
        f'<tr><td>Stuff+</td><td class="dr-mono" style="text-align:right">{a.stuff_mar:.0f}</td><td class="dr-mono" style="text-align:right"><b>{a.stuff_harp:.0f}</b></td></tr>'
        f'<tr><td>IVB</td><td class="dr-mono" style="text-align:right">{na.ivb:.1f} in</td><td class="dr-mono" style="text-align:right"><b>{arrow(ea.ivb - na.ivb, .3)} {ea.ivb:.1f} in</b></td></tr>'
        f'<tr><td>HB</td><td class="dr-mono" style="text-align:right">{na.hb:.1f} in</td><td class="dr-mono" style="text-align:right"><b>{arrow(abs(ea.hb) - abs(na.hb), .3)} {ea.hb:.1f} in</b></td></tr></table>'
        f'<div style="margin:10px 0"><span class="dr-label">Outcomes</span> <span class="dr-pill">Whiff {arrow(dw, .002)} {100 * dw:+.1f} pp</span> '
        f'<span class="dr-pill">Chase {arrow(dc, .002)} {100 * dc:+.1f} pp</span></div>'
        f'<div><span class="dr-label">Net effect</span> <b>{net}</b><div class="dr-note" style="margin-top:4px">{ui.esc(net_note)} {ui.esc(note)}</div></div></div>',
        unsafe_allow_html=True)

# ---------------- Why? ----------------
st.write("")
st.markdown('<span class="dr-h2">Why?</span> <span class="dr-note">&nbsp;Factores principales detrás de cada lectura</span>', unsafe_allow_html=True)
ev_note = {"Robust": "Evidencia Robust: el efecto (variante física) se sostiene entre ventanas, en el chequeo inverso y en el Stuff+ observado del mismo pitcher.",
           "Conditional": "Evidencia Conditional: el efecto va en la misma dirección que lo observado, pero sin confirmarse con certeza en ambas temporadas.",
           "Uncertain": "Evidencia Uncertain: efecto pequeño o inestable para este tipo de pitch, o muestra limitada."}
for _, a in A[A.n >= 50].iterrows():
    if not has_projection(a):
        continue
    g = fac[fac.AutoPitchType == a.AutoPitchType].set_index("env")
    title = f"{a.AutoPitchType} · {a.action} · Δ {a.delta:+.1f} Stuff+ · {a.evidence}"
    with st.expander(title, expanded=(a.AutoPitchType == (focus or rec_pitch))):
        f1 = f"IVB {g.loc['No Altitude'].ivb:.1f} → {g.loc['Extreme Altitude'].ivb:.1f} in y HB {g.loc['No Altitude'].hb:.1f} → {g.loc['Extreme Altitude'].hb:.1f} in en Extreme"
        f2 = f"Whiff por pitch modelado {100 * a.whiff_mar:.1f}% → {100 * a.whiff_harp:.1f}%"
        f3 = f"Chase modelado {100 * a.chase_mar:.1f}% → {100 * a.chase_harp:.1f}%"
        f4 = (f"Stuff+ observado del pitcher: Extreme − No Altitude = {a.delta_observado:+.1f}" if a.delta_observado == a.delta_observado
              else "Sin suficientes pitches en ambos entornos para el contraste observado")
        f4 += f" · con el modelo de arsenal el efecto sería {a.delta_oficial:+.1f}"
        st.markdown(
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px 24px">'
            + "".join(f'<div><span class="dr-pill">{s}</span> {ui.esc(t)}</div>' for s, t in [("↓", f1), ("↓" if a.whiff_harp < a.whiff_mar else "↑", f2),
                                                                                            ("↓" if a.chase_harp < a.chase_mar else "↑", f3), ("≈", f4)])
            + f'</div><div class="dr-note" style="margin-top:10px">{ev_note[a.evidence]} n = {int(a.n):,} pitches en {year}.</div>', unsafe_allow_html=True)

# ---------------- Development Targets ----------------
st.write("")
TG = data.load("targets"); TG = TG[(TG.year == year) & (TG.pitcher_anon_id == pid)].sort_values(["verb", "AutoPitchType"])
rows = "".join(
    f'<tr><td><span class="dr-code">{t.code}</span> <b>{t.variable}</b></td><td class="dr-mono">{t.baseline:.1f} {t.unit}'
    + ('' if t.estado == "movimiento" else '<div class="dr-note">todos los parques</div>') + '</td>'
    + (f'<td class="dr-mono"><b>{t.harp:.1f} {t.unit}</b></td>' if t.estado == "movimiento" else '<td class="dr-note">no cambia con el parque</td>') +
    f'<td><span class="dr-note">{t.verb.upper()}</span> <b class="dr-mono">{t.zone_lo:.1f}–{t.zone_hi:.1f} {t.unit}</b></td>'
    f'<td>{ui.esc(t.direction)}</td><td>{ui.conf_bars(t.confidence)}</td></tr>' for t in TG.itertuples())
st.markdown(
    f'<span class="dr-h2">Development Targets</span> <span class="dr-note">&nbsp;Zonas de los pitches de Stuff+ alto lanzados en Extreme · exploratorias, no recetas causales</span>'
    f'<div class="dr-card" style="padding:0;margin-top:10px"><table class="dr-tbl"><tr><th>Pitch · variable</th><th>Baseline (No Altitude)</th><th>En Harp Helú</th><th>Target en Harp Helú</th>'
    f'<th>Expected direction</th><th>Confianza</th></tr>{rows or "<tr><td colspan=6 class=dr-note>Sin rasgos del perfil para sus tipos de pitch.</td></tr>"}</table>'
    f'<div class="dr-note" style="padding:10px 14px">Baseline = el mismo pitch a nivel del mar (igual que el Arsenal Fingerprint). El target se midió en pitches '
    f'lanzados en altitud, así que el movimiento (IVB, HB) se compara con su valor en Harp Helú. Velocidad, spin y release no cambian con el parque.</div></div>',
    unsafe_allow_html=True)
st.write("")
st.markdown(f'<div class="dr-card" style="display:flex;justify-content:space-between;align-items:center"><div><div class="dr-label">Siguiente paso</div>'
            f'<div style="font:700 18px IBM Plex Sans;margin-top:6px">Probar los ajustes del {worst.AutoPitchType} en contexto Extreme Altitude</div></div>'
            f'<a class="dr-btn" style="background:{ui.ACCENT};color:#fff !important" target="_self" href="simulator?pid={pid}&season={year}&pitch={worst.AutoPitchType}">'
            f'Open in What-if Simulator →</a></div>', unsafe_allow_html=True)
