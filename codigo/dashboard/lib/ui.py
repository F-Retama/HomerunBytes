"""Sistema visual del Pitching Decision System (replica los wireframes de diseño)."""

import html

import plotly.graph_objects as go
import streamlit as st

# Paleta Diablos Rojos (rediseño nov-2026): rojo escarlata, negro y guinda sobre la base crema del uniforme de local.
# Lo que cae más de lo esperado va en rojo; lo que cae menos, en negro.
RED, YEL, GRN = "#E31E24", "#792527", "#111111"
ACCENT, BG, PANEL, PANEL2 = "#E31E24", "#F5EFE3", "#FFFFFF", "#F5EFE3"
TEXT, MUTED, DIM, LINE = "#111111", "#5E5A54", "#4D4A45", "#E6DDCB"
TEXT_RED, MAROON, SAND, TRACK, EXPECTED = "#C8161D", "#792527", "#CFC4AE", "#EDE5D5", "#A39E94"
ALERT = {"red": (RED, "Needs attention"), "yellow": (YEL, "Review"), "green": (EXPECTED, "Maintain")}
ACTION = {"Adjust": RED, "Review": YEL, "Maintain": DIM}
FIT_STYLE = {"Strong Fit": ("#111111", "#111111"), "Moderate Fit": ("#5E5A54", "#4D4A45"),
             "Low Fit": ("#CFC4AE", "#5E5A54"), "Poor Fit": ("#E31E24", "#C8161D"),
             "Sin muestra": ("#E6DDCB", "#5E5A54")}
CONF_LEVEL = {"Robust": 3, "Conditional": 2, "Uncertain": 1}
CODE = {"Four-Seam": "FF", "Sinker": "SI", "Cutter": "FC", "Slider": "SL", "Curveball": "CU", "Changeup": "CH", "Splitter": "FS"}
PITCH_ES = {"Four-Seam": "Recta", "Sinker": "Sinker", "Cutter": "Cutter", "Slider": "Slider", "Curveball": "Curva", "Changeup": "Cambio", "Splitter": "Splitter"}

CSS = f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap">
<style>
html, body, [class*="css"], .stApp {{ font-family: 'Inter', system-ui, sans-serif; color: {TEXT}; }}
.stApp {{ background: {BG}; }}
header[data-testid="stHeader"] {{ display: none; }}
.block-container {{ padding-top: 0 !important; max-width: 1360px; }}
div[data-testid="stToolbar"] {{ display: none; }}
.dr-topline {{ height: 4px; background: {ACCENT}; margin: 0 -5rem; }}
.dr-eyebrow {{ font: 600 13px 'Inter'; color: {TEXT_RED}; margin: 30px 0 6px; }}
.dr-h1 {{ font: 700 40px/1.1 'Space Grotesk', sans-serif; color: {TEXT}; margin: 0 0 8px; max-width: 900px; }}
.dr-h2 {{ font: 700 22px/1.2 'Space Grotesk', sans-serif; color: {TEXT}; margin: 0; }}
.dr-sub {{ font: 400 16px/1.55 'Inter'; color: {DIM}; margin-bottom: 18px; max-width: 820px; }}
.dr-mono {{ font-family: 'Inter', sans-serif; font-variant-numeric: tabular-nums; }}
.dr-label {{ font: 600 12px 'Inter'; letter-spacing: .02em; color: {MUTED}; }}
.dr-card {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 14px; padding: 18px 20px; }}
.dr-card-dark {{ background: #111111; color: #FFFFFF; border-radius: 14px; padding: 24px 26px; }}
.dr-card-dark .dr-label {{ color: #F59A9D; }}
.dr-kpi-v {{ font: 700 34px/1 'Space Grotesk', sans-serif; color: {TEXT}; }}
.dr-kpi-s {{ font: 400 13px 'Inter'; color: {DIM}; }}
.dr-badge {{ display: inline-flex; align-items: center; gap: 6px; font: 600 12px 'Inter'; padding: 4px 10px; border-radius: 999px; }}
.dr-pill {{ display: inline-flex; align-items: center; gap: 6px; font: 500 12px 'Inter'; padding: 3px 8px; border-radius: 6px;
            border: 1px solid {LINE}; background: {PANEL2}; color: {TEXT}; }}
.dr-code {{ font: 600 10.5px 'Inter'; padding: 1px 5px; border-radius: 4px; background: #111111; color: #FFFFFF; }}
.dr-tag {{ font: 600 9.5px 'Inter'; padding: 1px 5px; border-radius: 3px; border: 1px solid {SAND}; color: {MUTED}; }}
.dr-fit {{ font: 600 11.5px 'Inter'; padding: 2px 9px; border-radius: 999px; white-space: nowrap; }}
.dr-tbl {{ width: 100%; border-collapse: collapse; font-size: 13.5px; }}
.dr-tbl th {{ font: 600 11.5px 'Inter'; color: {MUTED}; text-align: left; padding: 10px 10px; border-bottom: 1px solid {LINE}; }}
.dr-tbl td {{ padding: 11px 10px; border-bottom: 1px solid #EEE6D6; color: {TEXT}; vertical-align: middle; }}
.dr-tbl tr:hover td {{ background: {PANEL2}; }}
.dr-tbl a {{ color: {TEXT}; text-decoration: none; }}
.dr-btn {{ display: inline-block; font: 600 13px 'Inter'; padding: 10px 16px; border-radius: 10px; background: {ACCENT}; color: #FFFFFF !important;
          text-decoration: none !important; white-space: nowrap; }}
.dr-btn-ghost {{ display: inline-block; font: 600 12.5px 'Inter'; padding: 8px 12px; border-radius: 10px; border: 1px solid {SAND};
                color: {TEXT} !important; text-decoration: none !important; background: {PANEL}; }}
.dr-env {{ display: inline-flex; gap: 10px; align-items: center; padding: 7px 12px; border: 1px solid {LINE}; border-radius: 10px; background: {PANEL}; }}
.dr-note {{ font: 400 12.5px/1.5 'Inter'; color: {MUTED}; }}
.dr-more {{ display: flex; align-items: center; gap: 16px; margin: 40px 0 18px; }}
.dr-more span:first-child {{ font: 600 13px 'Inter'; color: {MUTED}; white-space: nowrap; }}
.dr-more span:last-child {{ flex: 1; height: 1px; background: {SAND}; }}
.dr-legend {{ display: inline-flex; align-items: center; gap: 6px; font: 400 12.5px 'Inter'; color: {DIM}; margin-right: 14px; }}
.dr-legend i {{ width: 14px; height: 10px; border-radius: 3px; display: inline-block; }}
div[data-testid="stPageLink"] a {{ background: transparent; }}
div[data-testid="stPageLink"] p {{ font: 600 14px 'Inter' !important; color: {DIM}; }}
details summary p {{ font-weight: 600; }}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def esc(x):
    return html.escape(str(x))


def pid_label(pid):
    return "Pitcher " + str(pid).split("_")[-1]


def alert_badge(alert):
    c, label = ALERT[alert]
    return (f'<span class="dr-badge" style="background:{c}24;color:{c}"><span style="width:6px;height:6px;border-radius:50%;'
            f'background:{c}"></span>{label}</span>')


def type_badge(kind):
    c = {"Needs attention": RED, "Review": YEL, "Opportunity": GRN}[kind]
    return (f'<span class="dr-badge" style="background:{c}24;color:{c}"><span style="width:6px;height:6px;border-radius:50%;'
            f'background:{c}"></span>{kind}</span>')


def fit_pill(fit):
    b, c = FIT_STYLE.get(fit, FIT_STYLE["Sin muestra"])
    return f'<span class="dr-fit" style="border:1px solid {b};color:{c}">{esc(fit)}</span>'


def conf_bars(conf, with_label=True):
    lv = CONF_LEVEL.get(conf, 1)
    bars = "".join(f'<span style="display:inline-block;width:14px;height:3px;border-radius:2px;margin-right:2px;'
                   f'background:{"#111111" if lv >= k else "#E6DDCB"}"></span>' for k in (1, 2, 3))
    lab = f' <span style="font:600 12px Inter;color:{TEXT}">{conf}</span> <span class="dr-note">· altitud</span>' if with_label else ""
    return f'<span style="white-space:nowrap">{bars}{lab}</span>'


def action_dot(action):
    c = ACTION[action]
    return f'<span style="color:{c};font-weight:600;white-space:nowrap">● {action}</span>'


def delta_fmt(d, unit=""):
    if d != d:  # NaN
        return '<span class="dr-note">—</span>'
    arrow = "▲" if d > 0.05 else "▼" if d < -0.05 else "–"
    c = GRN if d > 0.05 else (RED if d <= -3.5 else YEL if d <= -2 else DIM)
    return f'<span class="dr-mono" style="color:{c}">{arrow} {d:+.1f}{unit}</span>'


def sparkline(vals, w=96, h=26):
    vals = [v for v in vals if v == v]
    if len(vals) < 2:
        return '<span class="dr-note">—</span>'
    lo, hi = min(vals), max(vals); rg = max(hi - lo, 4)
    pts = " ".join(f"{i * (w - 4) / (len(vals) - 1) + 2:.1f},{h - 3 - (v - lo) / rg * (h - 6):.1f}" for i, v in enumerate(vals))
    c = GRN if vals[-1] >= vals[0] + 1 else RED if vals[-1] <= vals[0] - 1 else DIM
    return f'<svg width="{w}" height="{h}"><polyline points="{pts}" fill="none" stroke="{c}" stroke-width="1.6"/></svg>'


def env_badge():
    return (f'<div class="dr-env" title="El modelo utiliza la categoría Extreme Altitude del dataset como proxy del entorno Harp Helú. '
            f'Esa categoría agrupa varios parques locales de gran altitud.">'
            f'<span style="color:{ACCENT}">◆</span><span><span style="font:600 13px Inter;color:{TEXT}">Extreme Altitude</span><br>'
            f'<span class="dr-mono" style="font-size:11px;color:{MUTED}">Proxy Harp Helú · CDMX</span></span>'
            f'<span style="border:1px solid {LINE};border-radius:50%;width:18px;height:18px;display:inline-flex;align-items:center;'
            f'justify-content:center;font-size:11px;color:{MUTED}">i</span></div>')


PAGES = [("Inicio", "views/inicio.py"), ("Scouting", "views/scouting.py"), ("Game", "views/game.py"),
         ("Simulator", "views/simulator.py"), ("Roster", "views/roster.py"), ("Player", "views/player.py")]


def header(active):
    """Navegación en el orden de la historia (Inicio, Scouting, Game, Simulator) y, aparte, el seguimiento semanal."""
    st.markdown('<div class="dr-topline"></div>', unsafe_allow_html=True)
    c = st.columns([3, 0.8, 0.95, 0.8, 1.0, 0.15, 0.85, 0.8], vertical_alignment="center")
    c[0].markdown(
        f'<div style="display:flex;align-items:center;gap:12px;padding:14px 0"><div style="width:34px;height:34px;border-radius:8px;'
        f'background:{ACCENT};display:flex;align-items:center;justify-content:center;font:700 15px/1 Space Grotesk;color:#fff">DR</div>'
        f'<div><div style="font:600 15px/1.1 Space Grotesk;color:{TEXT}">Pitching Decision System</div>'
        f'<div style="font:400 12px Inter;color:{MUTED}">Diablos Rojos · Harp Helú · 2,240 m</div></div></div>', unsafe_allow_html=True)
    slots = [1, 2, 3, 4, 6, 7]
    c[5].markdown(f'<div style="width:1px;height:24px;background:{SAND};margin:auto"></div>', unsafe_allow_html=True)
    for slot, (name, path) in zip(slots, PAGES):
        with c[slot]:
            st.page_link(path, label=name)
            if name == active:
                st.markdown(f'<div style="height:3px;background:{ACCENT};margin-top:-6px;border-radius:2px"></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="height:1px;background:{LINE};margin:0 -5rem 0"></div>', unsafe_allow_html=True)


def title(eyebrow, h1, sub=""):
    html_block(f'<div class="dr-eyebrow">{esc(eyebrow)}</div><div class="dr-h1">{esc(h1)}</div>'
               + (f'<div class="dr-sub">{esc(sub)}</div>' if sub else ""))


def mas_detalle(texto="Más detalle"):
    """Separador entre lo importante (arriba) y el resto de las herramientas (abajo)."""
    html_block(f'<div class="dr-more"><span>{esc(texto)}</span><span></span></div>')


def brecha(esperada, real, scale=None, dark=False, compact=False):
    """El diferenciador: caída esperada (gris) contra caída real (rojo) del Stuff+ en Harp Helú, y el exceso.

    esperada = delta - exceso, la misma regla de 23_dashboard_data.py (residuo después de controlar por el
    Stuff+ a nivel del mar dentro de cada tipo). Valores negativos = pierde."""
    if esperada != esperada or real != real:
        return '<span class="dr-note">Sin proyección en Harp Helú</span>'
    esperada, real = round(esperada, 1) + 0.0, round(real, 1) + 0.0  # sin "-0.0"
    exceso = round(real - esperada, 1) + 0.0
    scale = scale or max(8.0, abs(esperada), abs(real))
    track = "#2A2A2A" if dark else TRACK
    lab = "#DADADA" if dark else DIM
    h = 10 if compact else 16

    def bar(v, color):
        w = min(abs(v) / scale * 100, 100)
        return (f'<div style="flex:1;height:{h}px;background:{track};border-radius:4px;position:relative">'
                f'<div style="position:absolute;left:0;top:0;bottom:0;width:{w:.0f}%;background:{color};border-radius:4px"></div></div>')
    num = "16px" if compact else "20px"
    ex_c = (RED if exceso <= -1 else (TEXT if not dark else "#FFFFFF"))
    rows = (f'<div style="display:flex;align-items:center;gap:12px"><span style="width:112px;font:500 13px Inter;color:{lab}">Caída esperada</span>'
            f'{bar(esperada, EXPECTED)}<span class="dr-mono" style="width:52px;text-align:right;font:700 {num} Space Grotesk;color:{lab}">{esperada:+.1f}</span></div>'
            f'<div style="display:flex;align-items:center;gap:12px;margin-top:8px"><span style="width:112px;font:500 13px Inter;color:{lab}">Caída real</span>'
            f'{bar(real, RED)}<span class="dr-mono" style="width:52px;text-align:right;font:700 {num} Space Grotesk;color:{"#FFFFFF" if dark else TEXT_RED}">{real:+.1f}</span></div>')
    if compact:
        return rows
    verdict = ("cae más de lo esperado" if exceso <= -1 else "cae menos de lo esperado" if exceso >= 0.5 else "cae como se espera")
    return (rows + f'<div style="display:flex;align-items:baseline;gap:10px;margin-top:12px;padding-top:12px;border-top:1px solid {"#3A3A3A" if dark else LINE}">'
            f'<span style="font:700 32px Space Grotesk;color:{ex_c}">{exceso:+.1f}</span>'
            f'<span style="font:400 14px Inter;color:{lab}">contra lo esperado · {verdict}</span></div>')


def brecha_legend():
    return (f'<span class="dr-legend"><i style="background:{EXPECTED}"></i>Caída esperada para su tipo y nivel</span>'
            f'<span class="dr-legend"><i style="background:{RED}"></i>Caída real</span>')


def html_block(s):
    st.markdown(s, unsafe_allow_html=True)


GROUP_COLOR = {"Más efectivos": "#111111", "Menos efectivos": "#E31E24", "Resto": "rgba(17,17,17,0.12)"}


def movement_traces(fig, d, env="harp", size=8, legend=True):
    """Agrega a fig los pitches de la liga (una marca por pitcher x tipo) coloreados por efectividad."""
    for grp in ["Resto", "Más efectivos", "Menos efectivos"]:
        g = d[d.grupo == grp]
        if not len(g):
            continue
        fig.add_trace(go.Scatter(
            x=g[f"hb_{env}"], y=g[f"ivb_{env}"], mode="markers", name=f"{grp} (liga)" if grp != "Resto" else "Resto de la liga",
            showlegend=legend, marker=dict(size=size if grp != "Resto" else size - 3, color=GROUP_COLOR[grp], opacity=0.85 if grp != "Resto" else 1,
                                           symbol="circle" if grp != "Menos efectivos" else "x"),
            customdata=list(zip(g.pitcher_anon_id.str.replace("pitcher_", ""), g.AutoPitchType, g.stuff_harp, g.velo, g.n, 100 * g.obs_whiff)),
            hovertemplate="Pitcher %{customdata[0]} · %{customdata[1]}<br>Whiff real %{customdata[5]:.1f}% · Stuff+ Harp Helú %{customdata[2]:.1f}<br>"
                          "HB %{x:.1f} in · IVB %{y:.1f} in · %{customdata[3]:.1f} mph<br>n %{customdata[4]}<extra></extra>"))
    return fig


def movement_layout(fig, height=420):
    fig.add_hline(y=0, line=dict(color="rgba(17,17,17,0.15)")); fig.add_vline(x=0, line=dict(color="rgba(17,17,17,0.15)"))
    fig.update_layout(height=height, margin=dict(l=40, r=10, t=10, b=40), paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                      font=dict(family="Inter", color=DIM, size=11), legend=dict(font=dict(size=10), orientation="h", y=1.08, x=0),
                      xaxis=dict(title="HB (in) · + lado del brazo (convención de diestro)", gridcolor="rgba(17,17,17,0.05)", zeroline=False),
                      yaxis=dict(title="IVB (in)", gridcolor="rgba(17,17,17,0.05)", zeroline=False))
    return fig
