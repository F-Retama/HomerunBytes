"""Sistema visual del Pitching Decision System (replica los wireframes de diseño)."""

import html

import plotly.graph_objects as go
import streamlit as st

RED, YEL, GRN = "#f0645a", "#f2c14e", "#4fc98a"
ACCENT, BG, PANEL, PANEL2 = "#d4262f", "#101114", "#17181c", "#1c1d22"
TEXT, MUTED, DIM, LINE = "#eeede9", "#8d9097", "#a2a4ab", "rgba(255,255,255,0.07)"
ALERT = {"red": (RED, "Needs attention"), "yellow": (YEL, "Review"), "green": (GRN, "Maintain")}
ACTION = {"Adjust": RED, "Review": YEL, "Maintain": GRN}
FIT_STYLE = {"Strong Fit": ("rgba(255,255,255,0.6)", "#f1efea"), "Moderate Fit": ("rgba(255,255,255,0.32)", "#d4d5d9"),
             "Low Fit": ("rgba(255,255,255,0.18)", "#a2a4ab"), "Poor Fit": ("rgba(255,255,255,0.1)", "#8d9097"),
             "Sin muestra": ("rgba(255,255,255,0.08)", "#6b6e75")}
CONF_LEVEL = {"Robust": 3, "Conditional": 2, "Uncertain": 1}
CODE = {"Four-Seam": "FF", "Sinker": "SI", "Cutter": "FC", "Slider": "SL", "Curveball": "CU", "Changeup": "CH", "Splitter": "FS"}
PITCH_ES = {"Four-Seam": "Four-Seam", "Sinker": "Sinker", "Cutter": "Cutter", "Slider": "Slider", "Curveball": "Curveball", "Changeup": "Changeup", "Splitter": "Splitter"}

CSS = f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
html, body, [class*="css"], .stApp {{ font-family: 'IBM Plex Sans', system-ui, sans-serif; }}
.stApp {{ background: {BG}; }}
header[data-testid="stHeader"] {{ display: none; }}
.block-container {{ padding-top: 0 !important; max-width: 1440px; }}
div[data-testid="stToolbar"] {{ display: none; }}
.dr-topline {{ height: 3px; background: {ACCENT}; margin: 0 -5rem; }}
.dr-h1 {{ font: 700 44px/1.05 'Barlow Condensed', sans-serif; color: {TEXT}; margin: 26px 0 4px; letter-spacing: .005em; }}
.dr-h2 {{ font: 700 24px/1.1 'Barlow Condensed', sans-serif; color: {TEXT}; margin: 0; }}
.dr-sub {{ font: 400 14px 'IBM Plex Sans'; color: {DIM}; margin-bottom: 14px; }}
.dr-mono {{ font-family: 'IBM Plex Mono', ui-monospace, monospace; }}
.dr-label {{ font: 600 10.5px 'IBM Plex Sans'; letter-spacing: .09em; text-transform: uppercase; color: {MUTED}; }}
.dr-card {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 10px; padding: 16px 18px; }}
.dr-kpi-v {{ font: 700 34px/1 'Barlow Condensed', sans-serif; color: {TEXT}; }}
.dr-kpi-s {{ font: 400 12px 'IBM Plex Sans'; color: {MUTED}; }}
.dr-badge {{ display: inline-flex; align-items: center; gap: 6px; font: 600 11px 'IBM Plex Sans'; padding: 3px 9px; border-radius: 999px; }}
.dr-pill {{ display: inline-flex; align-items: center; gap: 6px; font: 500 12px 'IBM Plex Sans'; padding: 3px 8px; border-radius: 6px;
            border: 1px solid {LINE}; background: {PANEL2}; color: {TEXT}; }}
.dr-code {{ font: 600 10.5px 'IBM Plex Mono'; padding: 1px 5px; border-radius: 4px; background: #e9e7e2; color: #111; }}
.dr-tag {{ font: 600 9.5px 'IBM Plex Mono'; padding: 1px 5px; border-radius: 3px; border: 1px solid rgba(255,255,255,.18); color: {MUTED}; }}
.dr-fit {{ font: 500 11.5px 'IBM Plex Sans'; padding: 2px 9px; border-radius: 999px; white-space: nowrap; }}
.dr-tbl {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
.dr-tbl th {{ font: 600 10.5px 'IBM Plex Sans'; letter-spacing: .08em; text-transform: uppercase; color: {MUTED}; text-align: left;
             padding: 10px 10px; border-bottom: 1px solid {LINE}; }}
.dr-tbl td {{ padding: 11px 10px; border-bottom: 1px solid {LINE}; color: {TEXT}; vertical-align: middle; }}
.dr-tbl tr:hover td {{ background: rgba(255,255,255,0.025); }}
.dr-tbl a {{ color: {TEXT}; text-decoration: none; }}
.dr-btn {{ display: inline-block; font: 600 12.5px 'IBM Plex Sans'; padding: 7px 13px; border-radius: 7px; background: #eeede9; color: #111 !important;
          text-decoration: none !important; white-space: nowrap; }}
.dr-btn-ghost {{ display: inline-block; font: 600 12px 'IBM Plex Sans'; padding: 6px 11px; border-radius: 7px; border: 1px solid rgba(255,255,255,.18);
                color: {TEXT} !important; text-decoration: none !important; }}
.dr-env {{ display: inline-flex; gap: 10px; align-items: center; padding: 7px 12px; border: 1px solid {LINE}; border-radius: 8px; background: {PANEL}; }}
.dr-note {{ font: 400 11.5px 'IBM Plex Sans'; color: {MUTED}; }}
div[data-testid="stPageLink"] a {{ background: transparent; }}
div[data-testid="stPageLink"] p {{ font: 600 15px 'IBM Plex Sans' !important; color: {DIM}; }}
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
                   f'background:{"#c6c8cd" if lv >= k else "#3a3c42"}"></span>' for k in (1, 2, 3))
    lab = f' <span style="font:600 12px IBM Plex Sans;color:{TEXT}">{conf}</span> <span class="dr-note">evidence</span>' if with_label else ""
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
            f'<span style="color:{ACCENT}">◆</span><span><span style="font:600 13px IBM Plex Sans;color:{TEXT}">Extreme Altitude</span><br>'
            f'<span class="dr-mono" style="font-size:11px;color:{MUTED}">Proxy Harp Helú · CDMX</span></span>'
            f'<span style="border:1px solid {LINE};border-radius:50%;width:18px;height:18px;display:inline-flex;align-items:center;'
            f'justify-content:center;font-size:11px;color:{MUTED}">i</span></div>')


def header(active):
    st.markdown('<div class="dr-topline"></div>', unsafe_allow_html=True)
    c = st.columns([3.2, 0.9, 0.9, 1.1, 1.1, 0.9, 2.2], vertical_alignment="center")
    c[0].markdown(
        f'<div style="display:flex;align-items:center;gap:12px;padding:12px 0"><div style="width:30px;height:30px;border-radius:6px;'
        f'background:{ACCENT};display:flex;align-items:center;justify-content:center;font:700 15px/1 Barlow Condensed;color:#fff">DR</div>'
        f'<div><div style="font:600 14px/1.1 IBM Plex Sans;color:{TEXT}">Pitching Decision System</div>'
        f'<div style="font:400 11px IBM Plex Sans;color:{MUTED}">Diablos Rojos del México</div></div></div>', unsafe_allow_html=True)
    pages = [("Roster", "views/roster.py"), ("Player", "views/player.py"), ("Simulator", "views/simulator.py"),
             ("Scouting", "views/scouting.py"), ("Game", "views/game.py")]
    for i, (name, path) in enumerate(pages, start=1):
        with c[i]:
            st.page_link(path, label=name)
            if name == active:
                st.markdown(f'<div style="height:2px;background:{ACCENT};margin-top:-6px"></div>', unsafe_allow_html=True)
    c[6].markdown(f'<div style="text-align:right;font:400 12px IBM Plex Sans;color:{MUTED}">Pitching coach '
                  f'<span style="display:inline-flex;width:28px;height:28px;border-radius:50%;background:#2c2e34;color:{TEXT};'
                  f'align-items:center;justify-content:center;font:600 11px IBM Plex Sans;margin-left:8px">PC</span></div>',
                  unsafe_allow_html=True)
    st.markdown(f'<div style="height:1px;background:{LINE};margin:0 -5rem 0"></div>', unsafe_allow_html=True)


def html_block(s):
    st.markdown(s, unsafe_allow_html=True)


GROUP_COLOR = {"Más efectivos": "#5aa9e6", "Menos efectivos": "#e8963c", "Resto": "rgba(255,255,255,0.13)"}


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
            customdata=list(zip(g.pitcher_anon_id.str.replace("pitcher_", ""), g.AutoPitchType, g.stuff_harp, g.velo, g.n)),
            hovertemplate="Pitcher %{customdata[0]} · %{customdata[1]}<br>Stuff+ Harp Helú %{customdata[2]:.1f}<br>"
                          "HB %{x:.1f} in · IVB %{y:.1f} in · %{customdata[3]:.1f} mph<br>n %{customdata[4]}<extra></extra>"))
    return fig


def movement_layout(fig, height=420):
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.15)")); fig.add_vline(x=0, line=dict(color="rgba(255,255,255,0.15)"))
    fig.update_layout(height=height, margin=dict(l=40, r=10, t=10, b=40), paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                      font=dict(family="IBM Plex Sans", color=DIM, size=11), legend=dict(font=dict(size=10), orientation="h", y=1.08, x=0),
                      xaxis=dict(title="HB (in) · + lado del brazo (convención de diestro)", gridcolor="rgba(255,255,255,0.05)", zeroline=False),
                      yaxis=dict(title="IVB (in)", gridcolor="rgba(255,255,255,0.05)", zeroline=False))
    return fig
