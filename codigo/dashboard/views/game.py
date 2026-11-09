"""Game — lectura táctica rápida: arsenal por count y lado del bateador, con heatmaps de calidad y de daño."""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dashboard.lib import data, engine, ui  # noqa: E402

ui.inject_css()
ui.header("Game")
if not data.data_ready():
    st.error("Faltan las tablas del dashboard. Corre `python 23_dashboard_data.py` desde la raíz del repo.")
    st.stop()


state, qp = data.read_state(), st.query_params
P, A = data.load("pitchers"), data.load("arsenal")
year = 2026  # Game usa el modelo vigente (ventana 2024-25 -> 2026)
roster = data.roster_ids(year, state.get("roster_source", "Liga completa"))
pool = P[(P.year == year) & (P.n >= 100) & P.stuff_harp.notna()].copy()  # solo pitchers con pitches proyectables
pool["in_roster"] = pool.pitcher_anon_id.isin(roster)
pool = pool.sort_values(["in_roster", "n"], ascending=[True, False])  # reporte del rival: primero los de otros equipos
ids = pool.pitcher_anon_id.tolist()
pid0 = qp.get("pid", state.get("last_pid"))

ui.title("Game · reporte del rival", "Qué esperar del pitcher que viene al Harp Helú",
         "El mismo perfil que usamos para reclutar, leído desde el lado del bateador: qué pitches del rival se debilitan aquí "
         "y dónde atacarlo en cada count.")
c = st.columns([1.6, 1.1, 1.3, 1.1, 0.9, 1.4], vertical_alignment="bottom")
pid = c[0].selectbox("Pitcher", ids, index=ids.index(pid0) if pid0 in ids else 0,
                     format_func=lambda x: ui.pid_label(x) + ("  · nuestro roster" if x in set(roster) else ""))
bside = c[1].segmented_control("Bateador", ["Right", "Left"], default="Right", format_func=lambda s: "Derecho" if s == "Right" else "Zurdo")
balls = c[2].segmented_control("Bolas", [0, 1, 2, 3], default=0)
strikes = c[3].segmented_control("Strikes", [0, 1, 2], default=0)
outs = c[4].segmented_control("Outs", [0, 1, 2], default=0)
env = c[5].segmented_control("Entorno", ["Extreme Altitude", "No Altitude"], default="Extreme Altitude",
                             format_func=lambda e: "Harp Helú" if e == "Extreme Altitude" else "Nivel del mar")
bside, balls, strikes, outs, env = bside or "Right", balls or 0, strikes or 0, outs or 0, env or "Extreme Altitude"
ctx = (("BatterSide", bside), ("Balls", balls), ("Strikes", strikes), ("Outs", outs))
ars = A[(A.year == year) & (A.pitcher_anon_id == pid) & (A.n >= 30) & (A.usage >= 0.05)].sort_values("usage", ascending=False)
pp = pool.set_index("pitcher_anon_id").loc[pid]

# ---------------- reporte para el bateador (tablas de 23, sin motor) ----------------
# Diferenciador: cada pitch del rival contra la caída esperada para su tipo y nivel (esperada = delta - exceso).
R = ars[ars.delta.notna()].assign(esperada=lambda d: d.delta - d.exceso)
tips = []
falls = R[R.exceso <= -1].sort_values("usage", ascending=False)
if len(falls):
    f0 = falls.iloc[0]
    tips.append((f"Su {ui.PITCH_ES[f0.AutoPitchType].lower()} se cae aquí",
                 f"Pierde {abs(f0.delta):.1f} puntos, {abs(f0.exceso):.1f} más de lo esperado para su nivel. Es {f0.usage:.0%} de lo que lanza: búscalo."))
if len(falls) > 1:
    f1 = falls.iloc[1]
    tips.append((f"Su {ui.PITCH_ES[f1.AutoPitchType].lower()} también cae de más",
                 f"Pierde {abs(f1.delta):.1f} cuando lo esperado es {abs(f1.esperada):.1f}. Es {f1.usage:.0%} de su arsenal."))
hold = R[(R.delta > -1) & (R.usage >= 0.1)].sort_values("stuff_harp", ascending=False)
if len(hold):
    h0 = hold.iloc[0]
    tips.append((f"Su {ui.PITCH_ES[h0.AutoPitchType].lower()} no cambia",
                 f"Se queda en {h0.stuff_harp:.1f} de Stuff+ aquí y es {h0.usage:.0%} de su arsenal: será su recurso con dos strikes."))
if not tips:
    tips.append(("Sin cambios fuertes en altitud", "Ningún pitch de su arsenal cae de más en el Harp Helú: prepárate como a nivel del mar."))
tip_html = "".join(f'<div style="flex:1 1 220px;background:#2A2A2A;border-radius:12px;padding:16px">'
                   f'<div style="font:700 16px Space Grotesk;color:#fff">{ui.esc(a)}</div>'
                   f'<div style="font:400 14px/1.5 Inter;color:#DADADA;margin-top:6px">{ui.esc(b)}</div></div>' for a, b in tips)
ui.html_block(f'<div class="dr-card-dark" style="display:flex;flex-wrap:wrap;gap:24px;margin-top:8px">'
              f'<div style="flex:1 1 240px"><div class="dr-label">Para el bateador</div>'
              f'<div style="font:700 24px/1.2 Space Grotesk;margin-top:8px">{ui.pid_label(pid)}</div>'
              f'<div style="font:400 14px Inter;color:#DADADA;margin-top:6px">{pp.role} · {pp.hand} · {ui.esc(pp.team_label) if pp.team_label == pp.team_label else "equipo sin etiqueta"}</div></div>'
              f'<div style="flex:3 1 520px;display:flex;flex-wrap:wrap;gap:12px">{tip_html}</div></div>')

ui.html_block(f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin:28px 0 10px">'
              f'<span class="dr-h2">Su arsenal en el Harp Helú, contra lo esperado</span><span>{ui.brecha_legend()}</span></div>')
acols = st.columns(max(min(len(R), 4), 1))
for col, r in zip(acols * 2, R.head(8).itertuples()):
    lab, bg, fg = (("Atacable aquí", "#FCE4E4", ui.MAROON) if r.exceso <= -1 else
                   ("No cambia", "#EDE5D5", ui.TEXT) if r.delta > -1 else
                   ("Cae menos de lo esperado", "#EDE5D5", ui.TEXT) if r.exceso >= 0.5 else ("Como se espera", "#EDE5D5", ui.DIM))
    col.markdown(
        f'<div class="dr-card" style="border:{"2px solid " + ui.RED if r.exceso <= -1 else "1px solid " + ui.LINE};margin-bottom:12px">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;gap:8px"><b style="font:700 18px Space Grotesk">'
        f'{ui.PITCH_ES[r.AutoPitchType]}</b><span class="dr-badge" style="background:{bg};color:{fg}">{lab}</span></div>'
        f'<div class="dr-note" style="margin:4px 0 12px">{r.usage:.0%} de uso · Stuff+ {r.stuff_mar:.1f} → {r.stuff_harp:.1f}</div>'
        f'{ui.brecha(r.esperada, r.delta, scale=10, compact=True)}'
        f'<div style="display:flex;justify-content:space-between;margin-top:10px;font:600 13px Inter"><span class="dr-note">Abanicada ≈ '
        f'{r.whiff_mar:.0%} → {r.whiff_harp:.0%}</span><span style="color:{ui.TEXT_RED if r.exceso <= -1 else ui.TEXT}">'
        f'{r.exceso:+.1f} vs. esperado</span></div></div>', unsafe_allow_html=True)
if R.empty:
    st.info("Este pitcher no tiene pitches con proyección en Harp Helú.")

ui.mas_detalle("Más detalle · calidad por count y zona")


@st.cache_resource(show_spinner="Cargando modelos…")
def eng(version):
    import importlib
    importlib.reload(engine)
    return engine.Engine()


try:
    E = eng(Path(engine.__file__).stat().st_mtime)
except FileNotFoundError:
    st.info("Los mapas por zona necesitan los modelos (18_three_scores_bundle.pkl o dashboard/lite/engine_lite.pkl).")
    st.stop()


@st.cache_data(show_spinner=False, max_entries=256)
def grid(pid, ptype, ctx_items, env, version):
    return E.game_grid(pid, ptype, dict(ctx_items), env)


ver = Path(engine.__file__).stat().st_mtime


def zone_text(x, z, side):
    vert = "arriba" if z > 3.0 else "abajo" if z < 2.0 else "a media altura"
    if abs(x) < 0.3:
        hor = "al centro"
    else:
        inside = (side == "Right" and x < 0) or (side == "Left" and x > 0)
        hor = "adentro" if inside else "afuera"
    out = " (fuera de zona)" if abs(x) > 0.83 or z < 1.5 or z > 3.5 else ""
    return f"{vert} y {hor}{out}"


G, Gs = {}, {}
for t in ars.AutoPitchType:
    G[t] = grid(pid, t, ctx, env, ver)
    Gs[t] = grid(pid, t, ctx, "No Altitude" if env == "Extreme Altitude" else "Extreme Altitude", ver)

# ---------------- tarjetas del arsenal ----------------
# Sin resaltar "los mejores": Pitching+ no penaliza el daño por contacto (guía 34), así que cada tarjeta muestra la calidad
# esperada junto al costo del contacto en esa misma ubicación, y el usuario ve el trade-off en lugar de un ranking.
cards = st.columns(max(len(ars), 1))
for col, r in zip(cards, ars.itertuples()):
    g = G[r.AutoPitchType]
    i = np.unravel_index(g["pitching_plus"].argmax(), g["pitching_plus"].shape)
    x, z = g["xs"][i[1]], g["ys"][i[0]]
    zone = (np.abs(g["xs"])[None, :] <= 0.83) & (g["ys"][:, None] >= 1.5) & (g["ys"][:, None] <= 3.5)
    dmg, dmg_zone = g["damage"][i], g["damage"][zone].mean()
    dmg_style = f"color:{ui.YEL}" if dmg > dmg_zone else ""
    stuff_env = r.stuff_harp if env == "Extreme Altitude" else r.stuff_mar
    col.markdown(
        f'<div class="dr-card" style="min-height:236px"><div style="display:flex;justify-content:space-between;align-items:center">'
        f'<span><span class="dr-code">{r.code}</span> <b>{r.AutoPitchType}</b></span><span class="dr-note dr-mono">{r.usage:.0%}</span></div>'
        f'<div style="margin:10px 0 2px"><span class="dr-kpi-v" style="font-size:30px">{g["pitching_plus"][i]:.0f}</span> '
        f'<span class="dr-kpi-s">Pitching+ máximo</span></div><div class="dr-note">{zone_text(x, z, bside)}</div>'
        f'<div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:10px"><span class="dr-pill">Whiff ≈ {g["whiff_per_pitch"][i]:.0%}</span>'
        f'<span class="dr-pill">Strike cantado ≈ {g["called_strike"][i]:.0%}</span></div>'
        f'<div class="dr-note" style="margin-top:8px;{dmg_style}">Costo del contacto ahí: <b class="dr-mono">{dmg:.1f}</b> runs/100 '
        f'(zona: {dmg_zone:.1f})</div>'
        f'<div style="display:flex;justify-content:space-between;margin-top:10px"><span class="dr-note">Stuff+ {stuff_env:.0f}</span>{ui.conf_bars(r.evidence, False)}</div></div>',
        unsafe_allow_html=True)
st.markdown(f'<div class="dr-note" style="margin-top:6px">Tarjetas en orden de uso. Niveles de whiff y chase aproximados: la tasa base cambia entre temporadas (guía 35.4); los cambios entre entornos son más confiables que los niveles. Pitching+ mide strike, whiff y chase esperados y <b>no</b> incluye el costo '
            f'del contacto, que en Harp Helú es mayor (H3); por eso cada tarjeta muestra los dos. En guinda, cuando el contacto en la zona de mayor '
            f'Pitching+ cuesta más que el promedio de la zona. Es una lectura de calidad, no una política de selección: el modelo no estima el efecto '
            f'causal de elegir un pitch (guía 37).</div>',
            unsafe_allow_html=True)

# ---------------- heatmaps del pitch elegido ----------------
st.write("")
sel_t = st.segmented_control("Pitch", ars.AutoPitchType.tolist(), default=ars.AutoPitchType.iloc[0] if len(ars) else None,
                             format_func=lambda t: f"{ui.CODE[t]} · {t}", key=f"game_pitch_{pid}")
if sel_t:
    g, go_ = G[sel_t], Gs[sel_t]
    h = st.columns([1, 1, 0.9])

    def heat(z, title, scale, zmid, fmt, zmin=None, zmax=None):
        fig = go.Figure(go.Heatmap(x=g["xs"], y=g["ys"], z=z, colorscale=scale, zmid=zmid, zmin=zmin, zmax=zmax, zsmooth="best",
                                   colorbar=dict(thickness=8, len=0.8), hovertemplate=f"x %{{x:.2f}} · z %{{y:.2f}} ft<br>{fmt}<extra></extra>"))
        fig.add_shape(type="rect", x0=-0.83, x1=0.83, y0=1.5, y1=3.5, line=dict(color="rgba(17,17,17,0.75)", width=1.5))
        bx = -1.25 if bside == "Right" else 1.25
        fig.add_annotation(x=bx, y=3.85, text="Bateador", showarrow=False, font=dict(size=10, color=ui.MUTED))
        fig.update_layout(title=dict(text=title, font=dict(size=13, color=ui.TEXT), x=0.02), height=400, margin=dict(l=30, r=10, t=40, b=30),
                          paper_bgcolor=ui.PANEL, plot_bgcolor=ui.PANEL, font=dict(family="Inter", color=ui.DIM, size=10),
                          xaxis=dict(title="vista del catcher (ft)", scaleanchor="y", showgrid=False), yaxis=dict(title=None, showgrid=False))
        return fig
    h[0].plotly_chart(heat(g["pitching_plus"], "Calidad esperada · Pitching+", [[0, "#E31E24"], [0.5, "#F5EFE3"], [1, "#111111"]], 100, "Pitching+ %{z:.0f}"),
                      use_container_width=True, config={"displayModeBar": False})
    h[1].plotly_chart(heat(g["damage"], "Costo esperado del contacto en este count · runs/100", [[0, "#F5EFE3"], [1, "#E31E24"]], None, "daño %{z:.1f} runs/100",
                           zmin=0, zmax=max(G[t]["damage"].max() for t in G)), use_container_width=True, config={"displayModeBar": False})
    i = np.unravel_index(g["pitching_plus"].argmax(), g["pitching_plus"].shape)
    j = np.unravel_index(g["damage"].argmax(), g["damage"].shape)
    zone = (np.abs(g["xs"])[None, :] <= 0.83) & (g["ys"][:, None] >= 1.5) & (g["ys"][:, None] <= 3.5)
    dz, dz_o = g["damage"][zone].mean(), go_["damage"][zone].mean()
    other = "nivel del mar" if env == "Extreme Altitude" else "Harp Helú"
    here = "Harp Helú" if env == "Extreme Altitude" else "nivel del mar"
    h[2].markdown(
        f'<div class="dr-card" style="height:400px;overflow:auto"><div class="dr-label">Lectura</div>'
        f'<div style="font:600 15px/1.45 Inter;margin:10px 0">En {balls}-{strikes} vs. bateador {"derecho" if bside == "Right" else "zurdo"}, '
        f'el {sel_t} tiene su mayor calidad esperada {zone_text(g["xs"][i[1]], g["ys"][i[0]], bside)} (Pitching+ {g["pitching_plus"][i]:.0f}; '
        f'costo del contacto ahí {g["damage"][i]:.1f} runs/100). '
        f'Su contacto es más caro {zone_text(g["xs"][j[1]], g["ys"][j[0]], bside)}.</div>'
        f'<div style="display:flex;justify-content:space-between;margin-top:6px"><span>Costo del contacto (zona) · {here}</span><b class="dr-mono">{dz:.1f}</b></div>'
        f'<div style="display:flex;justify-content:space-between;margin-top:6px"><span>Mismo pitch · {other}</span><b class="dr-mono">{dz_o:.1f}</b></div>'
        f'<div class="dr-note" style="margin-top:4px">runs por 100 pitches, respecto al valor del count actual: casi 0 en counts neutros '
        f'(un contacto vale lo que el count) y mayor cuando el pitcher va adelante, porque el contacto le quita esa ventaja. El aire de altitud lo '
        f'encarece (H3).</div>'
        f'<div style="margin-top:14px">{ui.conf_bars(ars.set_index("AutoPitchType").evidence[sel_t])}</div></div>', unsafe_allow_html=True)
    with st.expander("Why? · detalle del modelo"):
        st.markdown(
            f'<div class="dr-note" style="font-size:12.5px;line-height:1.6">Cada celda promedia sus {g["n"]} {sel_t} reales de 2026 '
            f'({g["n_used"]} en una muestra fija si son más de 150), cada uno llevado a nivel del mar y después a {here} con la transformación validada '
            f'(IVB × 0.77, HB × 0.71 en Harp Helú): no un "pitch promedio", porque el modelo no es lineal. Esquema del modelo: nivel con la variante '
            f'{"con contexto de arsenal" if g["has_ars"] else "sin arsenal"} a nivel del mar y efecto del entorno con la variante física (guía 38), '
            f'igual que Player y el Simulator. <b>Calidad esperada</b> = Pitching+ (strike cantado, whiff y chase '
            f'ponderados por su valor en carreras, guía 33-34), con count, outs y lado del bateador. <b>Costo del contacto</b> = −P(bola en juego) × '
            f'valor esperado del contacto con el run expectancy LMB del entorno (rama de contacto, guía 34): no entra al score porque no mejoró la '
            f'predicción, pero sí muestra dónde y en qué entorno el contacto cuesta más. Probabilidades en la mejor zona: whiff '
            f'{g["whiff_per_pitch"][i]:.1%}, strike cantado {g["called_strike"][i]:.1%}, chase (si sale de la zona) {g["chase"][i]:.1%}. '
            f'Escenario: pitch anterior = ninguno, entrada 1.</div>', unsafe_allow_html=True)
