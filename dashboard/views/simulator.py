"""Simulator — Altitude + What-if (física y ubicación) + experimentos guardados."""

import sys
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dashboard.lib import data, engine, ui  # noqa: E402

ui.inject_css()
ui.header("Simulator")
if not data.data_ready():
    st.error("Faltan las tablas del dashboard. Corre `python 23_dashboard_data.py` desde la raíz del repo.")
    st.stop()


@st.cache_resource(show_spinner="Cargando modelos…")
def eng(version):
    import importlib
    importlib.reload(engine)
    return engine.Engine()


E = eng(Path(engine.__file__).stat().st_mtime)
FIELDS = [("RelSpeed", "Velocidad", "mph", 5.0, 0.1), ("InducedVertBreak", "IVB a nivel del mar", "in", 8.0, 0.1),
          ("HorzBreak", "HB a nivel del mar (+ = lado del brazo)", "in", 8.0, 0.1), ("SpinRate", "Spin rate", "rpm", 500.0, 10.0),
          ("SpinAxis", "Eje de spin (convención diestro)", "°", 45.0, 1.0), ("Extension", "Extensión", "ft", 1.0, 0.05),
          ("RelHeight", "Altura de release", "ft", 0.8, 0.05)]

top = st.columns([2.2, 1.6], vertical_alignment="bottom")
top[0].markdown('<div class="dr-h1">Simulator</div><div class="dr-sub">Mismo pitch a nivel del mar vs. Harp Helú, y qué pasa si cambia su física o su ubicación</div>',
                unsafe_allow_html=True)
top[1].markdown(f'<div style="display:flex;justify-content:flex-end">{ui.env_badge()}</div>', unsafe_allow_html=True)

# ---------------- origen del pitch ----------------
qp, state = st.query_params, data.read_state()
P, A, FP = data.load("pitchers"), data.load("arsenal"), data.load("fingerprint")
year = 2026  # el Simulator usa el modelo vigente (ventana 2024-25 -> 2026), el único con pitches para transformar
asked = qp.get("season")
# Solo pitchers con proyección (pitches con categoría de altitud y movimiento medido): sin ellos no hay qué transformar.
pool = P[(P.year == year) & (P.n >= 50) & P.stuff_harp.notna()].sort_values("n", ascending=False)
src = st.columns([1.2, 1.3, 1.2, 1.4], vertical_alignment="bottom")
mode = src[0].segmented_control("Origen", ["Pitcher del roster", "Pitch manual"], default="Pitcher del roster")
base_fb, usage, meta, rows, rows_ars = None, None, {}, None, None
if mode == "Pitch manual":
    ptype = src[1].selectbox("Tipo de pitch", list(ui.CODE))
    hand = src[2].selectbox("Mano", ["Right", "Left"], format_func=lambda h: "RHP" if h == "Right" else "LHP")
    m = ((E.F.AutoPitchType.astype(str) == ptype) & (E.F.PitcherThrows.astype(str) == hand)
         & E.F.altitude_category.astype(str).isin(engine.ENVS)).to_numpy()
    rows = E.F[m].sample(min(int(m.sum()), 3000), random_state=0).reset_index(drop=True)
    rows_ars = np.zeros(len(rows), bool)
    sea = E.sea_level(rows)
    base = {f: float(np.nanmean(sea[f])) for f in ["RelSpeed", "InducedVertBreak", "HorzBreak", "SpinRate", "Extension", "RelHeight", "RelSide"]}
    base["SpinAxis"] = float(np.degrees(np.arctan2(np.nanmean(np.sin(np.radians(sea.SpinAxis))), np.nanmean(np.cos(np.radians(sea.SpinAxis))))) % 360)
    meta = {"pitcher": "Manual", "pitch": ptype, "label": f"{ptype} manual"}
    src[3].markdown(f'<div class="dr-note">Actual = {ptype} promedio de la liga ({"RHP" if hand == "Right" else "LHP"}, a nivel del mar). Para un prospecto, '
                    'mueve los controles a sus métricas medidas a nivel del mar: se simula con la dispersión real de la liga centrada en sus valores '
                    '(variante sin arsenal, la que recibe un pitcher nuevo).</div>', unsafe_allow_html=True)
    ev_t = A[(A.year == year) & (A.AutoPitchType == ptype) & (A.n >= 50)].evidence
    evidence = ev_t.mode().iloc[0] if len(ev_t) else "Uncertain"
else:
    pid0 = qp.get("pid", state.get("last_pid"))
    ids = pool.pitcher_anon_id.tolist()
    if pid0 and pid0 not in ids:
        st.info(f"{ui.pid_label(pid0)} no tiene suficientes pitches proyectables en 2026 (con categoría de altitud y movimiento medido); "
                f"el Simulator usa el modelo vigente (2026). "
                "Elige otro pitcher o usa 'Pitch manual' con sus métricas.")
    elif asked and str(asked) != "2026":
        st.caption("El Simulator usa la temporada 2026 (modelo vigente).")
    pid = src[1].selectbox("Pitcher", ids, index=ids.index(pid0) if pid0 in ids else 0, format_func=ui.pid_label)
    ars = A[(A.year == year) & (A.pitcher_anon_id == pid) & (A.n >= 20)].sort_values("usage", ascending=False)
    types = ars.AutoPitchType.tolist()
    pt0 = qp.get("pitch")
    ptype = src[2].selectbox("Pitch", types, index=types.index(pt0) if pt0 in types else 0)
    row = ars.set_index("AutoPitchType").loc[ptype]
    hand = {"RHP": "Right", "LHP": "Left"}.get(pool.set_index("pitcher_anon_id").loc[pid, "hand"], "Right")
    rows, rows_ars = E.pitcher_rows(pid, ptype)
    sea = E.sea_level(rows)
    base = {f: float(np.nanmean(sea[f])) for f in ["RelSpeed", "InducedVertBreak", "HorzBreak", "SpinRate", "Extension", "RelHeight", "RelSide"]}
    base["SpinAxis"] = float(np.degrees(np.arctan2(np.nanmean(np.sin(np.radians(sea.SpinAxis))), np.nanmean(np.cos(np.radians(sea.SpinAxis))))) % 360)
    fbs = ars[ars.AutoPitchType.isin(["Four-Seam", "Sinker"])].sort_values("n", ascending=False)
    if len(fbs):
        f = fbs.iloc[0]
        ffp = FP[(FP.year == year) & (FP.pitcher_anon_id == pid) & (FP.AutoPitchType == f.AutoPitchType)].set_index("env")
        base_fb = {"RelSpeed": f.velo, "InducedVertBreak": ffp.loc["No Altitude", "ivb"], "HorzBreak": ffp.loc["No Altitude", "hb"],
                   "SpinRate": f.spin, "Extension": f.extension, "RelHeight": f.rel_height, "RelSide": f.rel_side}
        usage = float(row.usage)
    evidence = row.evidence
    meta = {"pitcher": pid, "pitch": ptype, "label": f"{ui.pid_label(pid)} · {ptype}", "fb": None if base_fb is None else str(fbs.iloc[0].AutoPitchType)}
    src[3].markdown(f'<div class="dr-note">Punto de partida = sus {len(rows)} {ptype} de 2026, cada uno llevado a nivel del mar; el what-if desplaza todos por igual. '
                    f'Recta de referencia para el contexto de arsenal: {meta["fb"] or "sin recta primaria"}.</div>', unsafe_allow_html=True)
STEP = {f: st_ for f, _l, _u, _s, st_ in FIELDS}
base = {k: (round(float(v) / STEP[k]) * STEP[k] if k in STEP else float(v)) for k, v in base.items()}
base.update(AutoPitchType=ptype, PitcherThrows=hand)
key = f"{meta['pitcher']}|{ptype}|{hand}"

if st.session_state.pop("saved_msg", None):
    st.toast("Experimento guardado.", icon="✅")
tabs = st.tabs(["What-if · Altitud", "Ubicación · Location+", f"Experimentos guardados ({len(data.list_experiments())})"])

# =============== Tab 1: What-if + altitud ===============
with tabs[0]:
    left, right = st.columns([1, 1.9], gap="large")
    with left:
        st.markdown('<div class="dr-label" style="margin-bottom:6px">Escenario (what-if)</div>', unsafe_allow_html=True)
        if st.button("Restablecer al pitch actual", use_container_width=True):
            for f, *_ in FIELDS:
                st.session_state.pop(f"{key}|{f}", None)
        scen = dict(base)
        for f, lab, unit, span, step in FIELDS:
            v0 = base[f]
            lo, hi = (v0 - span, v0 + span) if f != "SpinAxis" else (max(0.0, v0 - span), min(360.0, v0 + span))
            scen[f] = st.slider(f"{lab} ({unit})", float(round(lo, 2)), float(round(hi, 2)), float(round(v0, 2)), step, key=f"{key}|{f}")
        use_ars = base_fb is not None
        st.markdown('<div class="dr-note" style="margin-top:6px">Nivel = Stuff+ oficial a nivel del mar; efecto de altitud = variante física del '
                    'modelo (titular, guía 38: coincide con lo observado). La lectura muestra también la sensibilidad con el modelo de arsenal.</div>',
                    unsafe_allow_html=True)
    fb, us = (base_fb, usage) if use_ars else (None, None)
    if fb is not None and ptype in ("Four-Seam", "Sinker") and meta.get("fb") == ptype:
        fb_s = dict(fb); fb_s.update({k: scen[k] for k in ["RelSpeed", "InducedVertBreak", "HorzBreak", "SpinRate", "Extension", "RelHeight"]})
    else:
        fb_s = fb
    shifts = {f: scen[f] - base[f] for f, *_ in FIELDS}
    pop_rows, pop_ars = rows, rows_ars  # pitches del escenario; 'rows' se reutiliza abajo para la tabla
    sens = None
    if rows is not None:
        r_ars = rows_ars
        tb = E.population_table(rows, r_ars)
        ts = E.population_table(rows, r_ars, shifts)
        if r_ars.any():
            so = E.population_table(rows, r_ars, shifts, hybrid=False).set_index("env").stuff_plus
            sens = so["Extreme Altitude"] - so["No Altitude"]
    else:
        tb = E.altitude_table(base, fb, us)
        ts = E.altitude_table(scen, fb_s, us)
    changed = [lab for f, lab, _u, _s, stp in FIELDS if abs(scen[f] - base[f]) > stp / 2]
    with right:
        cards = st.columns(3)
        for col, (i, r) in zip(cards, ts.iterrows()):
            b = tb.loc[i]
            d = r.stuff_plus - b.stuff_plus
            hl = "border-color:rgba(212,38,47,.55);" if r.env == "Extreme Altitude" else ""
            col.markdown(f'<div class="dr-card" style="{hl}"><div class="dr-label">{r.label}</div>'
                         f'<div style="margin-top:8px"><span class="dr-kpi-v">{r.stuff_plus:.1f}</span> <span class="dr-kpi-s">Stuff+ escenario</span></div>'
                         f'<div class="dr-note" style="margin-top:6px">Actual {b.stuff_plus:.1f} · cambio {ui.delta_fmt(d)}</div></div>', unsafe_allow_html=True)
        hh = tb.set_index("env"); ss = ts.set_index("env")
        rows = []
        for lab, col, fmt in [("Stuff+", "stuff_plus", "{:.1f}"), ("IVB (in)", "IVB", "{:.1f}"), ("HB (in)", "HB", "{:.1f}"),
                              ("Whiff por pitch", "p_whiff_per_pitch", "{:.1%}"), ("Chase (si sale de la zona)", "p_chase", "{:.1%}")]:
            cells = "".join(f'<td class="dr-mono">{fmt.format(hh.loc[e, col])}</td><td class="dr-mono"><b>{fmt.format(ss.loc[e, col])}</b></td>'
                            for e in ["No Altitude", "Extreme Altitude"])
            rows.append(f"<tr><td>{lab}</td>{cells}</tr>")
        st.markdown(f'<div class="dr-card" style="padding:0;margin-top:12px"><table class="dr-tbl"><tr><th></th><th>Nivel del mar · actual</th>'
                    f'<th>Nivel del mar · escenario</th><th>Harp Helú · actual</th><th>Harp Helú · escenario</th></tr>{"".join(rows)}</table>'
                    f'<div class="dr-note" style="padding:8px 14px">Niveles de whiff y chase aproximados: la tasa base cambia entre temporadas (guía 35.4); los cambios entre entornos son más confiables que los niveles.</div></div>',
                    unsafe_allow_html=True)
        g = st.columns([1.3, 1])
        fig = go.Figure()
        fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.15)")); fig.add_vline(x=0, line=dict(color="rgba(255,255,255,0.15)"))
        with g[0]:
            show_lg = st.toggle(f"Comparar con los {ptype} de la liga con más y menos whiff real", value=True, key="sim_mm")
        mm = data.movement_map(year)
        mm = mm[(mm.AutoPitchType == ptype) & (mm.grupo != "Resto")]
        if show_lg and len(mm):
            ui.movement_traces(fig, mm, "harp", size=7)
        for df_, name, color in [(tb, "Actual", "#eeede9"), (ts, "Escenario", ui.ACCENT)]:
            d = df_.set_index("env")
            fig.add_trace(go.Scatter(x=d.HB, y=d.IVB, mode="lines", line=dict(color=color, dash="dot", width=1.5), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=[d.loc["No Altitude", "HB"]], y=[d.loc["No Altitude", "IVB"]], mode="markers", name=f"{name} · nivel del mar",
                                     marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(color=color, width=2.5))))
            fig.add_trace(go.Scatter(x=[d.loc["Extreme Altitude", "HB"]], y=[d.loc["Extreme Altitude", "IVB"]], mode="markers", name=f"{name} · Harp Helú",
                                     marker=dict(size=16, color=color, line=dict(color="#15161a", width=2))))
        fig.update_layout(height=330, margin=dict(l=40, r=10, t=10, b=40), paper_bgcolor=ui.PANEL, plot_bgcolor=ui.PANEL,
                          font=dict(family="IBM Plex Sans", color=ui.DIM, size=11), legend=dict(font=dict(size=10), y=1, x=1.02),
                          xaxis=dict(title="HB (in) · + lado del brazo", gridcolor="rgba(255,255,255,0.05)", zeroline=False),
                          yaxis=dict(title="IVB (in)", gridcolor="rgba(255,255,255,0.05)", zeroline=False))
        g[0].plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        if show_lg and len(mm):
            g[0].markdown('<div class="dr-note">● azul / ✕ naranja = quintil alto / bajo de whiff real por pitch (resultados de la temporada) entre los pitchers de la liga con ese tipo (≥ 50 pitches), con su movimiento en Harp Helú. Compáralos con el marcador relleno del pitch.</div>', unsafe_allow_html=True)
        warn = E.out_of_range(scen)
        dh = ss.loc["Extreme Altitude", "stuff_plus"] - hh.loc["Extreme Altitude", "stuff_plus"]
        loss = hh.loc["No Altitude", "stuff_plus"] - hh.loc["Extreme Altitude", "stuff_plus"]
        loss_txt = (f"pierde {loss:.1f} de Stuff+" if loss >= 0.05 else f"gana {-loss:.1f} de Stuff+" if loss <= -0.05 else "conserva su Stuff+")
        msg = (f'<div class="dr-label">Lectura</div><div style="font:600 14px/1.45 IBM Plex Sans;margin:8px 0 12px">'
               + (f"Con los cambios en {', '.join(changed).lower()}, el Stuff+ en Harp Helú pasa de {hh.loc['Extreme Altitude', 'stuff_plus']:.1f} a "
                  f"{ss.loc['Extreme Altitude', 'stuff_plus']:.1f} ({dh:+.1f})." if changed else
                  f"El mismo pitch {loss_txt} al pasar del nivel del mar "
                  f"a Harp Helú (IVB {hh.loc['No Altitude', 'IVB']:.1f} → {hh.loc['Extreme Altitude', 'IVB']:.1f} in). Mueve los controles para explorar ajustes.")
               + f'</div><div style="margin-bottom:10px">{ui.conf_bars(evidence)}</div>'
               + ("" if sens is None else f'<div class="dr-note">Sensibilidad · efecto de altitud con el modelo de arsenal: {sens:+.1f} '
                  f'(además penaliza que la separación contra la recta se encoja; no aparece en lo observado).</div>')
               + "".join(f'<div class="dr-note" style="color:{ui.YEL}">⚠ {ui.esc(w)}: el modelo extrapola.</div>' for w in warn)
               + '<div class="dr-note" style="margin-top:8px">What-if exploratorio: cambia la física del pitch dentro del modelo. No garantiza que el '
                 'pitcher pueda lograr ese cambio ni que el efecto sea causal. Medium usa los factores de 20 (todas las temporadas).</div>')
        g[1].markdown(f'<div class="dr-card" style="height:330px;overflow:auto">{msg}</div>', unsafe_allow_html=True)
        s1, s2 = st.columns([2, 1], vertical_alignment="bottom")
        name = s1.text_input("Nombre del experimento", value=f"{meta['label']} · {'what-if' if changed else 'actual'}")
        if s2.button("Guardar experimento", type="primary", use_container_width=True):
            data.save_experiment({"id": uuid.uuid4().hex[:10], "name": name, "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
                                  "season": year, "pitcher": meta["pitcher"], "pitch": ptype, "hand": hand, "arsenal_context": bool(use_ars),
                                  "base": {k: base[k] for k, *_ in FIELDS}, "scenario": {k: scen[k] for k, *_ in FIELDS}, "changed": changed,
                                  "results": {"actual_mar": hh.loc["No Altitude", "stuff_plus"], "actual_harp": hh.loc["Extreme Altitude", "stuff_plus"],
                                              "escenario_mar": ss.loc["No Altitude", "stuff_plus"], "escenario_harp": ss.loc["Extreme Altitude", "stuff_plus"]},
                                  "warnings": warn})
            st.session_state["saved_msg"] = name
            st.rerun()

# =============== Tab 2: ubicación ===============
with tabs[1]:
    c = st.columns([1, 1.6, 1.2], gap="large")
    with c[0]:
        st.markdown('<div class="dr-label" style="margin-bottom:6px">Contexto</div>', unsafe_allow_html=True)
        balls = st.select_slider("Bolas", [0, 1, 2, 3], value=0)
        strikes = st.select_slider("Strikes", [0, 1, 2], value=0)
        bside = st.segmented_control("Bateador", ["Right", "Left"], default="Right", format_func=lambda s: "Derecho" if s == "Right" else "Zurdo")
        px = st.slider("Ubicación horizontal (ft, vista del catcher)", -1.6, 1.6, 0.0, 0.05)
        pz = st.slider("Altura (ft)", 0.6, 4.2, 2.5, 0.05)
        st.markdown('<div class="dr-note">Location+ usa la ubicación, el count y el tipo de pitch (no la física). Pitching+ promedia los pitches '
                    'reales del escenario de la pestaña anterior en esta ubicación y count.</div>', unsafe_allow_html=True)
    ctx = {"Balls": balls, "Strikes": strikes, "BatterSide": bside or "Right", "PlateLocSide": px, "PlateLocHeight": pz}

    @st.cache_data(show_spinner=False, max_entries=64)
    def grid(pitch_items, ctx_items):
        xs, ys, z = E.location_grid(dict(pitch_items), dict(ctx_items))
        return xs, ys, z
    xs, ys, z = grid(tuple(sorted((k, v) for k, v in scen.items())), tuple(sorted((k, v) for k, v in ctx.items() if k not in ("PlateLocSide", "PlateLocHeight"))))
    with c[1]:
        fig = go.Figure(go.Heatmap(x=xs, y=ys, z=z, colorscale=[[0, "#1d2a3a"], [0.5, "#2c2e34"], [1, "#d4262f"]], zmid=100,
                                   colorbar=dict(title="Location+", thickness=10), hovertemplate="x %{x:.2f} ft · z %{y:.2f} ft<br>Location+ %{z:.1f}<extra></extra>"))
        fig.add_shape(type="rect", x0=-0.83, x1=0.83, y0=1.5, y1=3.5, line=dict(color="rgba(255,255,255,0.7)", width=1.5))
        fig.add_trace(go.Scatter(x=[px], y=[pz], mode="markers", marker=dict(size=14, color="#eeede9", line=dict(color="#111", width=2)), showlegend=False))
        fig.update_layout(height=430, margin=dict(l=40, r=10, t=10, b=40), paper_bgcolor=ui.PANEL, plot_bgcolor=ui.PANEL,
                          font=dict(family="IBM Plex Sans", color=ui.DIM, size=11),
                          xaxis=dict(title="Lado (ft) · vista del catcher", scaleanchor="y"), yaxis=dict(title="Altura (ft)"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with c[2]:
        loc_s, lp = E.score("location", *E.frame(scen, ctx=ctx))
        if pop_rows is not None:  # población del escenario (los mismos pitches que el What-if), no un pitch promedio (guía 37.5)
            sea_s, a_s = E._shift(E.sea_level(pop_rows), shifts), pop_ars
            if len(sea_s) > E.MAX_GRID_ROWS:
                keep = np.sort(np.random.default_rng(0).choice(len(sea_s), E.MAX_GRID_ROWS, replace=False))
                sea_s, a_s = sea_s.iloc[keep].reset_index(drop=True), a_s[keep]
            at = pd.DataFrame([{"env": e, "label": engine.ENV_LABEL[e],
                                "pitching_plus": float(E.population_pitching(sea_s, a_s, ctx, np.array([px]), np.array([pz]), e)["pitching_plus"][0])}
                               for e in engine.ENVS]).set_index("env")
        else:
            at = E.altitude_table(scen, fb_s, us, ctx=ctx).set_index("env")
        st.markdown(
            f'<div class="dr-card"><div class="dr-label">En esta ubicación y count</div>'
            f'<div style="margin-top:10px"><span class="dr-kpi-v">{loc_s[0]:.1f}</span> <span class="dr-kpi-s">Location+</span></div>'
            f'<div class="dr-note">Called strike {lp["called_strike"][0]:.1%} · whiff {lp["whiff_per_pitch"][0]:.1%} · chase {lp["chase"][0]:.1%}</div>'
            f'<div style="height:1px;background:{ui.LINE};margin:14px 0"></div><div class="dr-label">Pitching+ (física + ubicación + contexto)</div>'
            + "".join(f'<div style="display:flex;justify-content:space-between;margin-top:8px"><span>{at.loc[e, "label"]}</span>'
                      f'<b class="dr-mono">{at.loc[e, "pitching_plus"]:.1f}</b></div>' for e in engine.ENVS)
            + '<div class="dr-note" style="margin-top:12px">Location+ es neutral al entorno. Pitching+ sí incluye el ambiente y el movimiento '
              'transformado. Es calidad esperada, no "el mejor pitch": no hay política causal de selección (guía 37).</div>'
            + f'<div class="dr-note" style="margin-top:8px;color:{ui.YEL}">⚠ Location+ y Pitching+ valoran sobre todo strike, whiff y chase. '
              'No penalizan el daño por contacto (esa rama no mejoró el modelo, guía 34): con pocos strikes, el centro de la zona sale alto. '
              'Para el riesgo de contacto en altitud, ver H3.</div></div>',
            unsafe_allow_html=True)

# =============== Tab 3: experimentos ===============
with tabs[2]:
    exps = data.list_experiments()
    if not exps:
        st.markdown('<div class="dr-card dr-note">Aún no hay experimentos guardados. Guárdalos desde la pestaña What-if · Altitud.</div>', unsafe_allow_html=True)
    else:
        rows = []
        for e in reversed(exps):
            r = e["results"]
            ch = ", ".join(f"{k}: {e['base'][k]:.1f}→{e['scenario'][k]:.1f}" for k, _l, _u, _s, stp in FIELDS if abs(e["scenario"][k] - e["base"][k]) > stp / 2) or "sin cambios"
            rows.append(f'<tr><td><b>{ui.esc(e["name"])}</b><div class="dr-note">{e["created"]} · temporada {e["season"]}</div></td>'
                        f'<td>{ui.esc(ui.pid_label(e["pitcher"]) if e["pitcher"] != "Manual" else "Manual")} · {e["pitch"]}</td>'
                        f'<td class="dr-note">{ui.esc(ch)}</td>'
                        f'<td class="dr-mono">{r["actual_mar"]:.1f} → <b>{r["escenario_mar"]:.1f}</b></td>'
                        f'<td class="dr-mono">{r["actual_harp"]:.1f} → <b>{r["escenario_harp"]:.1f}</b> {ui.delta_fmt(r["escenario_harp"] - r["actual_harp"])}</td>'
                        f'<td>{"⚠" if e.get("warnings") else ""}</td></tr>')
        st.markdown(f'<div class="dr-card" style="padding:0"><table class="dr-tbl"><tr><th>Experimento</th><th>Pitch</th><th>Cambios</th>'
                    f'<th>Stuff+ nivel del mar</th><th>Stuff+ Harp Helú</th><th></th></tr>{"".join(rows)}</table></div>', unsafe_allow_html=True)
        d = st.multiselect("Eliminar experimentos", [e["id"] for e in exps], format_func=lambda i: next(e["name"] for e in exps if e["id"] == i))
        if d and st.button("Eliminar seleccionados"):
            data.delete_experiments(d)
            st.rerun()
