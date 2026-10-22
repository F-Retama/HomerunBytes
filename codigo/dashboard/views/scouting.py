"""Scouting — perfil ideal, explorador de la liga, evaluación de prospectos, comparador y shortlist."""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dashboard.lib import data, engine, ui  # noqa: E402

ui.inject_css()
ui.header("Scouting")
if not data.data_ready():
    st.error("Faltan las tablas del dashboard. Corre `python 23_dashboard_data.py` desde la raíz del repo.")
    st.stop()


@st.cache_resource(show_spinner="Cargando modelos…")
def eng(version):
    import importlib
    importlib.reload(engine)
    return engine.Engine()


FEAT = {"RelSpeed": ("Velocidad", "mph"), "InducedVertBreak": ("IVB", "in"), "HorzBreak": ("HB (+ brazo)", "in"), "SpinRate": ("Spin", "rpm"),
        "SpinAxis": ("Eje de spin", "°"), "RelHeight": ("Altura de release", "ft"), "RelSide": ("Lado de release", "ft"),
        "Extension": ("Extensión", "ft"), "velo_diff_vs_fb": ("Δ velocidad vs recta", "mph"), "ivb_diff_vs_fb": ("Δ IVB vs recta", "in"),
        "hb_diff_vs_fb": ("Δ HB vs recta", "in"), "pitch_usage_pct": ("Uso", "")}
TYPES = list(ui.CODE)
state = data.read_state()
P, A, H = data.load("pitchers"), data.load("arsenal"), data.load("history")
top = st.columns([2.2, 1.6], vertical_alignment="bottom")
top[0].markdown('<div class="dr-h1">Scouting</div><div class="dr-sub">Qué perfil buscar, quién lo tiene y cómo se proyecta en Harp Helú</div>',
                unsafe_allow_html=True)
with top[1]:
    st.markdown(f'<div style="display:flex;justify-content:flex-end;margin-bottom:8px">{ui.env_badge()}</div>', unsafe_allow_html=True)
    yrs = data.seasons()
    year = st.selectbox("Temporada", yrs, index=yrs.index(state.get("season", yrs[0])) if state.get("season") in yrs else 0)
SL = data.get_shortlist()
if st.session_state.get("sl_msg"):
    st.toast(st.session_state.pop("sl_msg"), icon="✅")
tabs = st.tabs(["Perfil ideal", "Explorar liga", "Evaluar prospecto", "Comparar", f"Shortlist ({len(SL)})", "Arsenal ideal"])


def hist_spark(pid):
    h = H[H.pitcher_anon_id == pid].sort_values("year")
    return ui.sparkline(h.stuff_plus.tolist(), w=70, h=22) + f'<span class="dr-note"> {" · ".join(f"{int(y) % 100}:{v:.0f}" for y, v in zip(h.year, h.stuff_plus))}</span>'


# =============== Perfil ideal ===============
with tabs[0]:
    PR, PV, STB = data.load("profiles"), data.load("profile_validation"), data.load("stability_by_type")
    EV, REC = data.load("evidence_levels").set_index("AutoPitchType"), data.load("type_recommendations").set_index("AutoPitchType")
    t = st.segmented_control("Tipo de pitch", TYPES, default="Four-Seam")
    if t:
        ev = EV.loc[t]
        DT = data.load("delta_by_type")
        d_of = float(DT[(DT.variante == "oficial") & (DT.AutoPitchType == t) & DT.window.astype(str).str.endswith("2026")].delta_stuff.iloc[0])
        lvl = {"Robusto": "Robust", "Condicional": "Conditional", "Incierto": "Uncertain"}[ev.nivel_evidencia]
        c = st.columns([1.2, 1])
        c[0].markdown(
            f'<div class="dr-card"><div style="display:flex;gap:10px;align-items:center"><span class="dr-code">{ui.CODE[t]}</span>'
            f'<span class="dr-h2">{t} en Harp Helú</span></div>'
            f'<div style="display:flex;gap:28px;margin:14px 0"><div><div class="dr-label">Δ Stuff+ modelado (2026 · física)</div>'
            f'<span class="dr-kpi-v">{ev.delta_2026:+.1f}</span> <span class="dr-note">[{ev.ci95_lo:+.1f}, {ev.ci95_hi:+.1f}]</span></div>'
            f'<div><div class="dr-label">Con modelo de arsenal</div><span class="dr-kpi-v" style="color:{ui.DIM}">{d_of:+.1f}</span>'
            f' <span class="dr-note">sensibilidad</span></div>'
            f'<div><div class="dr-label">Observado mismo pitcher</div><span class="dr-kpi-v" style="color:{ui.DIM}">'
            f'{"—" if ev.observado_2026 != ev.observado_2026 else f"{ev.observado_2026:+.1f}"}</span></div>'
            f'<div><div class="dr-label">Evidencia</div><div style="margin-top:10px">{ui.conf_bars(lvl)}</div></div></div>'
            f'<div style="font:500 13.5px/1.5 IBM Plex Sans;color:{ui.TEXT}">{ui.esc(REC.loc[t, "recomendacion"])}</div></div>', unsafe_allow_html=True)
        v = PV[PV.AutoPitchType == t].set_index("resultado")
        s = STB.set_index("AutoPitchType").loc[t]
        vtxt = "".join(f'<div style="display:flex;justify-content:space-between;margin-top:8px"><span>{lab}</span><b class="dr-mono">'
                       f'{v.loc[k, "brecha_quintil_alto_menos_bajo_pp"]:+.1f} pp <span class="dr-note">[{v.loc[k, "ci95_lo"]:+.1f}, {v.loc[k, "ci95_hi"]:+.1f}]</span></b></div>'
                       for k, lab in [("is_whiff", "Whiff"), ("chase", "Chase")] if k in v.index)
        c[1].markdown(
            f'<div class="dr-card"><div class="dr-label">¿El Stuff+ alto se traduce en resultados en Extreme? (2026)</div>{vtxt}'
            f'<div class="dr-note" style="margin-top:6px">Quintil alto menos quintil bajo de Stuff+, pitches lanzados en Extreme. IC95% bootstrap por pitcher.</div>'
            f'<div style="height:1px;background:{ui.LINE};margin:12px 0"></div><div class="dr-label">Estabilidad del Stuff+ (pitcher × tipo, año a año)</div>'
            f'<div class="dr-mono" style="margin-top:8px">{s.anio_a_anio_24_25:.2f} (24→25) · {s.anio_a_anio_25_26:.2f} (25→26)</div>'
            f'<div class="dr-note">Rasgo repetible: lo que se ve hoy se sostiene la temporada siguiente.</div></div>', unsafe_allow_html=True)
        pr = PR[PR.AutoPitchType == t]
        rows = []
        for r in pr.sort_values(["en_perfil", "lift_extreme_2025"], ascending=[False, False], key=lambda x: x.abs() if x.name == "lift_extreme_2025" else x).itertuples():
            lab, unit = FEAT.get(r.feature, (r.feature, ""))
            sc = 100 if r.feature == "pitch_usage_pct" else 1
            arrow = "más" if r.lift_extreme_2025 > 0 else "menos"
            res = "" if r.resiliencia_spearman != r.resiliencia_spearman or abs(r.resiliencia_spearman) < 0.1 else \
                f'{"más" if r.resiliencia_spearman > 0 else "menos"} {lab.lower()} → pierde menos de lo esperado (ρ {r.resiliencia_spearman:+.2f})'
            rows.append(f'<tr style="{"" if r.en_perfil else "opacity:.45"}"><td><b>{lab}</b></td>'
                        f'<td class="dr-mono">{"<b>" if r.en_perfil else ""}{r.zona_p25 * sc:.1f}–{r.zona_p75 * sc:.1f} {unit}{"</b>" if r.en_perfil else ""}</td>'
                        f'<td class="dr-mono">{r.mediana_tipo * sc:.1f}</td><td>{arrow} <span class="dr-note dr-mono">lift {r.lift_extreme_2025:+.2f} → {r.lift_extreme_2026:+.2f}</span></td>'
                        f'<td>{ui.esc(r.entorno) if r.en_perfil else "<span class=dr-note>no cumple criterio</span>"}</td><td class="dr-note">{ui.esc(res)}</td></tr>')
        st.markdown(f'<div class="dr-card" style="padding:0;margin-top:14px"><div style="padding:14px 18px"><span class="dr-h2">Perfil físico ideal</span> '
                    f'<span class="dr-note">&nbsp;zona P25–P75 de los {t} de Stuff+ alto lanzados en Extreme (aprendido en 2025, confirmado en 2026)</span></div>'
                    f'<table class="dr-tbl"><tr><th>Rasgo</th><th>Zona ideal</th><th>Mediana del tipo</th><th>Dirección</th><th>Entorno</th><th>Resiliencia a la altitud (ajustada por nivel)</th></tr>'
                    f'{"".join(rows)}</table><div class="dr-note" style="padding:10px 14px">En negritas, los rasgos que cumplen el criterio (|lift| ≥ 0.10, IC sin 0 y '
                    f'mismo signo en 2026). "Compartido con nivel del mar" = también distingue a los mejores lejos de la altitud. Resiliencia = correlación del '
                    f'rasgo con lo que el pitch pierde de más o de menos frente a lo esperado para su Stuff+ a nivel del mar y su tipo '
                    f'(no con los puntos brutos que pierde: los mejores pitches siempre pierden más). Perfiles exploratorios: describen el modelo y se validan con whiff y chase reales.</div></div>',
                    unsafe_allow_html=True)

        # ---- Mapa de movimiento: más vs. menos efectivos ----
        st.markdown(f'<div style="margin-top:22px"><span class="dr-h2">Mapa de movimiento</span> <span class="dr-note">&nbsp;{t} de la liga en {year}: '
                    f'los más y los menos efectivos</span></div>', unsafe_allow_html=True)
        eff_lab = {"whiff": "Whiff real", "chase": "Chase real", "stuff": "Stuff+ proyectado"}
        eff_by = st.segmented_control("Efectividad según", list(eff_lab), format_func=eff_lab.get, default="whiff", key="mm_by") or "whiff"
        MM = data.movement_map(year, by=eff_by)
        mt = MM[MM.AutoPitchType == t]
        if (mt.grupo != "Resto").sum() < 6:
            st.info(f"No hay suficientes {t} (pitcher × tipo con ≥ 50 pitches) para separar los más y los menos efectivos.")
        else:
            mc = st.columns([2.2, 1], gap="large")
            with mc[0]:
                env_lab = st.segmented_control("Movimiento en", ["Harp Helú", "Nivel del mar"], default="Harp Helú", key="mm_env") or "Harp Helú"
                env_k = "harp" if env_lab == "Harp Helú" else "mar"
                fig = ui.movement_layout(ui.movement_traces(go.Figure(), mt, env_k, size=10))
                for grp in ["Más efectivos", "Menos efectivos"]:
                    g_ = mt[mt.grupo == grp]
                    fig.add_trace(go.Scatter(x=[g_[f"hb_{env_k}"].mean()], y=[g_[f"ivb_{env_k}"].mean()], mode="markers", name=f"Centro · {grp.lower()}",
                                             marker=dict(size=20, symbol="diamond-open", color=ui.GROUP_COLOR[grp], line=dict(width=3)), hoverinfo="skip"))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            mt = mt.assign(obs_whiff_pct=100 * mt.obs_whiff, obs_chase_pct=100 * mt.obs_chase)
            hi_, lo_ = mt[mt.grupo == "Más efectivos"], mt[mt.grupo == "Menos efectivos"]
            cmp_rows = "".join(
                f'<tr><td>{lab}</td><td class="dr-mono" style="color:{ui.GROUP_COLOR["Más efectivos"]}"><b>{fmt.format(hi_[c].mean())}</b></td>'
                f'<td class="dr-mono" style="color:{ui.GROUP_COLOR["Menos efectivos"]}">{fmt.format(lo_[c].mean())}</td>'
                f'<td class="dr-mono">{fmt.replace("{:", "{:+").format(hi_[c].mean() - lo_[c].mean())}</td></tr>'
                for lab, c, fmt in [("Whiff real por pitch (%)", "obs_whiff_pct", "{:.1f}"), ("Chase real (%)", "obs_chase_pct", "{:.1f}"),
                                    ("Stuff+ en Harp Helú", "stuff_harp", "{:.1f}"), ("Velocidad (mph)", "velo", "{:.1f}"), ("IVB en Harp Helú (in)", "ivb_harp", "{:.1f}"),
                                    ("HB en Harp Helú (in)", "hb_harp", "{:.1f}"), ("IVB a nivel del mar (in)", "ivb_mar", "{:.1f}"),
                                    ("Spin (rpm)", "spin", "{:.0f}"), ("Δ altitud (Stuff+)", "delta", "{:.1f}")])
            mc[1].markdown(f'<div class="dr-card" style="padding:0;margin-top:44px"><table class="dr-tbl"><tr><th></th><th>Más efectivos</th><th>Menos efectivos</th>'
                           f'<th>Diferencia</th></tr>{cmp_rows}</table><div class="dr-note" style="padding:10px 14px">Cada marca es un pitcher × tipo con ≥ 50 '
                           f'pitches ({len(mt)} en total). Más / menos efectivos = quintil alto / bajo de {data.EFFECT_BY[eff_by][1].lower()} dentro del tipo '
                           f'({len(hi_)} y {len(lo_)}). '
                           + ("Whiff y chase son resultados reales de la temporada en todos los parques (whiff de un año al siguiente: correlación 0.67), "
                              "así que el mapa no depende del modelo. " if eff_by != "stuff" else
                              "Con Stuff+ el mapa muestra el movimiento que el propio modelo premia: es una lectura del modelo, no una validación. ")
                           + f'Movimiento en convención de diestro; en Harp Helú es el observado si hay ≥ 20 pitches allí y si no el '
                           f'proyectado. Promedios simples por pitcher.</div></div>', unsafe_allow_html=True)

# =============== Arsenal ideal ===============
with tabs[5]:
    if not data.has("arsenal_composicion"):
        st.info("Falta la tabla de arsenales. Corre `python 24_arsenal_harp.py` desde la raíz del repo.")
    else:
        AC, AR, AM = data.load("arsenal_composicion"), data.load("arsenal_rectas"), data.load("arsenal_mezcla")
        AT, ATOP = data.load("arsenal_tipos"), data.load("arsenal_top")
        ac, ar, at_, atop = AC[AC.year == year], AR[AR.year == year], AT[AT.year == year].set_index("AutoPitchType"), ATOP[ATOP.year == year]
        if not len(ac):
            st.info(f"Sin datos de arsenal para {year}.")
        else:
            am = AM[AM.year == year].iloc[0]
            sl_ = ar[ar.grupo.str.startswith("Pendiente")].set_index("grupo")
            s_mar, s_harp, s_d = (sl_.loc[k] for k in sl_.index)
            best = ac.iloc[0]
            CT = data.load("arsenal_contraste") if data.has("arsenal_contraste") else None
            ct = CT[CT.year == year].set_index("medida") if CT is not None else None
            k_val = "Valor real por 100 pitches en Extreme: pendiente por +10 pp de rectas"
            k_si, k_re = "Tasa de roletazos: sinker >= 30% del arsenal", "Tasa de roletazos: resto"
            real_txt = ""
            if ct is not None and k_val in ct.index:
                v = ct.loc[k_val]
                real_txt = (f"En resultados reales no se ve ese costo: el valor por 100 pitches lanzados en altitud no cambia con el porcentaje de rectas "
                            f"({v.valor:+.2f} carreras por +10 pp [{v.ci95_lo:+.2f}, {v.ci95_hi:+.2f}], {int(v.n_pitchers)} pitchers). ")
            if ct is not None and k_si in ct.index and k_re in ct.index:
                real_txt += (f"Y los pitchers con mucho sinker inducen más roletazos ({ct.loc[k_si].valor:.0%} contra {ct.loc[k_re].valor:.0%} de sus bolas en juego), que en altitud "
                             f"valen casi el doble (H3) y el Stuff+ casi no pondera.")
            st.markdown(
                f'<div class="dr-card"><div class="dr-label">¿Qué arsenal produce el Stuff+ más alto en Harp Helú? · {year}</div>'
                f'<div style="font:600 15px/1.5 IBM Plex Sans;margin-top:8px">En Stuff+, el que depende menos de la recta y la acompaña con pitches que el aire '
                f'casi no toca (slider, cutter, changeup): cada +10 puntos porcentuales de rectas (Four-Seam + Sinker) cuestan {abs(s_d.delta):.2f} de Stuff+ '
                f'adicionales en Harp Helú [{s_d.ci95_lo:+.2f}, {s_d.ci95_hi:+.2f}]; a nivel del mar no cambian el Stuff+ '
                f'({s_mar.delta:+.2f} [{s_mar.ci95_lo:+.2f}, {s_mar.ci95_hi:+.2f}]).</div>'
                + (f'<div style="font:600 14px/1.5 IBM Plex Sans;margin-top:8px;color:{ui.YEL}">{real_txt}</div>' if real_txt else '')
                + f'<div class="dr-note" style="margin-top:8px">Lectura: es una vista de swing-and-miss (el Stuff+ pondera sobre todo chase y whiff), no una '
                f'recomendación de roster; no descartar a un sinkerballer por ella. La pendiente es casi aritmética: las rectas pierden más en altitud y el resto '
                f'casi nada. La mezcla de tipos explica solo {am.r2_mezcla_harp:.0%} de las diferencias de Stuff+ entre pitchers en Harp Helú '
                f'({am.r2_mezcla_nivel_mar:.0%} a nivel del mar). Descriptivo, no causal.</div></div>', unsafe_allow_html=True)
            c = st.columns([1.25, 1], gap="large")
            with c[0]:
                st.markdown('<div style="margin-top:14px"><span class="dr-h2">Stuff+ en Harp Helú por composición del arsenal</span> '
                            '<span class="dr-note">&nbsp;tipos con uso ≥ 10% · composiciones con ≥ 10 pitchers</span></div>', unsafe_allow_html=True)
                o = ac.sort_values("stuff_harp")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=o.stuff_mar, y=o.composicion, mode="markers", name="Nivel del mar",
                                         marker=dict(size=11, color="rgba(0,0,0,0)", line=dict(color="#9b9da4", width=2))))
                fig.add_trace(go.Scatter(x=o.stuff_harp, y=o.composicion, mode="markers", name="Harp Helú (IC 95%)",
                                         marker=dict(size=12, color=ui.ACCENT), error_x=dict(type="data", symmetric=False, array=o.ci95_hi - o.stuff_harp,
                                                                                            arrayminus=o.stuff_harp - o.ci95_lo, color="rgba(212,38,47,.5)", thickness=1.5),
                                         customdata=o.n_pitchers, hovertemplate="%{y}<br>Harp Helú %{x:.1f} · %{customdata} pitchers<extra></extra>"))
                fig.update_layout(height=60 + 30 * len(o), margin=dict(l=10, r=10, t=10, b=40), paper_bgcolor=ui.PANEL, plot_bgcolor=ui.PANEL,
                                  font=dict(family="IBM Plex Sans", color=ui.DIM, size=11), legend=dict(font=dict(size=10), orientation="h", y=1.06, x=0),
                                  xaxis=dict(title="Stuff+ del arsenal (ponderado por uso, promedio de pitchers)", gridcolor="rgba(255,255,255,0.05)"),
                                  yaxis=dict(tickfont=dict(family="IBM Plex Mono", size=11, color=ui.TEXT),
                                             ticktext=[f"{r.composicion}  ·  n {r.n_pitchers}" for r in o.itertuples()], tickvals=o.composicion))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                pv = AC.pivot(index="composicion", columns="year", values="stuff_harp").dropna()
                stab = f" Entre temporadas el orden se conserva a medias (Spearman {pv.corr(method='spearman').iloc[0, 1]:.2f}, {len(pv)} composiciones)." if pv.shape[1] >= 2 and len(pv) >= 5 else ""
                st.markdown(f'<div class="dr-note">Los intervalos se traslapan: las composiciones intermedias no se distinguen entre sí.{stab} '
                            f'Lo estable son los extremos y el efecto del porcentaje de rectas.</div>', unsafe_allow_html=True)
            with c[1]:
                q = ar[~ar.grupo.str.startswith("Pendiente")]
                qrows = "".join(f'<tr><td>{r.grupo}</td><td class="dr-mono">{r.pct_rectas:.0f}%</td><td class="dr-mono">{r.stuff_mar:.1f}</td>'
                                f'<td class="dr-mono"><b>{r.stuff_harp:.1f}</b></td><td>{ui.delta_fmt(r.delta)}</td>'
                                f'<td class="dr-mono">{"—" if getattr(r, "gb_rate", float("nan")) != getattr(r, "gb_rate", float("nan")) else f"{r.gb_rate:.0%}"}</td></tr>' for r in q.itertuples())
                st.markdown(f'<div class="dr-card" style="padding:0;margin-top:14px"><div style="padding:12px 16px"><span class="dr-h2">Dependencia de la recta</span></div>'
                            f'<table class="dr-tbl"><tr><th>Cuartil</th><th>% rectas</th><th>Stuff+ mar</th><th>Stuff+ Harp</th><th>Δ</th><th>Roletazos (real)</th></tr>{qrows}</table>'
                            f'<div class="dr-note" style="padding:10px 14px">Pendiente en Harp Helú por +10 pp de rectas: {s_harp.delta:+.2f} '
                            f'[{s_harp.ci95_lo:+.2f}, {s_harp.ci95_hi:+.2f}] de Stuff+. {int(q.n_pitchers.sum())} pitchers con ≥ 100 pitches. '
                            f'Roletazos (real): roletazos entre bolas en juego sin toques, promedio por pitcher en todos los parques.</div></div>',
                            unsafe_allow_html=True)
                trows = "".join(f'<tr><td><span class="dr-code">{ui.CODE[k]}</span> {k}</td><td class="dr-mono">{r.stuff_mar:.1f}</td>'
                                f'<td class="dr-mono"><b>{r.stuff_harp:.1f}</b></td><td>{ui.delta_fmt(r.delta)}</td></tr>'
                                for k, r in at_.sort_values("stuff_harp", ascending=False).iterrows())
                st.markdown(f'<div class="dr-card" style="padding:0;margin-top:12px"><div style="padding:12px 16px"><span class="dr-h2">Bloques del arsenal</span> '
                            f'<span class="dr-note">&nbsp;Stuff+ medio de liga por tipo</span></div>'
                            f'<table class="dr-tbl"><tr><th>Tipo</th><th>Nivel del mar</th><th>Harp Helú</th><th>Δ</th></tr>{trows}</table></div>', unsafe_allow_html=True)
            trs = "".join(f'<tr><td><a target="_self" href="player?pid={r.pitcher_anon_id}&season={year}"><b>{ui.pid_label(r.pitcher_anon_id)}</b></a>'
                          f'<div class="dr-note">{r.role} · {r.hand} · n {int(r.n)}</div></td><td class="dr-mono">{r.usos}</td><td class="dr-mono">{r.pct_rectas:.0f}%</td>'
                          f'<td class="dr-mono">{r.stuff_mar:.1f} → <b>{r.stuff_harp:.1f}</b></td><td>{ui.delta_fmt(r.delta)}</td>'
                          f'<td class="dr-mono">{r.mezcla_harp:.1f}</td><td class="dr-mono">{r.calidad_harp:+.1f}</td></tr>' for r in atop.itertuples())
            st.markdown(f'<div class="dr-card" style="padding:0;margin-top:14px"><div style="padding:12px 16px"><span class="dr-h2">Los 10 arsenales con mayor Stuff+ en Harp Helú</span> '
                        f'<span class="dr-note">&nbsp;pitchers con ≥ 300 pitches en {year}</span></div>'
                        f'<table class="dr-tbl"><tr><th>Pitcher</th><th>Arsenal (uso)</th><th>% rectas</th><th>Stuff+ mar → Harp</th><th>Δ</th><th>Mezcla</th><th>Calidad</th></tr>'
                        f'{trs}</table><div class="dr-note" style="padding:10px 14px">Mezcla = Stuff+ que tendría en Harp Helú si cada pitch suyo fuera el promedio de liga '
                        f'de su tipo. Calidad = lo que agregan sus pitches sobre ese promedio. Stuff+ Harp = mezcla + calidad.</div></div>', unsafe_allow_html=True)

# =============== Explorar liga ===============
with tabs[1]:
    pool = P[(P.year == year)].copy()
    f = st.columns([1.1, 1.1, 1.5, 2, 1.6], vertical_alignment="bottom")
    role = f[0].segmented_control("Rol", ["All", "SP", "RP"], default="All", key="ex_role")
    hand = f[1].segmented_control("Mano", ["All", "RHP", "LHP"], default="All", key="ex_hand")
    minn = f[2].slider("Mínimo de pitches", 50, 800, 150, 25)
    need = f[3].multiselect("Debe lanzar", TYPES, placeholder="Cualquier arsenal")
    sig = f[4].selectbox("Señal", ["Todas", "Stuff+ por encima de resultados", "Resultados por encima de Stuff+"])
    ref = pool[pool.n >= 300]
    q75, q25 = ref.stuff_harp.quantile(.75), ref.stuff_harp.quantile(.25)
    v50, v75 = ref.value_per_100.quantile(.5), ref.value_per_100.quantile(.75)
    pool["signal"] = np.select([(pool.n >= 300) & (pool.stuff_harp >= q75) & (pool.value_per_100 <= v50),
                                (pool.n >= 300) & (pool.stuff_harp <= q25) & (pool.value_per_100 >= v75)],
                               ["Stuff+ por encima de resultados", "Resultados por encima de Stuff+"], "")
    V = pool[pool.n >= minn]
    if role and role != "All":
        V = V[V.role == role]
    if hand and hand != "All":
        V = V[V.hand == hand]
    for t_ in need:
        V = V[V.pitcher_anon_id.isin(A[(A.year == year) & (A.AutoPitchType == t_) & (A.n >= 30)].pitcher_anon_id)]
    if sig != "Todas":
        V = V[V.signal == sig]
    V = V.sort_values("stuff_harp", ascending=False)
    c = st.columns([1.4, 1])
    with c[0]:
        R = pool[pool.n >= 300]
        fig = go.Figure()
        fig.add_vline(x=q75, line=dict(color="rgba(255,255,255,0.18)", dash="dot")); fig.add_hline(y=v50, line=dict(color="rgba(255,255,255,0.18)", dash="dot"))
        for sgl, col in [("", "#6b6e75"), ("Stuff+ por encima de resultados", ui.GRN), ("Resultados por encima de Stuff+", ui.YEL)]:
            m = R[R.signal == sgl]
            fig.add_trace(go.Scatter(x=m.stuff_harp, y=m.value_per_100, mode="markers", name=sgl or "Resto",
                                     marker=dict(color=col, size=8, line=dict(color="#15161a", width=1)),
                                     text=[ui.pid_label(x) for x in m.pitcher_anon_id], hovertemplate="%{text}<br>Stuff+ Harp %{x:.1f}<br>Valor/100 %{y:+.2f}<extra></extra>"))
        fig.update_layout(height=420, margin=dict(l=40, r=10, t=10, b=40), paper_bgcolor=ui.PANEL, plot_bgcolor=ui.PANEL,
                          font=dict(family="IBM Plex Sans", color=ui.DIM, size=11), legend=dict(orientation="h", y=1.1, font=dict(size=10)),
                          xaxis=dict(title="Stuff+ proyectado en Harp Helú", gridcolor="rgba(255,255,255,0.05)"),
                          yaxis=dict(title="Valor realizado (runs / 100 pitches)", gridcolor="rgba(255,255,255,0.05)"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with c[1]:
        st.markdown(f'<div class="dr-card"><div class="dr-label">Cómo leer el mapa</div><div style="font:500 13.5px/1.5 IBM Plex Sans;margin-top:8px">'
                    f'<span style="color:{ui.GRN}">●</span> <b>Stuff+ por encima de resultados</b>: calidad física en el cuartil alto pero valor realizado bajo la '
                    f'mediana: posible infravalorado.<br><span style="color:{ui.YEL}">●</span> <b>Resultados por encima de Stuff+</b>: buenos resultados con '
                    f'Stuff+ bajo: riesgo de regresión.</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="dr-note">Señales exploratorias (pitchers con ≥ 300 pitches): Stuff+ en el cuartil alto con valor realizado bajo la mediana '
                    '(posible infravalorado) o al revés. Stuff+ es mucho más estable que el valor realizado (año a año 0.77 vs 0.22) y lo mejor '
                    'es combinarlos (guía 35): una señal es motivo para revisar, no para contratar.</div>', unsafe_allow_html=True)
        add = st.multiselect("Agregar a shortlist", V.pitcher_anon_id.head(40).tolist(), format_func=ui.pid_label)
        if add and st.button("Agregar", type="primary"):
            n = data.add_to_shortlist([{"key": f"p:{x}", "kind": "pitcher", "pid": x, "season": int(year), "added": datetime.now().strftime("%Y-%m-%d"),
                                        "priority": "B", "note": ""} for x in add])
            st.session_state["sl_msg"] = f"Shortlist: {n} candidatos"
            st.rerun()

    rows = []
    for r in V.head(40).itertuples():
        sg = {"Stuff+ por encima de resultados": (ui.GRN, "▲ Stuff+ > resultados"), "Resultados por encima de Stuff+": (ui.YEL, "▼ resultados > Stuff+")}.get(r.signal)
        rows.append(f'<tr><td><a target="_self" href="player?pid={r.pitcher_anon_id}&season={year}"><b>{ui.pid_label(r.pitcher_anon_id)}</b></a>'
                    f'<div class="dr-note">{r.role} · {r.hand} · {ui.esc(r.team_label if isinstance(r.team_label, str) else "—")}</div></td>'
                    f'<td class="dr-mono">{r.stuff_mar:.1f} → <b>{r.stuff_harp:.1f}</b></td><td>{ui.delta_fmt(r.delta)}</td><td>{ui.fit_pill(r.fit)}</td>'
                    f'<td class="dr-mono">{r.location_plus:.0f} / {r.pitching_plus:.0f}</td><td class="dr-mono">{r.value_per_100:+.1f}</td>'
                    f'<td>{hist_spark(r.pitcher_anon_id)}</td>'
                    f'<td>{"" if not sg else f"<span style=color:{sg[0]};font-weight:600;white-space:nowrap>{sg[1]}</span>"}</td></tr>')
    st.write("")
    st.markdown(f'<div class="dr-card" style="padding:0"><div style="padding:14px 18px"><span class="dr-h2">Candidatos</span> '
                  f'<span class="dr-note">&nbsp;{len(V)} pitchers · ordenados por Stuff+ proyectado en Harp Helú (top 40)</span></div>'
                  f'<table class="dr-tbl"><tr><th>Pitcher</th><th>Stuff+ mar → Harp</th><th>Δ</th><th>Fit</th><th>Loc+ / Pitch+</th><th>Valor/100</th>'
                  f'<th>Historial Stuff+</th><th>Señal</th></tr>{"".join(rows) or "<tr><td colspan=8 class=dr-note>Sin candidatos con estos filtros.</td></tr>"}</table></div>',
                  unsafe_allow_html=True)
# =============== Evaluar prospecto ===============
with tabs[2]:
    E = eng(Path(engine.__file__).stat().st_mtime)
    ENV_OPTS = {"No Altitude": "Nivel del mar", "Medium Altitude": "Medium Altitude", "Extreme Altitude": "Harp Helú (Extreme)"}
    MOVE = ("InducedVertBreak", "HorzBreak")
    c = st.columns([1.2, 0.8, 1.2, 1.4], vertical_alignment="bottom")
    pname = c[0].text_input("Nombre del prospecto", value="Prospecto 1")
    phand = c[1].selectbox("Mano", ["Right", "Left"], format_func=lambda h: "RHP" if h == "Right" else "LHP", key="p_hand")
    penv = c[2].selectbox("Medido en", list(ENV_OPTS), format_func=ENV_OPTS.get, key="p_env",
                          help="Entorno donde se midieron las métricas. El movimiento (IVB, HB) se lleva a nivel del mar con los factores validados "
                               "(guía 36); velocidad, spin y release no cambian con el parque. Cambiar el entorno reinicia la tabla.")
    use_ars = c[3].toggle("Contexto de arsenal (su recta y uso)", value=False,
                          help="Apagado = variante sin arsenal, la validada para pitchers nuevos (guía 35). Encendido = usa las diferencias contra su "
                               "recta y su uso, como un pitcher con historial: más información, pero extrapola fuera de la LMB.")

    def r_env(env, var):
        """Fracción del movimiento a nivel del mar que conserva el pitch en ese entorno (1 = nivel del mar)."""
        return 1.0 if env == "No Altitude" else float(E.fac[env][var])

    def to_sea(d):
        return {k: (v / r_env(penv, k) if k in MOVE else v) for k, v in d.items()}

    def default_row(t, usage):
        m = E.league_sample(t, phand)[2]  # promedio de la liga a nivel del mar, expresado en el entorno de medición
        return {"Tipo": t, "Uso %": usage, "Velocidad": round(m["RelSpeed"], 1), "IVB": round(m["InducedVertBreak"] * r_env(penv, "InducedVertBreak"), 1),
                "HB": round(m["HorzBreak"] * r_env(penv, "HorzBreak"), 1), "Spin": round(m["SpinRate"]), "Eje": round(m["SpinAxis"]),
                "Extensión": round(m["Extension"], 2), "Altura rel.": round(m["RelHeight"], 2), "Lado rel.": round(m["RelSide"], 2)}
    ed_key = f"pros_{phand}_{penv}"
    if ed_key not in st.session_state:
        st.session_state[ed_key] = pd.DataFrame([default_row("Four-Seam", 50), default_row("Slider", 30), default_row("Changeup", 20)])
    st.markdown(f'<div class="dr-note">Métricas tal como se midieron en <b>{ENV_OPTS[penv]}</b>, en convención de diestro: HB positivo = lado del brazo. '
                f'El movimiento se convierte a nivel del mar antes de simular. Valores iniciales = promedio de la liga para el tipo y la mano, '
                f'medido en ese entorno. Agrega o quita filas.</div>', unsafe_allow_html=True)
    ed = st.data_editor(st.session_state[ed_key], num_rows="dynamic", use_container_width=True, key=f"{ed_key}_ed",
                        column_config={"Tipo": st.column_config.SelectboxColumn(options=TYPES, required=True)})
    ed = ed.dropna(subset=["Tipo"])
    if len(ed):
        fbr = ed[ed.Tipo.isin(["Four-Seam", "Sinker"])].sort_values("Uso %", ascending=False)
        cols = {"Velocidad": "RelSpeed", "IVB": "InducedVertBreak", "HB": "HorzBreak", "Spin": "SpinRate", "Eje": "SpinAxis",
                "Extensión": "Extension", "Altura rel.": "RelHeight", "Lado rel.": "RelSide"}
        fb = None
        if use_ars and len(fbr):
            fb = to_sea({cols[k]: float(fbr.iloc[0][k]) for k in cols})
        if fb:  # misma variante que la referencia oficial (scores de 18 con arsenal cuando existe)
            AL = A[(A.year == 2026) & (A.n >= 50)].rename(columns={"stuff_harp": "harp"})
            PL = P[(P.year == 2026) & (P.n >= 100)].rename(columns={"stuff_harp": "harp"})
        else:   # referencia de la liga calculada con la variante sin arsenal (comparación justa)
            with st.spinner("Calculando la referencia de la liga sin arsenal (una sola vez)…"):
                AL, PL = E.reference_base()
        PRF = data.load("profiles")
        out = []
        for r in ed.to_dict("records"):
            vals = to_sea({cols[k]: float(r[k]) for k in cols})
            # Las zonas del perfil se midieron en pitches lanzados en Extreme: el movimiento se compara en Harp Helú (guía 40.3).
            vals_harp = {k: (v * r_env("Extreme Altitude", k) if k in MOVE else v) for k, v in vals.items()}
            usage = float(r["Uso %"]) / 100
            t = r["Tipo"]
            tb = E.prospect(t, phand, vals, fb=fb, usage=usage if fb else None).set_index("env")
            d = tb.loc["Extreme Altitude", "stuff_plus"] - tb.loc["No Altitude", "stuff_plus"]
            d_of = d
            if fb:
                to = E.prospect(t, phand, vals, fb=fb, usage=usage, hybrid=False).set_index("env").stuff_plus
                d_of = to["Extreme Altitude"] - to["No Altitude"]
            ref_h = AL[AL.AutoPitchType == t].harp
            pct = float((ref_h <= tb.loc["Extreme Altitude", "stuff_plus"]).mean()) if len(ref_h) else np.nan
            fit = "Strong Fit" if pct >= .75 else "Moderate Fit" if pct >= .5 else "Low Fit" if pct >= .25 else "Poor Fit"
            prof = PRF[(PRF.AutoPitchType == t) & PRF.en_perfil & PRF.feature.isin(cols.values())]
            inz = sum(1 for q in prof.itertuples() if (q.zona_p25 <= vals_harp[q.feature] <= q.zona_p75)
                      or (q.lift_extreme_2025 > 0 and vals_harp[q.feature] > q.zona_p75) or (q.lift_extreme_2025 < 0 and vals_harp[q.feature] < q.zona_p25))
            out.append({"Tipo": t, "usage": usage, "mar": tb.loc["No Altitude", "stuff_plus"], "harp": tb.loc["Extreme Altitude", "stuff_plus"], "delta": d,
                        "sens": "" if abs(d_of - d) < 0.7 else f'<div class="dr-note">arsenal {d_of:+.1f}</div>',
                        "fit": fit, "perfil": f"{inz} de {len(prof)}" if len(prof) else "—", "whiff_harp": tb.loc["Extreme Altitude", "p_whiff_per_pitch"],
                        "IVB_harp": tb.loc["Extreme Altitude", "IVB"], "HB_harp": tb.loc["Extreme Altitude", "HB"], "warn": E.out_of_range({**vals, "AutoPitchType": t})})
        O = pd.DataFrame(out)
        w = O.usage / O.usage.sum() if O.usage.sum() > 0 else np.full(len(O), 1 / len(O))
        ars_mar, ars_harp = float((O.mar * w).sum()), float((O.harp * w).sum())
        ars_d = ars_harp - ars_mar
        sens_pct = float((PL.delta <= ars_d).mean()); fit_pct = harp_pct = float((PL.harp <= ars_harp).mean())
        fit_all = "Strong Fit" if fit_pct >= .75 else "Moderate Fit" if fit_pct >= .5 else "Low Fit" if fit_pct >= .25 else "Poor Fit"
        k = st.columns(4)
        for col, lab, val, sub in [(k[0], "Stuff+ arsenal · nivel del mar", f"{ars_mar:.1f}", "ponderado por uso"),
                                   (k[1], "Stuff+ arsenal · Harp Helú", f"{ars_harp:.1f}", f"percentil {harp_pct:.0%} de la liga"),
                                   (k[2], "Δ altitud", f"{ars_d:+.1f}", f"percentil {sens_pct:.0%} (alto = pierde menos)"),
                                   (k[3], "Harp Helú Fit", "", "")]:
            col.markdown(f'<div class="dr-card" style="height:96px"><div class="dr-label">{lab}</div><div style="margin-top:8px">'
                         + (ui.fit_pill(fit_all) if lab == "Harp Helú Fit" else f'<span class="dr-kpi-v">{val}</span> <span class="dr-kpi-s">{sub}</span>')
                         + '</div></div>', unsafe_allow_html=True)
        rows = "".join(f'<tr><td><span class="dr-code">{ui.CODE[o.Tipo]}</span> <b>{o.Tipo}</b></td><td class="dr-mono">{o.usage:.0%}</td>'
                       f'<td class="dr-mono">{o.mar:.1f} → <b>{o.harp:.1f}</b></td><td>{ui.delta_fmt(o.delta)}{o.sens}</td><td>{ui.fit_pill(o.fit)}</td>'
                       f'<td class="dr-mono">{o.IVB_harp:.1f} / {o.HB_harp:.1f}</td><td class="dr-mono">{o.whiff_harp:.1%}</td><td class="dr-mono">{o.perfil}</td>'
                       f'<td class="dr-note" style="color:{ui.YEL}">{"⚠ " + "; ".join(o.warn) if o.warn else ""}</td></tr>' for o in O.itertuples())
        st.markdown(f'<div class="dr-card" style="padding:0;margin-top:12px"><table class="dr-tbl"><tr><th>Pitch</th><th>Uso</th><th>Stuff+ mar → Harp</th><th>Δ</th>'
                    f'<th>Fit vs. su tipo</th><th>IVB / HB en Harp</th><th>Whiff en Harp (aprox.)</th><th>Rasgos del perfil</th><th></th></tr>{rows}</table>'
                    f'<div class="dr-note" style="padding:10px 14px">Cada pitch se simula con la dispersión real de la liga (tipo y mano) centrada en sus métricas. '
                    f'Fit = percentil de su Stuff+ proyectado en Harp Helú entre pitches del mismo tipo en la liga (2026). Δ = variante física '
                    f'(titular); "arsenal" = sensibilidad con el modelo de arsenal cuando difiere ≥ 0.7. "Rasgos del perfil" = cuántos rasgos del perfil ideal cumple '
                    f'(dentro de la zona o más allá en la dirección favorable), con el movimiento en Harp Helú porque las zonas se midieron en altitud. '
                    f'Medido en {ENV_OPTS[penv]}. Variante {"con" if fb else "sin"} contexto de arsenal; los percentiles se '
                    f'calculan contra la liga evaluada con la misma variante.</div></div>',
                    unsafe_allow_html=True)
        if st.button("Guardar prospecto en shortlist", type="primary"):
            n = data.add_to_shortlist([{"key": f"x:{pname}", "kind": "prospect", "name": pname, "hand": phand, "added": datetime.now().strftime("%Y-%m-%d"),
                                        "priority": "B", "note": "", "arsenal_context": bool(fb), "measured_in": penv,
                                        "results": {"stuff_mar": ars_mar, "stuff_harp": ars_harp, "delta": ars_d, "fit": fit_all,
                                                    "pitches": O.drop(columns=["warn", "sens"]).to_dict("records")},
                                        "inputs": ed.to_dict("records")}])
            st.session_state["sl_msg"] = f"Shortlist: {n} candidatos"
            st.rerun()

# =============== Comparar ===============
with tabs[3]:
    SL = data.get_shortlist()
    opts = {e["key"]: (ui.pid_label(e["pid"]) if e["kind"] == "pitcher" else f'{e["name"]} (prospecto)') for e in SL}
    for pid in P[(P.year == year) & (P.n >= 150)].sort_values("stuff_harp", ascending=False).pitcher_anon_id.head(60):
        opts.setdefault(f"p:{pid}", ui.pid_label(pid))
    sel = st.multiselect("Candidatos (shortlist primero, luego los mejores Stuff+ en Harp Helú)", list(opts), format_func=opts.get, max_selections=4,
                         default=[e["key"] for e in SL[:3]])
    if len(sel) >= 2:
        cols = []
        for kk in sel:
            e = next((x for x in SL if x["key"] == kk), {"key": kk, "kind": "pitcher", "pid": kk[2:]})
            if e["kind"] == "pitcher":
                pp = P[(P.year == year) & (P.pitcher_anon_id == e["pid"])]
                if pp.empty:
                    continue
                pp = pp.iloc[0]
                ar = A[(A.year == year) & (A.pitcher_anon_id == e["pid"]) & (A.n >= 20)]
                cols.append({"name": ui.pid_label(e["pid"]), "sub": f"{pp.role} · {pp.hand}", "mar": pp.stuff_mar, "harp": pp.stuff_harp, "delta": pp.delta,
                             "fit": pp.fit, "loc": f"{pp.location_plus:.0f} / {pp.pitching_plus:.0f}", "val": f"{pp.value_per_100:+.1f}", "conf": pp.confidence,
                             "hist": hist_spark(e["pid"]), "ars": {r.AutoPitchType: (r.usage, r.stuff_mar, r.stuff_harp) for r in ar.itertuples()},
                             "variant": "Oficial (con arsenal si tiene historial)"})
            else:
                rr = e["results"]
                cols.append({"name": e["name"], "sub": f'Prospecto · {"RHP" if e["hand"] == "Right" else "LHP"}', "mar": rr["stuff_mar"], "harp": rr["stuff_harp"],
                             "delta": rr["delta"], "fit": rr["fit"], "loc": "— (sin ubicación)", "val": "—", "conf": "Uncertain", "hist": "—",
                             "ars": {p_["Tipo"]: (p_["usage"], p_["mar"], p_["harp"]) for p_ in rr["pitches"]},
                             "variant": "Con arsenal" if e.get("arsenal_context") else "Sin arsenal (pitcher nuevo)"})
        head = "".join(f'<th>{ui.esc(c_["name"])}<div class="dr-note" style="text-transform:none;letter-spacing:0">{c_["sub"]}</div></th>' for c_ in cols)
        def row(lbl, fn):
            return f"<tr><td class='dr-note'>{lbl}</td>" + "".join(f"<td>{fn(c_)}</td>" for c_ in cols) + "</tr>"
        types_all = [t for t in TYPES if any(t in c_["ars"] for c_ in cols)]
        ars_rows = "".join(row(f'<span class="dr-code">{ui.CODE[t]}</span> {t}', lambda c_, t=t: (
            f'<span class="dr-mono">{c_["ars"][t][1]:.0f} → <b>{c_["ars"][t][2]:.0f}</b></span> <span class="dr-note">{c_["ars"][t][0]:.0%}</span>'
            if t in c_["ars"] else '<span class="dr-note">—</span>')) for t in types_all)
        st.markdown(
            f'<div class="dr-card" style="padding:0"><table class="dr-tbl"><tr><th></th>{head}</tr>'
            + row("Stuff+ nivel del mar → Harp Helú", lambda c_: f'<span class="dr-mono">{c_["mar"]:.1f} → <b>{c_["harp"]:.1f}</b></span>')
            + row("Δ altitud", lambda c_: ui.delta_fmt(c_["delta"]))
            + row("Harp Helú Fit", lambda c_: ui.fit_pill(c_["fit"]))
            + row("Location+ / Pitching+", lambda c_: f'<span class="dr-mono">{c_["loc"]}</span>')
            + row("Valor realizado / 100", lambda c_: f'<span class="dr-mono">{c_["val"]}</span>')
            + row("Historial Stuff+", lambda c_: c_["hist"])
            + row("Confianza del efecto de altitud", lambda c_: ui.conf_bars(c_["conf"]))
            + row("Variante del modelo", lambda c_: f'<span class="dr-note">{c_["variant"]}</span>')
            + f'<tr><td colspan="{len(cols) + 1}" class="dr-label" style="padding-top:16px">Arsenal · Stuff+ mar → Harp (uso)</td></tr>' + ars_rows
            + "</table></div>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="dr-card dr-note">Selecciona 2 a 4 candidatos.</div>', unsafe_allow_html=True)

# =============== Shortlist ===============
with tabs[4]:
    SL = data.get_shortlist()
    if not SL:
        st.markdown('<div class="dr-card dr-note">La shortlist está vacía. Agrega candidatos desde Explorar liga o Evaluar prospecto.</div>', unsafe_allow_html=True)
    else:
        rows = []
        for e in SL:
            if e["kind"] == "pitcher":
                pp = P[(P.year == e["season"]) & (P.pitcher_anon_id == e["pid"])]
                pp = pp.iloc[0] if len(pp) else None
                name, sub = ui.pid_label(e["pid"]), (f"{pp.role} · {pp.hand}" if pp is not None else "")
                mar, harp, fit = (pp.stuff_mar, pp.stuff_harp, pp.fit) if pp is not None else (np.nan, np.nan, "Sin muestra")
            else:
                name, sub = e["name"], "Prospecto"
                mar, harp, fit = e["results"]["stuff_mar"], e["results"]["stuff_harp"], e["results"]["fit"]
            rows.append({"key": e["key"], "Candidato": name, "Tipo": sub, "Stuff+ mar": round(mar, 1), "Stuff+ Harp": round(harp, 1),
                         "Fit": fit, "Prioridad": e.get("priority", "B"), "Nota": e.get("note", ""), "Agregado": e["added"], "Quitar": False})
        df = pd.DataFrame(rows)
        ed = st.data_editor(df.drop(columns="key"), use_container_width=True, hide_index=True, disabled=["Candidato", "Tipo", "Stuff+ mar", "Stuff+ Harp", "Fit", "Agregado"],
                            column_config={"Prioridad": st.column_config.SelectboxColumn(options=["A", "B", "C"]),
                                           "Quitar": st.column_config.CheckboxColumn()})
        if st.button("Guardar cambios", type="primary"):
            new = []
            for e, (_, r) in zip(SL, ed.iterrows()):
                if not r["Quitar"]:
                    e.update(priority=r["Prioridad"], note=r["Nota"])
                    new.append(e)
            data.save_shortlist(new)
            st.toast("Shortlist actualizada", icon="✅")
            st.rerun()
        st.markdown('<div class="dr-note">Prioridad A/B/C y notas se guardan en dashboard/state/. Los prospectos guardan sus métricas y resultados.</div>',
                    unsafe_allow_html=True)
