"""Capa de datos del dashboard (Pitching Decision System): convierte los outputs congelados del
modelo (18-22) en tablas listas para cada pantalla. No entrena ni cambia ningun modelo.

Todas las reglas de producto (Harp Helu Fit, confianza, semaforo, alertas, inbox, targets) estan
aqui, documentadas y con umbrales explicitos, para que el dashboard solo muestre.

Definiciones
  Temporada      = temporada evaluada fuera de muestra: 2025 (modelo 2024 -> 2025) y 2026 (modelo
                   2024-25 -> 2026). Scores oficiales de 18 (stuff_plus, location_plus, pitching_plus).
  Delta altitud  = efecto del aire sobre el Stuff+ del mismo pitch (22), con la variante FISICA del modelo
                   (titular desde la auditoria 2, guia 38: coincide con lo observado). delta_oficial = misma
                   cuenta con la variante con arsenal (sensibilidad). Stuff+ en Harp Helu = Stuff+ oficial a
                   nivel del mar + Delta altitud.
  Delta observado= Stuff+ real en Extreme menos en No Altitude del mismo pitcher x tipo (>= 20 pitches
                   en ambos); se pondera por el uso total del pitcher. Secundario: incluye adaptacion.
  Harp Helu Fit  = CALIDAD donde importa (decision de la auditoria 2): percentil del Stuff+ proyectado en
                   Harp Helu entre pitchers con >= 100 pitches. Strong >= p75, Moderate p50-p75, Low p25-p50,
                   Poor < p25. La sensibilidad a la altitud (Delta) se reporta aparte: medirla sola castigaba
                   a los mejores pitchers (corr Stuff+ vs Delta = -0.52).
  Exceso         = residual del Delta tras controlar el Stuff+ a nivel del mar dentro de cada tipo de pitch:
                   cuanto pierde un pitch MAS (negativo) o MENOS de lo esperado para su nivel.
  Fit por pitch  = percentil del Stuff+ proyectado en Harp Helu frente a pitches del mismo tipo (>= 50 pitches).
  Confianza      = por pitcher: Robust si >= 60% de sus pitches son de tipos con evidencia Robusta (21)
                   y tiene >= 300 pitches; Uncertain si < 150 pitches o >= 40% en tipos Inciertos;
                   Conditional en otro caso. Por pitch: el nivel de su tipo (21), Uncertain si < 50 pitches.
  Semaforo pitch = 'pierde mas de lo esperado': Adjust si su Exceso esta en el 10% mas bajo de su tipo y el tipo
                   tiene evidencia Robusta; Review si esta en el 25% mas bajo; Maintain en otro caso.
                   Prioridades de revision, no instrucciones de uso.
  Alerta pitcher = Needs attention si un pitch con uso >= 15% esta en Adjust; Review si un pitch central
                   (uso >= 30%) esta en Review; Maintain en otro caso.
  Inbox          = un item por pitcher en alerta (su pitch mas afectado) + oportunidades: pitches con
                   Stuff+ >= 105 que pierden < 1.5 en altitud ("calidad que se sostiene en Harp Helu").
                   Nunca prescribe porcentajes de uso (sin politica causal de pitch mix).
  Targets        = rasgos fisicos del perfil de 22 para su tipo (zona P25-P75 del quintil superior de
                   Stuff+ en Extreme): 'Explore' si el valor actual esta fuera de la zona del lado desfavorable
                   (contra el signo del lift); 'Maintain' si esta dentro o mas alla en la direccion favorable. Exploratorios, no recetas causales. No incluye uso.
  Valor/100      = valor realizado por pitch (RE LMB por conteo x outs, perspectiva del pitcher) x 100: runs
                   ahorrados por 100 pitches respecto al promedio del estado. Ruidoso (guia 35); solo para senales.
  Equipos        = reconstruidos por co-aparicion: pitchers que lanzan en la misma mitad de entrada de
                   >= 3 juegos y >= 50% de las apariciones del menos activo. 'Local' = categoria de
                   altitud mayoritaria de los juegos donde pitchea la parte alta. Anonimos: el dataset no
                   identifica a Diablos; el roster real se configura en el dashboard.
"""

import importlib.util
import sys
import uuid
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUT = BASE_DIR / "dashboard" / "data"
WINDOWS = {2025: "2024 -> 2025", 2026: "2024-2025 -> 2026"}
CODE = {"Four-Seam": "FF", "Sinker": "SI", "Cutter": "FC", "Slider": "SL", "Curveball": "CU", "Changeup": "CH", "Splitter": "FS"}
EVID_EN = {"Robusto": "Robust", "Condicional": "Conditional", "Incierto": "Uncertain"}
MIN_FIT, MIN_PITCH, MIN_ENV = 100, 50, 20


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, BASE_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


t22 = _load("profile_recommendations_export", "22_profile_recommendations_export.py")
t21, t18, t16 = t22.t21, t22.t18, t22.t16


def uuid_dt(u):
    try:
        x = uuid.UUID(str(u))
        # Ticks de 100 ns desde 1582-10-15; 0x01B21DD213814000 = offset al epoch Unix.
        return pd.Timestamp((x.time - 0x01B21DD213814000) * 100, unit="ns") if x.version == 1 else pd.NaT
    except (ValueError, TypeError):
        return pd.NaT


def teams(raw, year):
    x = raw[raw.year == year].drop_duplicates(["pitcher_anon_id", "game_anon_id", "Top/Bottom"])
    x = x.assign(side=x.game_anon_id.astype(str) + "|" + x["Top/Bottom"].astype(str))
    apps = x.groupby("pitcher_anon_id").size()
    w = defaultdict(int)
    for _side, g in x.groupby("side"):
        ps = sorted(g.pitcher_anon_id.unique())
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                w[(ps[i], ps[j])] += 1
    parent = {}

    def f(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for (a, b), c in w.items():
        if c >= 3 and c >= 0.5 * min(apps[a], apps[b]):
            parent[f(a)] = f(b)
    comp = pd.Series({p: f(p) for p in x.pitcher_anon_id.unique()})
    x["team_root"] = x.pitcher_anon_id.map(comp)
    size = comp.value_counts()
    home = (x[x["Top/Bottom"].astype(str) == "Top"].drop_duplicates(["team_root", "game_anon_id"])
            .groupby("team_root").altitude_category.agg(lambda s: s.astype(str).value_counts().idxmax()))
    rows, counters = [], defaultdict(int)
    for root in size[size >= 8].sort_values(ascending=False).index:
        env = home.get(root, "None")
        short = {"Extreme Altitude": "Extreme", "Medium Altitude": "Medium", "No Altitude": "No Alt"}.get(env, "?")
        counters[short] += 1
        rows.append({"team_root": root, "team_label": f"Equipo {short}-{counters[short]}", "team_home_env": env, "team_size": int(size[root])})
    tl = pd.DataFrame(rows)
    out = comp.rename("team_root").reset_index().rename(columns={"index": "pitcher_anon_id"}).merge(tl, on="team_root", how="left")
    out["year"] = year
    return out.drop(columns="team_root")


def roles(raw, year):
    a = (raw[raw.year == year].groupby(["pitcher_anon_id", "game_anon_id"])
         .agg(n=("PitchUID", "size"), first_inn=("Inning", "min")).reset_index())
    p = a.groupby("pitcher_anon_id").agg(apps=("n", "size"), pitches_per_app=("n", "median"), start_share=("first_inn", lambda s: (s == 1).mean()))
    p["role"] = np.where((p.pitches_per_app >= 50) & (p.start_share >= 0.5), "SP", "RP")
    return p.reset_index().assign(year=year)


def pct_band(x, ref):
    q = np.nanpercentile(ref, [25, 50, 75])
    return np.select([x >= q[2], x >= q[1], x >= q[0]], ["Strong Fit", "Moderate Fit", "Low Fit"], "Poor Fit")


def pct_rank(v, ref):
    """Percentil de v en ref; NaN si v falta (sin esto, un valor faltante contaba como percentil 0 = el peor)."""
    ref = pd.Series(ref).dropna()
    return np.nan if v != v or ref.empty else float((ref <= v).mean())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mods = t16.load_modules(); cfg = t18.build_cfg(mods)
    bundle = joblib.load(BASE_DIR / "18_three_scores_bundle.pkl")
    fac_all = t22.factors()
    ev = pd.read_csv(BASE_DIR / "21_evidence_levels.csv").set_index("AutoPitchType")
    ev_map = {t: EVID_EN[ev.loc[t, "nivel_evidencia"]] for t in ev.index}
    prof = pd.read_csv(BASE_DIR / "22_pitch_type_profiles.csv")
    proj_all = pd.read_parquet(BASE_DIR / "22_pitch_projection.parquet")
    sc_all = pd.read_csv(BASE_DIR / "18_three_scores.csv", usecols=["window", "PitchUID", "pitcher_anon_id", "AutoPitchType", "PitcherThrows",
                                                                    "altitude_category", "information_level", "stuff_plus", "location_plus", "pitching_plus",
                                                                    "realized_value_pitcher"])
    raw = pd.read_parquet(BASE_DIR / "stuff_model_df.parquet", columns=["PitchUID", "pitcher_anon_id", "game_anon_id", "year", "Inning", "Top/Bottom",
                                                                         "altitude_category", "PitcherThrows", "AutoPitchType", "RelSpeed", "RelHeight", "RelSide",
                                                                         "Extension", "SpinRate", "SpinAxis", "InducedVertBreak", "HorzBreak", "is_whiff",
                                                                         "outside_strike_zone", "swung_outside_strike_zone"]).dropna(subset=["pitcher_anon_id"])
    for c in ["AutoPitchType", "altitude_category", "PitcherThrows"]:
        raw[c] = raw[c].astype(str)
    raw["HB_r"] = raw["HorzBreak"] * raw["PitcherThrows"].astype(str).map({"Left": -1.0}).fillna(1.0)
    left = raw["PitcherThrows"].eq("Left")
    raw["SA_r"] = np.where(left, 360 - raw["SpinAxis"], raw["SpinAxis"])
    raw["SA_sin"], raw["SA_cos"] = np.sin(np.radians(raw["SA_r"])), np.cos(np.radians(raw["SA_r"]))
    raw["RelSide_r"] = raw["RelSide"] * raw["PitcherThrows"].astype(str).map({"Left": -1.0}).fillna(1.0)
    # Pitches sin movimiento medido (IVB o HB faltantes): la transformacion de altitud no los mueve y 22 les deja delta 0,
    # lo que inflaba su Stuff+ proyectado en Harp Helu. Se tratan como sin proyeccion, igual que los que no tienen altitud.
    sin_mov = set(raw.loc[raw["InducedVertBreak"].isna() | raw["HorzBreak"].isna(), "PitchUID"])
    seq = pd.read_parquet(BASE_DIR / "17_pitch_sequence.parquet", columns=["PitchUID"])
    seq["date"] = seq["PitchUID"].map(uuid_dt)

    pitchers, arsenal, fps, inbox, targets, trends, team_rows = [], [], [], [], [], [], []
    for year, win in WINDOWS.items():
        entry, fac = bundle[win], fac_all[win]
        sc = sc_all[sc_all.window == win].copy()
        pr = proj_all[proj_all.window == win][["PitchUID", "stuff_nivel_mar", "stuff_harp_helu", "delta_oficial"]]
        sc = sc.merge(pr, on="PitchUID", how="left").merge(seq, on="PitchUID", how="left")
        sc.loc[sc["PitchUID"].isin(sin_mov), ["stuff_nivel_mar", "stuff_harp_helu", "delta_oficial"]] = np.nan
        sc["delta"] = sc["stuff_harp_helu"] - sc["stuff_nivel_mar"]

        # Componentes fisicos del Stuff+ (whiff, chase) a nivel del mar vs Harp Helu, con el MISMO esquema que el
        # Stuff+ proyectado (guia 38): nivel = variante oficial a nivel del mar; efecto de altitud = variante fisica
        # (sin arsenal). Antes el efecto salia de la variante con arsenal y podia contradecir al delta mostrado.
        F, ars = entry["future_frame"], entry["f_ars"]
        alt = F["altitude_category"].astype(str).to_numpy()
        fam = entry["families"]["stuff"]
        comp = {c: np.full(len(F), np.nan) for c in ["whiff_per_pitch_mar", "whiff_per_pitch_harp", "chase_mar", "chase_harp"]}
        for env in ["No Altitude", "Medium Altitude", "Extreme Altitude"]:
            m = alt == env
            if not m.any():
                continue
            x_mar, x_harp = t22.to_env(F[m], env, "No Altitude", fac), t22.to_env(F[m], env, "Extreme Altitude", fac)
            no_ars = np.zeros(int(m.sum()), bool)
            for c in ["whiff_per_pitch", "chase"]:
                pred = lambda x, a_: t16.family_predict(fam["models"][c], x, a_, fam["variants"])
                comp[f"{c}_mar"][m] = pred(x_mar, ars[m])
                comp[f"{c}_harp"][m] = comp[f"{c}_mar"][m] + pred(x_harp, no_ars) - pred(x_mar, no_ars)
        compdf = pd.DataFrame(comp); compdf["PitchUID"] = F["PitchUID"].to_numpy()
        sc = sc.merge(compdf, on="PitchUID", how="left")

        r = raw[raw.year == year]
        rl = roles(raw, year); tm = teams(raw, year)
        team_rows.append(tm)

        # ---------------- pitcher x tipo ----------------
        g = sc.groupby(["pitcher_anon_id", "AutoPitchType"])
        a = g.agg(n=("PitchUID", "size"), stuff_plus=("stuff_plus", "mean"), location_plus=("location_plus", "mean"),
                  pitching_plus=("pitching_plus", "mean"), stuff_mar=("stuff_nivel_mar", "mean"), stuff_harp=("stuff_harp_helu", "mean"),
                  delta=("delta", "mean"), delta_oficial=("delta_oficial", "mean"), whiff_mar=("whiff_per_pitch_mar", "mean"), whiff_harp=("whiff_per_pitch_harp", "mean"),
                  chase_mar=("chase_mar", "mean"), chase_harp=("chase_harp", "mean")).reset_index()
        a["usage"] = a["n"] / a.groupby("pitcher_anon_id")["n"].transform("sum")
        obs = sc[sc.altitude_category.isin(["No Altitude", "Extreme Altitude"])].groupby(["pitcher_anon_id", "AutoPitchType", "altitude_category"])["stuff_plus"].agg(["size", "mean"]).unstack()
        obs = obs[(obs[("size", "No Altitude")] >= MIN_ENV) & (obs[("size", "Extreme Altitude")] >= MIN_ENV)]
        od = (obs[("mean", "Extreme Altitude")] - obs[("mean", "No Altitude")]).rename("delta_observado").reset_index()
        a = a.merge(od, on=["pitcher_anon_id", "AutoPitchType"], how="left")
        phys = r.groupby(["pitcher_anon_id", "AutoPitchType"]).agg(velo=("RelSpeed", "mean"), rel_height=("RelHeight", "mean"),
                                                                    extension=("Extension", "mean"), spin=("SpinRate", "mean"),
                                                                    ivb=("InducedVertBreak", "mean"), hb=("HB_r", "mean"), rel_side=("RelSide_r", "mean"),
                                                                    sa_sin=("SA_sin", "mean"), sa_cos=("SA_cos", "mean")).reset_index()
        phys["spin_axis"] = np.degrees(np.arctan2(phys.pop("sa_sin"), phys.pop("sa_cos"))) % 360
        a = a.merge(phys, on=["pitcher_anon_id", "AutoPitchType"], how="left")
        a["code"] = a["AutoPitchType"].map(CODE)
        a["evidence"] = np.where(a["n"] < MIN_PITCH, "Uncertain", a["AutoPitchType"].map(ev_map).fillna("Uncertain"))
        robust = a["AutoPitchType"].map(ev_map).eq("Robust")
        # Fit por pitch = calidad proyectada en Harp Helu frente a pitches del MISMO tipo (>= 50 pitches).
        a["fit"] = "Sin muestra"
        a["exceso"] = np.nan
        a["exceso_pct"] = np.nan
        for t, idx in a.groupby("AutoPitchType").groups.items():
            g_ = a.loc[idx]; ok = (g_.n >= MIN_PITCH) & g_.delta.notna() & g_.stuff_mar.notna()
            if ok.sum() < 10:
                continue
            a.loc[g_.index[ok], "fit"] = pct_band(g_.loc[ok, "stuff_harp"].to_numpy(), g_.loc[ok, "stuff_harp"].to_numpy())
            # Exceso de perdida = residual del delta tras controlar el Stuff+ a nivel del mar dentro del tipo
            # (los mejores pitches pierden mas puntos; corr -0.52). Negativo = pierde mas de lo esperado para su nivel.
            b1, b0 = np.polyfit(g_.loc[ok, "stuff_mar"], g_.loc[ok, "delta"], 1)
            a.loc[idx, "exceso"] = g_["delta"] - (b0 + b1 * g_["stuff_mar"])
            ref_ = a.loc[g_.index[ok], "exceso"]
            a.loc[idx, "exceso_pct"] = a.loc[idx, "exceso"].apply(lambda v: pct_rank(v, ref_))
        a["action"] = np.select([(a.exceso_pct <= 0.10) & robust & (a.n >= MIN_PITCH), (a.exceso_pct <= 0.25) & (a.n >= MIN_PITCH)],
                                ["Adjust", "Review"], "Maintain")
        a["year"] = year
        # Movimiento por entorno: observado si hay >= 20 pitches; si no, proyectado con los factores de 20/21.
        mv = r[r.altitude_category.isin(["No Altitude", "Medium Altitude", "Extreme Altitude"])].groupby(
            ["pitcher_anon_id", "AutoPitchType", "altitude_category"]).agg(n=("PitchUID", "size"), ivb=("InducedVertBreak", "mean"), hb=("HB_r", "mean")).reset_index()
        for (pid, ptype), gg in mv.groupby(["pitcher_anon_id", "AutoPitchType"]):
            gg = gg.set_index("altitude_category")
            ref_env = gg["n"].idxmax(); rI = {"No Altitude": 1.0, **{k: v["InducedVertBreak"] for k, v in fac.items()}}; rH = {"No Altitude": 1.0, **{k: v["HorzBreak"] for k, v in fac.items()}}
            base_ivb, base_hb = gg.loc[ref_env, "ivb"] / rI[ref_env], gg.loc[ref_env, "hb"] / rH[ref_env]
            for env in ["No Altitude", "Extreme Altitude"]:
                if env in gg.index and gg.loc[env, "n"] >= MIN_ENV:
                    fps.append({"year": year, "pitcher_anon_id": pid, "AutoPitchType": ptype, "env": env, "ivb": gg.loc[env, "ivb"], "hb": gg.loc[env, "hb"], "fuente": "observado", "n": int(gg.loc[env, "n"])})
                else:
                    fps.append({"year": year, "pitcher_anon_id": pid, "AutoPitchType": ptype, "env": env, "ivb": base_ivb * rI[env], "hb": base_hb * rH[env], "fuente": "proyectado", "n": 0})

        # ---------------- pitcher ----------------
        # Promedio ponderado por pitches de los tipos con valor. Un tipo sin proyeccion (p. ej. pitches sin altitude_category)
        # ya no anula el promedio de todo el arsenal; NaN solo si ningun tipo tiene valor.
        wavg = lambda df, col: (np.average(df.loc[df[col].notna(), col], weights=df.loc[df[col].notna(), "n"])
                                if df[col].notna().any() else np.nan)
        prow = []
        for pid, gg in a.groupby("pitcher_anon_id"):
            n = gg["n"].sum()
            od_ok = gg.dropna(subset=["delta_observado"])
            row = {"year": year, "pitcher_anon_id": pid, "n": int(n), "stuff_plus": wavg(gg, "stuff_plus"), "location_plus": wavg(gg, "location_plus"),
                   "pitching_plus": wavg(gg, "pitching_plus"), "stuff_mar": wavg(gg, "stuff_mar"), "stuff_harp": wavg(gg, "stuff_harp"),
                   "delta": wavg(gg, "delta"), "delta_oficial": wavg(gg, "delta_oficial"),
                   "exceso": np.average(gg["exceso"].fillna(0), weights=gg["n"]) if gg["exceso"].notna().any() else np.nan,
                   "delta_observado": np.average(od_ok["delta_observado"], weights=od_ok["n"]) if len(od_ok) else np.nan,
                   "share_robust": gg.loc[gg.AutoPitchType.map(ev_map).eq("Robust"), "n"].sum() / n,
                   "share_uncertain": gg.loc[gg.AutoPitchType.map(ev_map).eq("Uncertain"), "n"].sum() / n}
            red = (gg[gg.usage >= 0.15]["action"] == "Adjust").any()
            yellow = (gg[gg.usage >= 0.30]["action"] != "Maintain").any()
            row["alert"] = "red" if red else "yellow" if yellow else "green"
            # Mas afectado / mas resiliente = exceso (pierde mas o menos de lo esperado para su nivel y tipo), igual que las
            # alertas; el delta bruto senalaba casi siempre a la recta. Mejor pitch = calidad proyectada en Harp Helu.
            cand = gg[gg.usage >= 0.10] if (gg.usage >= 0.10).any() else gg
            key = "exceso" if cand["exceso"].notna().any() else "delta"
            cand = cand[cand[key].notna()] if cand[key].notna().any() else cand
            row["worst_pitch"] = cand.sort_values(key).iloc[0]["AutoPitchType"]
            row["most_resilient_pitch"] = cand.sort_values(key, ascending=False).iloc[0]["AutoPitchType"]
            bkey = "stuff_harp" if gg["stuff_harp"].notna().any() else "stuff_plus"
            row["best_pitch"] = gg.sort_values(bkey, ascending=False).iloc[0]["AutoPitchType"]
            prow.append(row)
        p = pd.DataFrame(prow)
        val = sc.groupby("pitcher_anon_id")["realized_value_pitcher"].mean().mul(100).rename("value_per_100")
        p = p.merge(val, on="pitcher_anon_id", how="left")
        hand = sc.assign(PitcherThrows=sc["PitcherThrows"].astype(str)).groupby("pitcher_anon_id")["PitcherThrows"].agg(lambda s: s.value_counts().idxmax())
        p["hand"] = p["pitcher_anon_id"].map(hand).map({"Right": "RHP", "Left": "LHP"}).fillna("—")
        p = p.merge(rl[["pitcher_anon_id", "role", "apps", "pitches_per_app"]], on="pitcher_anon_id", how="left").merge(
            tm[["pitcher_anon_id", "team_label", "team_home_env"]], on="pitcher_anon_id", how="left")
        elig = p.n >= MIN_FIT
        p["fit"] = np.where(elig & p["stuff_harp"].notna(), pct_band(p["stuff_harp"].to_numpy(), p.loc[elig, "stuff_harp"]), "Sin muestra")
        p["fit_pct"] = p["stuff_harp"].apply(lambda v: pct_rank(v, p.loc[elig, "stuff_harp"]))
        p["sens_pct"] = p["delta"].apply(lambda v: pct_rank(v, p.loc[elig, "delta"]))  # alto = pierde menos
        p["exceso_pct"] = p["exceso"].apply(lambda v: pct_rank(v, p.loc[elig, "exceso"]))  # alto = pierde menos de lo esperado
        p["confidence"] = np.select([(p.share_robust >= 0.6) & (p.n >= 300), (p.n < 150) | (p.share_uncertain >= 0.4)], ["Robust", "Uncertain"], "Conditional")
        # Tendencia semanal (Stuff+ medio por semana ISO con >= 20 pitches).
        sc["week"] = sc["date"].dt.to_period("W").dt.start_time
        tw = sc.groupby(["pitcher_anon_id", "week"]).agg(n=("PitchUID", "size"), stuff_plus=("stuff_plus", "mean")).reset_index()
        tw = tw[tw.n >= 10].assign(year=year)
        trends.append(tw)
        pitchers.append(p); arsenal.append(a)

        # ---------------- inbox ----------------
        fpy = pd.DataFrame([r for r in fps if r["year"] == year]).set_index(["pitcher_anon_id", "AutoPitchType", "env"])

        def mov_text(pid, ptype):
            # componente que mas cambia (IVB en rectas, HB en cambios), nivel del mar -> Harp Helu
            try:
                m0, m1 = fpy.loc[(pid, ptype, "No Altitude")], fpy.loc[(pid, ptype, "Extreme Altitude")]
            except KeyError:
                return ""
            c, lab = max([("ivb", "IVB"), ("hb", "HB")], key=lambda t: abs(m1[t[0]] - m0[t[0]]))
            src = "observado" if (m0.fuente == "observado") and (m1.fuente == "observado") else "proyectado"
            return f"{lab} {src} {m0[c]:.1f} → {m1[c]:.1f} in · "

        for _, pp in p[p.alert.isin(["red", "yellow"])].iterrows():
            gg = a[(a.pitcher_anon_id == pp.pitcher_anon_id) & (a.usage >= (0.15 if pp.alert == "red" else 0.30)) & (a.action != "Maintain")].sort_values("delta").iloc[0]
            inbox.append({"year": year, "pitcher_anon_id": pp.pitcher_anon_id, "type": "Needs attention" if pp.alert == "red" else "Review",
                          "alert": pp.alert, "AutoPitchType": gg.AutoPitchType, "code": gg.code, "delta": gg.delta,
                          "headline": (f"El {gg.AutoPitchType} pierde {abs(gg.delta):.1f} de Stuff+ en Harp Helú, "
                                       f"{abs(gg.exceso):.1f} más de lo esperado para un {gg.AutoPitchType} de su nivel"),
                          "evidence_text": (f"{mov_text(pp.pitcher_anon_id, gg.AutoPitchType)}whiff modelado "
                                            f"{100 * gg.whiff_mar:.1f}% → {100 * gg.whiff_harp:.1f}% · n = {int(gg.n)}"),
                          "confidence": gg.evidence, "usage": gg.usage})
        opp = a[(a.stuff_harp >= 105) & (a.exceso_pct >= 0.5) & (a.n >= MIN_PITCH)].merge(p[["pitcher_anon_id", "n"]].rename(columns={"n": "n_p"}), on="pitcher_anon_id")
        opp = opp[opp.n_p >= MIN_FIT]
        for _, gg in opp.iterrows():
            inbox.append({"year": year, "pitcher_anon_id": gg.pitcher_anon_id, "type": "Opportunity", "alert": "green", "AutoPitchType": gg.AutoPitchType,
                          "code": gg.code, "delta": gg.delta, "headline": f"El {gg.AutoPitchType} se proyecta élite en Harp Helú (Stuff+ {gg.stuff_harp:.0f}) y pierde menos de lo esperado (Δ {gg.delta:+.1f})",
                          "evidence_text": f"Stuff+ proyectado {gg.stuff_mar:.0f} → {gg.stuff_harp:.0f} · n = {int(gg.n)}", "confidence": gg.evidence, "usage": gg.usage})

        # ---------------- targets ----------------
        feat_map = {"RelSpeed": ("velo", "Velocidad", "mph"), "InducedVertBreak": ("ivb", "IVB", "in"), "HorzBreak": ("hb", "HB", "in"),
                    "SpinRate": ("spin", "Spin", "rpm"), "Extension": ("extension", "Extensión", "ft"), "RelHeight": ("rel_height", "Altura de release", "ft")}
        pf = prof[prof.en_perfil & prof.feature.isin(feat_map)]
        for _, gg in a[a.n >= MIN_PITCH].iterrows():
            for _, f in pf[pf.AutoPitchType == gg.AutoPitchType].iterrows():
                col, label, unit = feat_map[f.feature]
                # Estado explicito de cada numero: la zona del perfil se midio en pitches lanzados en Extreme, asi que
                # el movimiento se compara en Harp Helu; baseline = el mismo pitch a nivel del mar (fingerprint).
                # Velocidad, spin y release no cambian con el parque: promedio de todos sus pitches.
                if col in ("ivb", "hb"):
                    try:
                        base = float(fpy.loc[(gg.pitcher_anon_id, gg.AutoPitchType, "No Altitude"), col])
                        cur = float(fpy.loc[(gg.pitcher_anon_id, gg.AutoPitchType, "Extreme Altitude"), col])
                    except KeyError:
                        continue
                    estado = "movimiento"
                else:
                    base = cur = gg[col]
                    estado = "todos los parques"
                if pd.isna(cur):
                    continue
                inside = f.zona_p25 <= cur <= f.zona_p75
                # Del lado favorable de la zona (mas alla en la direccion del lift) no es un problema.
                beyond = (f.lift_extreme_2025 > 0 and cur > f.zona_p75) or (f.lift_extreme_2025 < 0 and cur < f.zona_p25)
                verb = "Maintain" if (inside or beyond) else "Explore"
                if inside:
                    direction = "Ya está en la zona de los pitches de Stuff+ alto en Extreme"
                elif beyond:
                    direction = f"Más allá de la zona en la dirección favorable ({'más' if f.lift_extreme_2025 > 0 else 'menos'} {label.lower()} se asocia a más Stuff+)"
                else:
                    direction = (f"Los pitches de Stuff+ alto en Extreme tienen {'más' if f.lift_extreme_2025 > 0 else 'menos'} {label.lower()}: "
                                 f"{f.zona_p25:.1f}–{f.zona_p75:.1f} {unit} (lift {f.lift_extreme_2025:+.2f})")
                targets.append({"year": year, "pitcher_anon_id": gg.pitcher_anon_id, "AutoPitchType": gg.AutoPitchType, "code": gg.code, "variable": label,
                                "unit": unit, "baseline": base, "harp": cur, "estado": estado, "current": cur, "zone_lo": f.zona_p25, "zone_hi": f.zona_p75, "verb": verb, "direction": direction,
                                "confidence": "Conditional" if gg.evidence == "Robust" else gg.evidence})

    pd.concat(pitchers).to_parquet(OUT / "pitchers.parquet", index=False)
    pd.concat(arsenal).to_parquet(OUT / "arsenal.parquet", index=False)
    pd.DataFrame(fps).to_parquet(OUT / "fingerprint.parquet", index=False)
    pd.DataFrame(inbox).to_parquet(OUT / "inbox.parquet", index=False)
    pd.DataFrame(targets).to_parquet(OUT / "targets.parquet", index=False)
    pd.concat(trends).to_parquet(OUT / "trends.parquet", index=False)
    pd.concat(team_rows).to_parquet(OUT / "teams.parquet", index=False)
    # Historial por temporada (scores out-of-fold de la CV por pitcher de 19, incluye 2024).
    oof = pd.read_parquet(BASE_DIR / "19_cv_oof.parquet")
    hist = oof.groupby(["pitcher_anon_id", "year"]).agg(n=("PitchUID", "size"), stuff_plus=("stuff_plus", "mean"),
                                                        location_plus=("location_plus", "mean"), pitching_plus=("pitching_plus", "mean")).reset_index()
    hist[hist.n >= 50].to_parquet(OUT / "history.parquet", index=False)
    hist_t = oof.groupby(["pitcher_anon_id", "year", "AutoPitchType"]).agg(n=("PitchUID", "size"), stuff_plus=("stuff_plus", "mean")).reset_index()
    hist_t[hist_t.n >= 20].to_parquet(OUT / "history_by_type.parquet", index=False)
    # Tablas del perfil ideal y de evidencia (22, 21) para Scouting.
    for src_, dst in [("22_pitch_type_profiles.csv", "profiles"), ("22_profile_outcome_validation.csv", "profile_validation"),
                      ("22_stuff_stability_by_type.csv", "stability_by_type"), ("21_evidence_levels.csv", "evidence_levels"),
                      ("22_recommendations.csv", "type_recommendations"), ("21_delta_by_pitch_type.csv", "delta_by_type")]:
        pd.read_csv(BASE_DIR / src_).to_parquet(OUT / f"{dst}.parquet", index=False)
    meta = {"factores": fac_all, "evidencia_por_tipo": ev_map,
            "umbrales": {"MIN_FIT": MIN_FIT, "MIN_PITCH": MIN_PITCH, "MIN_ENV": MIN_ENV}}
    pd.Series({"meta": str(meta)}).to_json(OUT / "meta.json")
    for f in sorted(OUT.glob("*.parquet")):
        print(f.name, len(pd.read_parquet(f)))


if __name__ == "__main__":
    main()
