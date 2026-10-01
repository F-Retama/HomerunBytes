"""Capa de datos del dashboard (Pitching Decision System): convierte los outputs congelados del
modelo (18-22) en tablas listas para cada pantalla. No entrena ni cambia ningun modelo.

Todas las reglas de producto (Harp Helu Fit, confianza, semaforo, alertas, inbox, targets) estan
aqui, documentadas y con umbrales explicitos, para que el dashboard solo muestre.

Definiciones
  Temporada      = temporada evaluada fuera de muestra: 2025 (modelo 2024 -> 2025) y 2026 (modelo
                   2024-25 -> 2026). Scores oficiales de 18 (stuff_plus, location_plus, pitching_plus).
  Delta altitud  = Stuff+ proyectado en Harp Helu (proxy Extreme) menos Stuff+ proyectado a nivel del
                   mar (22): mismo pitch, solo cambia el movimiento por el aire. Es neutral al lugar
                   donde se lanzo el pitch.
  Delta observado= Stuff+ real en Extreme menos en No Altitude del mismo pitcher x tipo (>= 20 pitches
                   en ambos); se pondera por el uso total del pitcher. Secundario: incluye adaptacion.
  Harp Helu Fit  = resiliencia del arsenal a la altitud: percentil del Delta altitud (ponderado por uso)
                   entre pitchers con >= 100 pitches en la temporada. Strong >= p75 (pierde menos),
                   Moderate p50-p75, Low p25-p50, Poor < p25. NO es calidad: la calidad es Stuff+.
  Fit por pitch  = mismo criterio de percentiles, pero comparando el Delta del pitch contra los pitches del
                   mismo tipo en la liga (>= 50 pitches): separa 'este four-seam pierde mas que otros four-seam'
                   de 'los four-seam pierden en altitud'.
  Confianza      = por pitcher: Robust si >= 60% de sus pitches son de tipos con evidencia Robusta (21)
                   y tiene >= 300 pitches; Uncertain si < 150 pitches o >= 40% en tipos Inciertos;
                   Conditional en otro caso. Por pitch: el nivel de su tipo (21), Uncertain si < 50 pitches.
  Semaforo pitch = RELATIVO a su tipo (todos los tipos pierden en altitud; un umbral absoluto marcaba a la
                   mitad de la liga): Adjust si Fit por pitch = Poor (cuartil que mas pierde entre pitches del
                   mismo tipo) y el tipo tiene evidencia Robusta; Review si Fit = Poor o Low; Maintain en otro
                   caso. Prioridades de revision, no instrucciones de uso.
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
    seq = pd.read_parquet(BASE_DIR / "17_pitch_sequence.parquet", columns=["PitchUID"])
    seq["date"] = seq["PitchUID"].map(uuid_dt)

    pitchers, arsenal, fps, inbox, targets, trends, team_rows = [], [], [], [], [], [], []
    for year, win in WINDOWS.items():
        entry, fac = bundle[win], fac_all[win]
        sc = sc_all[sc_all.window == win].copy()
        pr = proj_all[proj_all.window == win][["PitchUID", "stuff_nivel_mar", "stuff_harp_helu"]]
        sc = sc.merge(pr, on="PitchUID", how="left").merge(seq, on="PitchUID", how="left")
        sc["delta"] = sc["stuff_harp_helu"] - sc["stuff_nivel_mar"]

        # Componentes fisicos del Stuff+ (whiff, chase) a nivel del mar vs Harp Helu.
        F, ars = entry["future_frame"], entry["f_ars"]
        alt = F["altitude_category"].astype(str).to_numpy()
        fam = entry["families"]["stuff"]
        comp = {c: np.full(len(F), np.nan) for c in ["whiff_per_pitch_mar", "whiff_per_pitch_harp", "chase_mar", "chase_harp"]}
        for env in ["No Altitude", "Medium Altitude", "Extreme Altitude"]:
            m = alt == env
            if not m.any():
                continue
            for tgt, key in [("No Altitude", "mar"), ("Extreme Altitude", "harp")]:
                x = t22.to_env(F[m], env, tgt, fac)
                for c in ["whiff_per_pitch", "chase"]:
                    comp[f"{c}_{key}"][m] = t16.family_predict(fam["models"][c], x, ars[m], fam["variants"])
        compdf = pd.DataFrame(comp); compdf["PitchUID"] = F["PitchUID"].to_numpy()
        sc = sc.merge(compdf, on="PitchUID", how="left")

        r = raw[raw.year == year]
        rl = roles(raw, year); tm = teams(raw, year)
        team_rows.append(tm)

        # ---------------- pitcher x tipo ----------------
        g = sc.groupby(["pitcher_anon_id", "AutoPitchType"])
        a = g.agg(n=("PitchUID", "size"), stuff_plus=("stuff_plus", "mean"), location_plus=("location_plus", "mean"),
                  pitching_plus=("pitching_plus", "mean"), stuff_mar=("stuff_nivel_mar", "mean"), stuff_harp=("stuff_harp_helu", "mean"),
                  delta=("delta", "mean"), whiff_mar=("whiff_per_pitch_mar", "mean"), whiff_harp=("whiff_per_pitch_harp", "mean"),
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
        # Fit por pitch: resiliencia frente a pitches del MISMO tipo en la liga (>= 50 pitches).
        a["fit"] = "Sin muestra"
        for t, idx in a[a.n >= MIN_PITCH].groupby("AutoPitchType").groups.items():
            a.loc[idx, "fit"] = pct_band(a.loc[idx, "delta"].to_numpy(), a.loc[idx, "delta"].to_numpy())
        # Semaforo relativo: todos los tipos pierden en altitud; se alerta a quien pierde mas que su tipo.
        a["action"] = np.select([(a.fit == "Poor Fit") & robust, a.fit.isin(["Poor Fit", "Low Fit"])], ["Adjust", "Review"], "Maintain")
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
        wavg = lambda df, col: np.average(df[col], weights=df["n"]) if df[col].notna().all() else np.nan
        prow = []
        for pid, gg in a.groupby("pitcher_anon_id"):
            n = gg["n"].sum()
            od_ok = gg.dropna(subset=["delta_observado"])
            row = {"year": year, "pitcher_anon_id": pid, "n": int(n), "stuff_plus": wavg(gg, "stuff_plus"), "location_plus": wavg(gg, "location_plus"),
                   "pitching_plus": wavg(gg, "pitching_plus"), "stuff_mar": wavg(gg, "stuff_mar"), "stuff_harp": wavg(gg, "stuff_harp"),
                   "delta": wavg(gg, "delta"),
                   "delta_observado": np.average(od_ok["delta_observado"], weights=od_ok["n"]) if len(od_ok) else np.nan,
                   "share_robust": gg.loc[gg.AutoPitchType.map(ev_map).eq("Robust"), "n"].sum() / n,
                   "share_uncertain": gg.loc[gg.AutoPitchType.map(ev_map).eq("Uncertain"), "n"].sum() / n}
            red = (gg[gg.usage >= 0.15]["action"] == "Adjust").any()
            yellow = (gg[gg.usage >= 0.30]["action"] != "Maintain").any()
            row["alert"] = "red" if red else "yellow" if yellow else "green"
            worst = gg.sort_values("delta").iloc[0]
            row["worst_pitch"] = worst["AutoPitchType"]; row["best_pitch"] = gg.sort_values("stuff_plus", ascending=False).iloc[0]["AutoPitchType"]
            row["most_resilient_pitch"] = gg[gg.usage >= 0.10].sort_values("delta", ascending=False)["AutoPitchType"].iloc[0] if (gg.usage >= 0.10).any() else worst["AutoPitchType"]
            prow.append(row)
        p = pd.DataFrame(prow)
        val = sc.groupby("pitcher_anon_id")["realized_value_pitcher"].mean().mul(100).rename("value_per_100")
        p = p.merge(val, on="pitcher_anon_id", how="left")
        hand = sc.assign(PitcherThrows=sc["PitcherThrows"].astype(str)).groupby("pitcher_anon_id")["PitcherThrows"].agg(lambda s: s.value_counts().idxmax())
        p["hand"] = p["pitcher_anon_id"].map(hand).map({"Right": "RHP", "Left": "LHP"}).fillna("—")
        p = p.merge(rl[["pitcher_anon_id", "role", "apps", "pitches_per_app"]], on="pitcher_anon_id", how="left").merge(
            tm[["pitcher_anon_id", "team_label", "team_home_env"]], on="pitcher_anon_id", how="left")
        ref = p.loc[p.n >= MIN_FIT, "delta"]
        p["fit"] = np.where(p.n >= MIN_FIT, pct_band(p["delta"].to_numpy(), ref), "Sin muestra")
        p["fit_pct"] = p["delta"].rank(pct=True)
        p["confidence"] = np.select([(p.share_robust >= 0.6) & (p.n >= 300), (p.n < 150) | (p.share_uncertain >= 0.4)], ["Robust", "Uncertain"], "Conditional")
        # Tendencia semanal (Stuff+ medio por semana ISO con >= 20 pitches).
        sc["week"] = sc["date"].dt.to_period("W").dt.start_time
        tw = sc.groupby(["pitcher_anon_id", "week"]).agg(n=("PitchUID", "size"), stuff_plus=("stuff_plus", "mean")).reset_index()
        tw = tw[tw.n >= 10].assign(year=year)
        trends.append(tw)
        pitchers.append(p); arsenal.append(a)

        # ---------------- inbox ----------------
        for _, pp in p[p.alert.isin(["red", "yellow"])].iterrows():
            gg = a[(a.pitcher_anon_id == pp.pitcher_anon_id) & (a.usage >= (0.15 if pp.alert == "red" else 0.30)) & (a.action != "Maintain")].sort_values("delta").iloc[0]
            inbox.append({"year": year, "pitcher_anon_id": pp.pitcher_anon_id, "type": "Needs attention" if pp.alert == "red" else "Review",
                          "alert": pp.alert, "AutoPitchType": gg.AutoPitchType, "code": gg.code, "delta": gg.delta,
                          "headline": f"El {gg.AutoPitchType} pierde {abs(gg.delta):.1f} de Stuff+ en Harp Helú, más que la mayoría de los {gg.AutoPitchType} de la liga",
                          "evidence_text": (f"Movimiento IVB {gg.ivb:.1f} → {gg.ivb * fac['Extreme Altitude']['InducedVertBreak']:.1f} in · whiff modelado "
                                            f"{100 * gg.whiff_mar:.1f}% → {100 * gg.whiff_harp:.1f}% · n = {int(gg.n)}"),
                          "confidence": gg.evidence, "usage": gg.usage})
        opp = a[(a.stuff_plus >= 105) & (a.delta > -1.5) & (a.n >= MIN_PITCH)].merge(p[["pitcher_anon_id", "n"]].rename(columns={"n": "n_p"}), on="pitcher_anon_id")
        opp = opp[opp.n_p >= MIN_FIT]
        for _, gg in opp.iterrows():
            inbox.append({"year": year, "pitcher_anon_id": gg.pitcher_anon_id, "type": "Opportunity", "alert": "green", "AutoPitchType": gg.AutoPitchType,
                          "code": gg.code, "delta": gg.delta, "headline": f"El {gg.AutoPitchType} sostiene su calidad en Harp Helú (Stuff+ {gg.stuff_plus:.0f}, Δ {gg.delta:+.1f})",
                          "evidence_text": f"Stuff+ proyectado {gg.stuff_mar:.0f} → {gg.stuff_harp:.0f} · n = {int(gg.n)}", "confidence": gg.evidence, "usage": gg.usage})

        # ---------------- targets ----------------
        feat_map = {"RelSpeed": ("velo", "Velocidad", "mph"), "InducedVertBreak": ("ivb", "IVB", "in"), "HorzBreak": ("hb", "HB", "in"),
                    "SpinRate": ("spin", "Spin", "rpm"), "Extension": ("extension", "Extensión", "ft"), "RelHeight": ("rel_height", "Altura de release", "ft")}
        pf = prof[prof.en_perfil & prof.feature.isin(feat_map)]
        for _, gg in a[a.n >= MIN_PITCH].iterrows():
            for _, f in pf[pf.AutoPitchType == gg.AutoPitchType].iterrows():
                col, label, unit = feat_map[f.feature]
                cur = gg[col]
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
                                "unit": unit, "current": cur, "zone_lo": f.zona_p25, "zone_hi": f.zona_p75, "verb": verb, "direction": direction,
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
