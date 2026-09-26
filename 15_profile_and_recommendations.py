"""Perfil fisico/arsenal por pitch type en Extreme Altitude y recomendaciones operativas.

Decisiones congeladas (25/09/2026): V1 (pesos iguales) es la escala oficial de
Stuff+/Pitching+; V2 (pesos ligados a valor) es sensibilidad. Las
recomendaciones no salen solo de V1 ni solo de V2: combinan efecto fisico
(11), estabilidad entre ventanas, bootstrap, chequeo inverso y sensibilidad
V1/V2 (14). "Extreme Altitude" es proxy del entorno Harp Helu.

Perfil = ZONAS de caracteristicas, no un optimo puntual
  - Para cada pitch type se comparan los pitches del quintil superior de Stuff+
    V1 en Extreme Altitude contra todos los pitches del tipo en ese entorno:
    lift = (media del quintil superior - media de todos) / SD, con IC95%
    bootstrap por pitcher. La zona es el rango P25-P75 del quintil superior.
  - Como V1 pondera mucho ubicacion (called strike, chase), el perfil FISICO se
    lee por el lift: una caracteristica sin lift no discrimina y no se reporta
    como zona. Una caracteristica se usa en el perfil solo si |lift| >= 0.10
    (umbral interno), el IC excluye 0 y el signo se repite en la validacion.
  - Horizontales espejados por lateralidad (HorzBreak, RelSide, HorzApprAngle,
    hb_diff_vs_fb se multiplican por -1 en zurdos); sin eso se mezclan signos.
  - Se comparan tambien los lifts en No Altitude: 'compartido con nivel del mar'
    vs 'especifico de altitud'.

Validacion temporal: unicos scores fuera de muestra disponibles: 2025 (modelos
de 2024) y 2026 (modelos de 2024-25). El perfil se APRENDE con 2025 y se VALIDA
en 2026 (los scores de 2024 serian in-sample). Score de ajuste = suma de
lift x z de las caracteristicas usadas. Validacion: correlacion de rangos entre
ajuste y Stuff+ V1 en 2026, brecha entre quintil alto y bajo de ajuste, y a
nivel pitcher x tipo, con bootstrap por pitcher, contra un baseline de solo
velocidad. Un perfil solo se marca 'validado' si ambos IC excluyen 0.

Validacion INDEPENDIENTE: correlacionar el ajuste con el propio Stuff+ es en gran
parte circular (el perfil recupera las features que el modelo ya usa). Por eso el
criterio de "perfil validado" usa el valor realizado por pitch en 2026 (perfectiva
del pitcher, de 13_count_value_weights.py, con conteo-valor aprendido solo con
2024-25): brecha entre el quintil alto y bajo de ajuste en Extreme, con IC95%
bootstrap por pitcher, contra un baseline de solo velocidad. La brecha en whiff
observado y la correlacion con Stuff+ se reportan como secundarias.

Ajuste de scouting: VertApprAngle y HorzApprAngle dependen de donde cruza el
pitch el plato (ubicacion), no solo de propiedades fisicas: se reportan como
descriptivos y NO entran al score de ajuste ni a las zonas.

Sensibilidad V2: el lift tambien se calcula con el score V2 (pesos de 13); una
caracteristica cuyo signo cambia entre V1 y V2 depende de la ponderacion (p.ej.
IVB en Four-Seam: V1 premia IVB bajo via groundball).

Limites: perfil aprendido con una sola temporada; Stuff+ V1 mezcla ubicacion;
correlacion no es causalidad; el uso operativo real deberia reaprender con todos
los datos disponibles.
"""

import importlib.util
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
FEATURES = ["RelSpeed", "InducedVertBreak", "HorzBreak", "VertApprAngle", "HorzApprAngle", "RelHeight",
            "RelSide", "Extension", "SpinRate", "velo_diff_vs_fb", "ivb_diff_vs_fb", "hb_diff_vs_fb",
            "pitch_usage_pct"]
HORZ = ["HorzBreak", "RelSide", "HorzApprAngle", "hb_diff_vs_fb"]
EXT, NOA = "Extreme Altitude", "No Altitude"
LEARN, VALID = "2024 -> 2025", "2024-2025 -> 2026"
MIN_LIFT = 0.10
LOCATION_DEPENDENT = ["VertApprAngle", "HorzApprAngle"]
ZCOLS = ["z_whiff_per_pitch", "z_whiff_given_swing", "z_groundball_given_contact", "z_called_strike", "z_chase_given_outside_zone"]


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, BASE_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
MIN_ROWS, MIN_PITCHERS, MIN_CELL = 300, 15, 30


def load_cohorts():
    bundle = joblib.load(BASE_DIR / "09_pitching_plus_bundle.pkl")
    scores = pd.read_csv(BASE_DIR / "09_pitching_plus_scores.csv")
    v2 = joblib.load(BASE_DIR / "13_v2_bundle.pkl")
    out = {}
    for label, entry in bundle.items():
        f = entry["future_frame"].copy()
        s = scores[scores["window"] == label].reset_index(drop=True)
        same = (len(f) == len(s) and (f["pitcher_anon_id"].to_numpy() == s["pitcher_anon_id"].to_numpy()).all()
                and (f["AutoPitchType"].astype(str).to_numpy() == s["AutoPitchType"].astype(str).to_numpy()).all())
        if not same:
            raise ValueError(f"{label}: el bundle y 09_pitching_plus_scores.csv no estan alineados.")
        f["stuff_v1"] = s["stuff_plus_score"].to_numpy()
        w2 = v2[label]["weights_norm"]
        f["stuff_v2"] = s[ZCOLS].to_numpy(float) @ w2
        f["altitude"] = f["altitude_category"].astype(str)
        f["type"] = f["AutoPitchType"].astype(str)
        sign = f["PitcherThrows"].astype(str).map({"Right": 1.0, "Left": -1.0})
        f = f[sign.notna()].copy()
        for col in HORZ:
            f[col] = f[col] * sign[sign.notna()]
        out[label] = f
    return out


def stats_for(codes, X, flag, n_p):
    n_all = np.bincount(codes, minlength=n_p).astype(float)
    n_top = np.bincount(codes, weights=flag.astype(float), minlength=n_p)
    sx_all = np.stack([np.bincount(codes, weights=X[:, j], minlength=n_p) for j in range(X.shape[1])], 1)
    sx_top = np.stack([np.bincount(codes, weights=X[:, j] * flag, minlength=n_p) for j in range(X.shape[1])], 1)
    return n_all, n_top, sx_all, sx_top


def lift_from(mult, st, sd):
    n_all, n_top, sx_all, sx_top = st
    # errstate: avisos falsos de matmul del entorno (guia, seccion 20); los resultados se validan finitos.
    with np.errstate(all="ignore"):
        return ((mult @ sx_top) / (mult @ n_top) - (mult @ sx_all) / (mult @ n_all)) / sd


def lifts(d, B, rng, score="stuff_v1"):
    d = d.dropna(subset=FEATURES)
    X, y = d[FEATURES].to_numpy(float), d[score].to_numpy()
    top = y >= np.quantile(y, 0.8)
    pit = pd.Index(d["pitcher_anon_id"].unique())
    codes = pit.get_indexer(d["pitcher_anon_id"])
    sd = X.std(axis=0)
    sd[sd == 0] = np.nan
    st = stats_for(codes, X, top, len(pit))
    point = lift_from(np.ones(len(pit)), st, sd)
    reps = np.stack([lift_from(np.bincount(rng.integers(0, len(pit), len(pit)), minlength=len(pit)).astype(float), st, sd)
                     for _ in range(B)])
    return {"point": point, "lo": np.nanpercentile(reps, 2.5, axis=0), "hi": np.nanpercentile(reps, 97.5, axis=0),
            "X": X, "top": top, "mu": X.mean(axis=0), "sd": sd, "n": len(d), "n_pitchers": len(pit)}


def rank_corr_boot(a, b, codes, n_p, B, rng):
    ra = pd.Series(a).rank().to_numpy(); rb = pd.Series(b).rank().to_numpy()
    S = np.stack([np.bincount(codes, minlength=n_p).astype(float), np.bincount(codes, weights=ra, minlength=n_p),
                  np.bincount(codes, weights=rb, minlength=n_p), np.bincount(codes, weights=ra * ra, minlength=n_p),
                  np.bincount(codes, weights=rb * rb, minlength=n_p), np.bincount(codes, weights=ra * rb, minlength=n_p)], 1)

    def corr(m):
        n, sa, sb, saa, sbb, sab = m @ S
        return (sab / n - sa * sb / n ** 2) / np.sqrt((saa / n - (sa / n) ** 2) * (sbb / n - (sb / n) ** 2))
    pt = corr(np.ones(n_p))
    reps = [corr(np.bincount(rng.integers(0, n_p, n_p), minlength=n_p).astype(float)) for _ in range(B)]
    return pt, np.nanpercentile(reps, 2.5), np.nanpercentile(reps, 97.5)


def quintile_gap_boot(fit, y, codes, n_p, B, rng):
    hi, lo = fit >= np.quantile(fit, 0.8), fit <= np.quantile(fit, 0.2)
    S = np.stack([np.bincount(codes, weights=hi.astype(float), minlength=n_p), np.bincount(codes, weights=y * hi, minlength=n_p),
                  np.bincount(codes, weights=lo.astype(float), minlength=n_p), np.bincount(codes, weights=y * lo, minlength=n_p)], 1)
    gap = lambda m: (m @ S[:, 1]) / (m @ S[:, 0]) - (m @ S[:, 3]) / (m @ S[:, 2])
    reps = [gap(np.bincount(rng.integers(0, n_p, n_p), minlength=n_p).astype(float)) for _ in range(B)]
    return gap(np.ones(n_p)), np.nanpercentile(reps, 2.5), np.nanpercentile(reps, 97.5)


def env_label(lo, hi, point, lo_no, hi_no, point_no):
    sig_no = lo_no > 0 or hi_no < 0
    if not sig_no:
        return "especifico de altitud"
    return "compartido con nivel del mar" if np.sign(point_no) == np.sign(point) else "opuesto en nivel del mar"


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--bootstrap", type=int, default=300); ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    B = args.bootstrap
    rng = np.random.default_rng(args.seed)
    pd.set_option("display.width", 240)
    cohorts = load_cohorts()
    learn, valid = cohorts[LEARN], cohorts[VALID]

    cv = _load("count_value", "13_count_value_weights.py")
    raw = pd.read_parquet(BASE_DIR / "stuff_model_features_v1.parquet",
                          columns=["PitchUID", "year", "Balls", "Strikes", "PitchCall", "play_result", "hit_type", "is_whiff"])
    tall = cv.classify(raw)
    tall["year"] = raw["year"].to_numpy()
    Vc = cv.fit_count_value(tall[tall["year"].isin([2024, 2025])])
    y_all = pd.Series(cv.realized_value(tall, Vc), index=tall["PitchUID"].to_numpy())
    w_all = pd.Series(raw["is_whiff"].to_numpy(float), index=raw["PitchUID"].to_numpy())
    valid["y_real"] = valid["PitchUID"].map(y_all).to_numpy()
    valid["whiff"] = valid["PitchUID"].map(w_all).to_numpy()

    prof_rows, val_rows, fit_models = [], [], {}
    types = [t for t, g in learn[learn["altitude"] == EXT].groupby("type")
             if len(g) >= MIN_ROWS and g["pitcher_anon_id"].nunique() >= MIN_PITCHERS]

    def fit_of(frame, model):
        X = frame[FEATURES].to_numpy(float)
        with np.errstate(all="ignore"):
            return ((X[:, model["idx"]] - model["mu"][model["idx"]]) / model["sd"][model["idx"]]) @ model["w"]

    def gap_on(d, fit, vals, scale):
        ok = ~np.isnan(vals)
        pit = pd.Index(d.loc[ok, "pitcher_anon_id"].unique())
        codes = pit.get_indexer(d.loc[ok, "pitcher_anon_id"])
        g, lo, hi = quintile_gap_boot(fit[ok], vals[ok], codes, len(pit), B, rng)
        return g * scale, lo * scale, hi * scale

    for t in sorted(types):
        sel = lambda fr, env: fr[(fr["type"] == t) & (fr["altitude"] == env)]
        L, Ln, V = lifts(sel(learn, EXT), B, rng), lifts(sel(learn, NOA), B, rng), lifts(sel(valid, EXT), B, rng)
        L2 = lifts(sel(learn, EXT), B, rng, score="stuff_v2")
        for arr in (L["point"], L["lo"], L["hi"]):
            if not np.isfinite(arr).all():
                raise ValueError(f"{t}: lifts no finitos.")
        dl = sel(learn, EXT).dropna(subset=FEATURES)
        zone = dl[FEATURES][L["top"]].quantile([.25, .5, .75])
        used = []
        for j, f in enumerate(FEATURES):
            loc_dep = f in LOCATION_DEPENDENT
            sig = (L["lo"][j] > 0 or L["hi"][j] < 0) and abs(L["point"][j]) >= MIN_LIFT
            repeats = np.sign(V["point"][j]) == np.sign(L["point"][j]) and (V["lo"][j] > 0 or V["hi"][j] < 0)
            use = bool(sig and repeats and not loc_dep)
            if use:
                used.append(j)
            sig2 = L2["lo"][j] > 0 or L2["hi"][j] < 0
            if not sig2:
                v2s = "V2 no discrimina"
            else:
                v2s = "mismo signo en V2" if np.sign(L2["point"][j]) == np.sign(L["point"][j]) else "signo distinto en V2 (depende de ponderacion)"
            prof_rows.append({
                "AutoPitchType": t, "feature": f, "depende_de_ubicacion": loc_dep,
                "lift_extreme_2025": L["point"][j], "lift_lo": L["lo"][j], "lift_hi": L["hi"][j],
                "lift_extreme_2026_validation": V["point"][j], "val_lo": V["lo"][j], "val_hi": V["hi"][j],
                "lift_no_altitude_2025": Ln["point"][j], "no_lo": Ln["lo"][j], "no_hi": Ln["hi"][j],
                "lift_extreme_2025_V2": L2["point"][j], "v2_lo": L2["lo"][j], "v2_hi": L2["hi"][j], "sensibilidad_V2": v2s,
                "zone_p25": zone.loc[.25, f], "zone_p50": zone.loc[.5, f], "zone_p75": zone.loc[.75, f],
                "all_p50": float(np.median(L["X"][:, j])), "usada_en_perfil": use,
                "entorno": env_label(L["lo"][j], L["hi"][j], L["point"][j], Ln["lo"][j], Ln["hi"][j], Ln["point"][j]) if use
                else ("descriptivo (depende de ubicacion)" if loc_dep and sig else "no discriminante o no validada"),
                "n_pitches_learn": L["n"], "n_pitchers_learn": L["n_pitchers"]})
        fit_models[t] = {"idx": used, "w": L["point"][used], "mu": L["mu"], "sd": L["sd"]}

        for env in (EXT, NOA):
            d = sel(valid, env).dropna(subset=FEATURES)
            if len(d) < MIN_ROWS or not used:
                continue
            pit = pd.Index(d["pitcher_anon_id"].unique()); codes = pit.get_indexer(d["pitcher_anon_id"])
            fit = fit_of(d, fit_models[t]); ys = d["stuff_v1"].to_numpy()
            speed = d["RelSpeed"].to_numpy()
            r, rlo, rhi = rank_corr_boot(fit, ys, codes, len(pit), B, rng)
            gy, gylo, gyhi = gap_on(d, fit, d["y_real"].to_numpy(), 100)
            sy, sylo, syhi = gap_on(d, speed, d["y_real"].to_numpy(), 100)
            gw, gwlo, gwhi = gap_on(d, fit, d["whiff"].to_numpy(), 100)
            row = {"AutoPitchType": t, "entorno_validacion": env, "n_pitches": len(d), "n_pitchers": len(pit),
                   "features_usadas": ",".join(FEATURES[i] for i in used),
                   "brecha_valor_realizado_runs_por_100": gy, "brecha_y_lo": gylo, "brecha_y_hi": gyhi,
                   "brecha_valor_solo_velocidad": sy, "brecha_vel_lo": sylo, "brecha_vel_hi": syhi,
                   "brecha_whiff_pp": gw, "brecha_whiff_lo": gwlo, "brecha_whiff_hi": gwhi,
                   "spearman_ajuste_vs_stuff_v1_circular": r, "sp_lo": rlo, "sp_hi": rhi}
            val_rows.append(row)
    profiles, validation = pd.DataFrame(prof_rows), pd.DataFrame(val_rows)
    validated = {}
    for t in types:
        v = validation[(validation["AutoPitchType"] == t) & (validation["entorno_validacion"] == EXT)]
        validated[t] = bool(len(v) and v["brecha_y_lo"].iloc[0] > 0)
    validation["perfil_validado"] = validation["AutoPitchType"].map(validated)
    profiles["perfil_validado"] = profiles["AutoPitchType"].map(validated)
    profiles.to_csv(BASE_DIR / "15_pitch_type_profiles.csv", index=False)
    validation.to_csv(BASE_DIR / "15_profile_validation.csv", index=False)

    f2 = lambda x: f"{x:.2f}"
    print("PERFIL POR PITCH TYPE (aprendido con 2025 en Extreme; horizontales en convencion de diestro). Caracteristicas usadas:")
    show = profiles[profiles["usada_en_perfil"]][["AutoPitchType", "feature", "lift_extreme_2025", "lift_lo", "lift_hi",
                                                  "lift_extreme_2026_validation", "zone_p25", "zone_p75", "all_p50", "entorno", "sensibilidad_V2"]]
    print(show.to_string(index=False, float_format=f2))
    for t in types:
        if not profiles[profiles.AutoPitchType == t]["usada_en_perfil"].any():
            print(f"  {t}: ninguna caracteristica cumple lift >= {MIN_LIFT}, IC sin 0 y validacion en 2026: sin perfil fisico discriminante.")
    print("\nDescriptivas dependientes de ubicacion (no entran al ajuste):")
    print(profiles[profiles["depende_de_ubicacion"] & (profiles["entorno"] != "no discriminante o no validada")][
        ["AutoPitchType", "feature", "lift_extreme_2025", "lift_lo", "lift_hi", "sensibilidad_V2"]].to_string(index=False, float_format=f2))
    print("\nVALIDACION EN 2026 (perfil aprendido con 2025), contra valor realizado por pitch (independiente del modelo):")
    vc = ["AutoPitchType", "entorno_validacion", "n_pitches", "brecha_valor_realizado_runs_por_100", "brecha_y_lo", "brecha_y_hi",
          "brecha_valor_solo_velocidad", "brecha_vel_lo", "brecha_vel_hi", "brecha_whiff_pp", "brecha_whiff_lo", "brecha_whiff_hi", "perfil_validado"]
    print(validation[vc].to_string(index=False, float_format=f2))
    print("(brecha = quintil alto de ajuste menos quintil bajo; runs/100 pitches desde la perspectiva del pitcher; whiff en pp)")

    ev = pd.read_csv(BASE_DIR / "14_evidence_levels.csv")
    mat = pd.read_csv(BASE_DIR / "14_v1_v2_matrix_by_window.csv")
    t11 = pd.read_csv(BASE_DIR / "11_altitude_total_by_pitch_type.csv")
    tr = pd.read_csv(BASE_DIR / "11_altitude_total_transform.csv")
    rec = []
    comp = ["p_whiff_per_pitch", "p_whiff_given_swing", "p_groundball_given_contact", "p_called_strike", "p_chase_given_outside_zone"]
    for _, e in ev.iterrows():
        t = e["AutoPitchType"]
        m = mat[(mat.window == VALID) & (mat.AutoPitchType == t)].iloc[0]
        r11 = t11[(t11.window == VALID) & (t11.AutoPitchType == t)].iloc[0]
        shift = tr[(tr.window == VALID) & (tr.AutoPitchType == t) & (tr.role == "shift")].set_index("variable")["delta_extreme_minus_no"]
        contrib = {c_: r11[f"contrib_{c_}"] for c_ in comp}
        whiff = contrib["p_whiff_per_pitch"] + contrib["p_whiff_given_swing"]
        gb, loc = contrib["p_groundball_given_contact"], contrib["p_called_strike"] + contrib["p_chase_given_outside_zone"]
        mech = (f"cambio fisico esperado (Extreme - No, convencion de diestro) IVB {shift['InducedVertBreak']:+.1f} in, HB {shift['HorzBreak']:+.1f} in; "
                f"contribucion V1: whiff {whiff:+.2f}, groundball {gb:+.2f}, ubicacion (called strike + chase) {loc:+.2f} pts")
        d1, d2, lvl = m["V1_delta"], m["V2_propio_delta"], e["nivel_evidencia"]
        ci1 = f"[{m['V1_lo']:+.2f}, {m['V1_hi']:+.2f}]"
        big = max(abs(d1), abs(d2)) >= 1.0
        if lvl == "Robusto":
            word = "pierde" if d1 < 0 else "gana"
            text = (f"{t}: {word} valor de forma consistente bajo el cambio fisico esperado de altitud (V1 {d1:+.2f} {ci1}, V2 {d2:+.2f} pts); "
                    f"magnitud {'relevante' if big else 'pequena (<1 pt)'}. Usarlo como ajuste marginal en la evaluacion, no como criterio de exclusion.")
        elif lvl == "Condicional":
            text = (f"{t}: la conclusion neta depende de la ponderacion (V1 {d1:+.2f} {ci1} vs V2 {d2:+.2f} pts). El componente de whiff "
                    f"{'cae' if whiff < 0 else 'sube'} {abs(whiff):.2f} pts y groundball {'sube' if gb > 0 else 'cae'} {abs(gb):.2f} pts; el neto en V1 "
                    f"depende de cuanto se valore groundball frente a whiff, y bajo la ponderacion ligada a valor de carrera el efecto es "
                    f"{'negativo' if d2 < 0 else 'positivo'}. No emitir recomendacion categorica; presentar ambos lentes.")
        else:
            text = (f"{t}: no concluir. Historial de {int(e['n_cells_min'])} celdas pitcher x temporada como minimo (umbral 30) o senal inestable entre ventanas "
                    f"(V1 {d1:+.2f} {ci1}, V2 {d2:+.2f} pts).")
        pr = profiles[(profiles.AutoPitchType == t) & profiles["usada_en_perfil"]]
        resumen = "; ".join(f"{r_['feature']} {r_['zone_p25']:.1f} a {r_['zone_p75']:.1f} (lift {r_['lift_extreme_2025']:+.2f}"
                            f"{', depende de ponderacion' if 'distinto' in r_['sensibilidad_V2'] else ''})" for _, r_ in pr.iterrows()) or "sin perfil fisico discriminante"
        rec.append({"AutoPitchType": t, "nivel_evidencia": lvl, "delta_V1_2026": d1, "V1_ci95_lo": m["V1_lo"], "V1_ci95_hi": m["V1_hi"],
                    "delta_V2_2026": d2, "V2_ci95_lo": m["V2_propio_lo"], "V2_ci95_hi": m["V2_propio_hi"],
                    "delta_V2_stuff_2026": m["V2-stuff_delta"], "delta_V1_2025": e["V1_2024->2025"], "delta_V2_2025": e["V2_propio_2024->2025"],
                    "chequeo_inverso_mismo_signo": bool(m["reverse_same_sign"]), "n_celdas_historial_min": int(e["n_cells_min"]),
                    "mecanismo": mech, "perfil_fisico_validado": validated.get(t, False), "perfil_resumen": resumen, "recomendacion": text})
    recdf = pd.DataFrame(rec)
    recdf.to_csv(BASE_DIR / "15_altitude_recommendations.csv", index=False)
    print("\nRECOMENDACIONES (niveles de evidencia de 14):")
    for _, r_ in recdf.iterrows():
        print(f"- [{r_['nivel_evidencia']}] {r_['recomendacion']}")
        print(f"    mecanismo: {r_['mecanismo']}")
        print(f"    perfil (validado={r_['perfil_fisico_validado']}): {r_['perfil_resumen']}")

    rows = []
    for t in types:
        model = fit_models[t]
        if not model["idx"]:
            continue
        d = valid[valid["type"] == t].dropna(subset=FEATURES).copy()
        d["fit"] = fit_of(d, model)
        g = d.groupby("pitcher_anon_id")
        cell = pd.DataFrame({"n_pitches": g.size(), "mean_fit": g["fit"].mean(), "mean_stuff_v1_all": g["stuff_v1"].mean()})
        for env, col in [(EXT, "mean_stuff_v1_extreme"), (NOA, "mean_stuff_v1_no_altitude")]:
            s = d[d["altitude"] == env].groupby("pitcher_anon_id")["stuff_v1"].agg(["size", "mean"])
            cell[col] = s["mean"].where(s["size"] >= 20)
        cell = cell[cell["n_pitches"] >= MIN_CELL].copy()
        cell["fit_percentile_in_type"] = cell["mean_fit"].rank(pct=True) * 100
        cell["AutoPitchType"] = t
        cell["perfil_validado"] = validated[t]
        cell["cercano_al_perfil"] = bool(validated[t]) & (cell["fit_percentile_in_type"] >= 90)
        rows.append(cell.reset_index())
    fits = pd.concat(rows, ignore_index=True)
    ok = fits[fits["perfil_validado"]]
    overall = pd.DataFrame([
        {"pitcher_anon_id": pid, "AutoPitchType": "ALL (validados, ponderado por pitches)", "n_pitches": x["n_pitches"].sum(),
         "fit_percentile_in_type": float(np.average(x["fit_percentile_in_type"], weights=x["n_pitches"])), "perfil_validado": True}
        for pid, x in ok.groupby("pitcher_anon_id")])
    fits = pd.concat([fits, overall], ignore_index=True)
    fits.to_csv(BASE_DIR / "15_pitcher_fit_scores.csv", index=False)
    print(f"\nPitcher fit (cohorte 2026, n>={MIN_CELL} pitches por tipo): {fits['pitcher_anon_id'].nunique()} pitchers; "
          f"cercanos al perfil (percentil>=90 en tipos validados): {int(fits['cercano_al_perfil'].eq(True).sum())} celdas pitcher x tipo")
    print("Tipos con perfil validado (brecha de valor realizado > 0, IC95%):", [t for t, v in validated.items() if v] or "ninguno",
          "| no validados:", [t for t, v in validated.items() if not v] or "ninguno")
    print("\nArchivos: 15_pitch_type_profiles.csv, 15_profile_validation.csv, 15_altitude_recommendations.csv, 15_pitcher_fit_scores.csv")


if __name__ == "__main__":
    main()
