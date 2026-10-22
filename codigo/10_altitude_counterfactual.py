"""Contrafactual de altitud sobre el Pitching+/Stuff+ de 09.

Carga los modelos EXACTOS entrenados por 09_pitching_plus.py
(09_pitching_plus_bundle.pkl); no reentrena ni cambia arquitectura.

Que mide y que NO mide (distincion fijada el 24/09/2026)
--------------------------------------------------------
delta_direct = Score(X, Extreme) - Score(X, No Altitude), con TODAS las demas
features identicas (incluido el movimiento observado). Es el efecto directo,
ajustado, del entorno condicionado en el movimiento observado: "para un pitch
que ya tiene exactamente este IVB/HB, que valor adicional le atribuye el
modelo al entorno". NO responde "que le pasaria fisicamente a este pitch al
llevarlo de nivel del mar a CDMX", porque esa pregunta incluye el cambio de
IVB/HB (y de los angulos de aproximacion) provocado por la atmosfera, que
aqui se mantiene fijo. Tampoco es causal puro (guia, seccion 7).

El efecto total por modelo (delta_total = Score(X^Extreme, Extreme) -
Score(X^No, No), con el movimiento esperado por ambiente) queda pendiente:
mover IVB/HB sin ajustar VertApprAngle/HorzApprAngle (feature #1 en whiff)
daria entradas fisicamente inconsistentes. Como aproximacion inmediata se
reporta un contraste OBSERVACIONAL within-pitcher/pitch-type: diferencia de
medias del score real entre pitches lanzados en Extreme y en No Altitude por
el mismo pitcher y tipo de pitch (celdas con n>=20 en ambos, ponderadas por
el menor n, como en 01). Incluye el efecto directo mas todo lo mediado por
movimiento observado, ubicacion, bateadores y parque, asi que es asociacion,
no efecto aislado. (observado - directo) ~ parte mediada por movimiento y
otros factores.

Lenguaje: la variable disponible es altitude_category, no una identificacion
fisica de Harp Helu. "Extreme Altitude" se reporta como proxy del entorno Harp
Helu. Ademas, la categoria probablemente agrupa pocos parques, asi que el
efecto directo tambien puede absorber diferencias de calibracion/parque.

Dominio: chase se aprendio solo fuera de zona; sobre pitches dentro de zona es
extrapolacion. En el delta ambas copias comparten ubicacion, por lo que ese
error se cancela en gran parte (no en el nivel del score). Aun asi se reporta
el delta por zona y una version sin chase (stuff_plus_no_chase).
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
LEVELS = ["No Altitude", "Medium Altitude", "Extreme Altitude"]
EXTREME_NOTE = "Extreme Altitude (proxy del entorno Harp Helu)"
COMPONENTS = ["whiff_per_pitch", "whiff_given_swing", "groundball", "called_strike", "chase"]
COMPONENT_LABELS = {
    "whiff_per_pitch": "p_whiff_per_pitch",
    "whiff_given_swing": "p_whiff_given_swing",
    "groundball": "p_groundball_given_contact",
    "called_strike": "p_called_strike",
    "chase": "p_chase_given_outside_zone",
}
MIN_CELL = 20
MIN_PITCHER_PITCHES = 100


def score_at_level(entry, frame, level):
    X = frame[entry["features"]].copy()
    cats = X["altitude_category"].cat.categories
    X["altitude_category"] = pd.Categorical([level] * len(X), categories=cats)
    probs, z = {}, {}
    for name in COMPONENTS:
        probs[name] = entry["models"][name].predict_proba(X)[:, 1]
        mean, std = entry["component_stats"][name]
        z[name] = (probs[name] - mean) / std
    raw5 = np.mean([z[n] for n in COMPONENTS], axis=0)
    raw4 = np.mean([z[n] for n in COMPONENTS if n != "chase"], axis=0)
    m5, s5 = entry["pp5"]
    m4, s4 = entry["pp4_no_chase"]
    return {
        "probs": probs,
        "stuff": 100 + 10 * (raw5 - m5) / s5,
        "stuff_no_chase": 100 + 10 * (raw4 - m4) / s4,
    }


def summarize(df, keys):
    grouped = df.groupby(keys, observed=True)["delta_stuff_direct"]
    out = grouped.agg(
        n="count", mean="mean", median="median", sd="std",
        p10=lambda s: s.quantile(.10), p25=lambda s: s.quantile(.25),
        p75=lambda s: s.quantile(.75), p90=lambda s: s.quantile(.90),
    )
    extras = df.groupby(keys, observed=True)[
        ["delta_no_chase"] + [f"d_{COMPONENT_LABELS[c]}" for c in COMPONENTS]
    ].mean()
    return out.join(extras).reset_index()


def add_shape_bins(df):
    def tercile(series):
        return pd.qcut(series, 3, labels=["low", "mid", "high"], duplicates="drop")
    df["ivb_bin"] = df.groupby("AutoPitchType", observed=True)["InducedVertBreak"].transform(
        lambda s: tercile(s).astype(str))
    df["hb_bin"] = df.groupby("AutoPitchType", observed=True)["HorzBreak"].transform(
        lambda s: tercile(s.abs()).astype(str))
    df["shape_bin"] = (df["AutoPitchType"].astype(str) + " | IVB " + df["ivb_bin"]
                       + " | |HB| " + df["hb_bin"])
    return df


def observed_contrast(df):
    """Contraste observacional within-pitcher + pitch type (ver docstring)."""
    actual = df[df["altitude_category"].isin(["No Altitude", "Extreme Altitude"])]
    cell = (actual.groupby(["pitcher_anon_id", "AutoPitchType", "altitude_category"], observed=True)
            .agg(n=("stuff_actual", "size"), stuff=("stuff_actual", "mean"),
                 direct=("delta_stuff_direct", "mean")).reset_index())
    wide = cell.pivot(index=["pitcher_anon_id", "AutoPitchType"], columns="altitude_category",
                      values=["n", "stuff", "direct"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.dropna()
    wide = wide[(wide["n_No Altitude"] >= MIN_CELL) & (wide["n_Extreme Altitude"] >= MIN_CELL)]
    wide["observed_delta"] = wide["stuff_Extreme Altitude"] - wide["stuff_No Altitude"]
    wide["direct_delta_same_cells"] = (wide["direct_Extreme Altitude"] + wide["direct_No Altitude"]) / 2
    wide["w"] = np.minimum(wide["n_No Altitude"], wide["n_Extreme Altitude"])
    wide = wide.reset_index()

    def wavg(g, col):
        return np.average(g[col], weights=g["w"])

    rows = []
    for pt, g in wide.groupby("AutoPitchType", observed=True):
        rows.append({"AutoPitchType": pt, "n_pitcher_cells": len(g),
                     "observed_delta_stuff": wavg(g, "observed_delta"),
                     "direct_delta_same_cells": wavg(g, "direct_delta_same_cells")})
    res = pd.DataFrame(rows)
    res["observed_minus_direct"] = res["observed_delta_stuff"] - res["direct_delta_same_cells"]
    return res.sort_values("n_pitcher_cells", ascending=False)


def run_entry(entry, saved_scores):
    label = entry["label"]
    frame = entry["future_frame"].copy()
    scored = {lvl: score_at_level(entry, frame, lvl) for lvl in LEVELS}

    alt = frame["altitude_category"].astype(str).to_numpy()
    stuff_actual = np.select([alt == lvl for lvl in LEVELS], [scored[l]["stuff"] for l in LEVELS])
    ref = saved_scores.loc[saved_scores["window"] == label, "stuff_plus_score"].to_numpy()
    max_diff = float(np.max(np.abs(stuff_actual - ref)))
    if max_diff > 1e-6:
        raise ValueError(f"{label}: los modelos del bundle no reproducen 09 (max diff {max_diff}).")

    df = frame.copy()
    df["window"] = label
    df["stuff_actual"] = stuff_actual
    df["stuff_no_alt"] = scored["No Altitude"]["stuff"]
    df["stuff_extreme"] = scored["Extreme Altitude"]["stuff"]
    df["delta_stuff_direct"] = df["stuff_extreme"] - df["stuff_no_alt"]
    df["delta_stuff_medium_vs_no"] = scored["Medium Altitude"]["stuff"] - df["stuff_no_alt"]
    df["delta_no_chase"] = scored["Extreme Altitude"]["stuff_no_chase"] - scored["No Altitude"]["stuff_no_chase"]
    for c in COMPONENTS:
        df[f"d_{COMPONENT_LABELS[c]}"] = (
            scored["Extreme Altitude"]["probs"][c] - scored["No Altitude"]["probs"][c])
    df = add_shape_bins(df)
    return df, max_diff


def main():
    bundle_path = BASE_DIR / "09_pitching_plus_bundle.pkl"
    if not bundle_path.exists():
        raise FileNotFoundError("Falta 09_pitching_plus_bundle.pkl: correr 09_pitching_plus.py primero.")
    bundle = joblib.load(bundle_path)
    saved_scores = pd.read_csv(BASE_DIR / "09_pitching_plus_scores.csv", usecols=["window", "stuff_plus_score"])

    frames, observed_all = [], []
    pd.set_option("display.width", 200)
    for label, entry in bundle.items():
        df, max_diff = run_entry(entry, saved_scores)
        frames.append(df)
        print(f"\n{'=' * 78}\nCONTRAFACTUAL {label} | cohorte n={len(df):,} | "
              f"identidad vs 09: max|diff|={max_diff:.2e} (OK)\n{'=' * 78}")
        print(f"delta_direct = Stuff+(Extreme) - Stuff+(No Altitude); Extreme = {EXTREME_NOTE}")
        by_type = summarize(df, ["AutoPitchType"]).sort_values("mean", ascending=False)
        print("\nPor AutoPitchType (delta en puntos Stuff+; componentes en puntos porcentuales de prob):")
        show = by_type.copy()
        for c in COMPONENTS:
            col = f"d_{COMPONENT_LABELS[c]}"
            show[col] = show[col] * 100
        print(show.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

        grad = df.groupby("AutoPitchType", observed=True)[
            ["delta_stuff_medium_vs_no", "delta_stuff_direct"]].mean()
        print("\nGradiente (Medium-No, Extreme-No), media por tipo:")
        print(grad.to_string(float_format=lambda x: f"{x:.2f}"))

        zone = df.assign(zona=np.where(df["outside_strike_zone"] == 1, "fuera de zona", "dentro/otra")
                         ).groupby("zona")[["delta_stuff_direct", "delta_no_chase"]].mean()
        print("\nDominio de chase: delta medio por zona (delta_direct vs version sin chase):")
        print(zone.to_string(float_format=lambda x: f"{x:.2f}"))

        obs = observed_contrast(df)
        obs.insert(0, "window", label)
        observed_all.append(obs)
        print("\nContraste OBSERVACIONAL within-pitcher/pitch-type (Extreme - No), vs efecto directo "
              "en las mismas celdas:")
        print(obs.drop(columns="window").to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    all_rows = pd.concat(frames, ignore_index=True)
    keep = ["window", "PitchUID", "pitcher_anon_id", "AutoPitchType", "altitude_category", "shape_bin",
            "stuff_no_alt", "stuff_extreme", "delta_stuff_direct", "delta_stuff_medium_vs_no",
            "delta_no_chase"] + [f"d_{COMPONENT_LABELS[c]}" for c in COMPONENTS]
    all_rows[keep].to_csv(BASE_DIR / "10_altitude_counterfactual_rows.csv", index=False)
    summarize(all_rows, ["window", "AutoPitchType"]).to_csv(
        BASE_DIR / "10_altitude_delta_by_pitch_type.csv", index=False)
    shape = summarize(all_rows, ["window", "AutoPitchType", "shape_bin"])
    shape.to_csv(BASE_DIR / "10_altitude_delta_by_shape_bin.csv", index=False)
    pitcher = summarize(all_rows, ["window", "pitcher_anon_id"])
    pitcher = pitcher[pitcher["n"] >= MIN_PITCHER_PITCHES]
    pitcher.to_csv(BASE_DIR / "10_altitude_delta_by_pitcher.csv", index=False)
    pd.concat(observed_all).to_csv(BASE_DIR / "10_altitude_observed_contrast.csv", index=False)

    primary = all_rows[all_rows["window"] == "2024-2025 -> 2026"]
    print(f"\n{'=' * 78}\nVENTANA PRIMARIA 2024-2025 -> 2026: shape bins con mayor ganancia/perdida (n>=200)\n{'=' * 78}")
    sb = summarize(primary, ["shape_bin"]).query("n >= 200").sort_values("mean")
    cols = ["shape_bin", "n", "mean", "sd", "p10", "p90"]
    print("Mayores perdidas:\n" + sb[cols].head(5).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("Mayores ganancias:\n" + sb[cols].tail(5).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    pp = pitcher[pitcher["window"] == "2024-2025 -> 2026"]
    print(f"\nPitchers (n>={MIN_PITCHER_PITCHES}): {len(pp)} | delta_direct medio {pp['mean'].mean():.2f} "
          f"| SD entre pitchers {pp['mean'].std():.2f}")
    print("\nPendiente: bootstrap por pitcher (intervalos) y delta_total con movimiento por ambiente.")
    print("Archivos: 10_altitude_counterfactual_rows.csv, 10_altitude_delta_by_{pitch_type,shape_bin,pitcher}.csv, "
          "10_altitude_observed_contrast.csv")


if __name__ == "__main__":
    main()
