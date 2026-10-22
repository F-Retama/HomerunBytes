import pandas as pd

df = pd.read_parquet("stuff_model_df.parquet")
diccionario = pd.read_csv("stuff_plus_data_dictionary.csv")

print("=" * 60)
print("DIMENSIONES")
print("=" * 60)
print(f"Filas: {df.shape[0]:,}")
print(f"Columnas: {df.shape[1]}")

print("\n" + "=" * 60)
print("TEMPORADAS")
print("=" * 60)
print(df["year"].value_counts().sort_index())

print("\n" + "=" * 60)
print("PITCHERS / JUEGOS")
print("=" * 60)
print("Pitchers:", df["pitcher_anon_id"].nunique())
print("Batters:", df["batter_anon_id"].nunique())
print("Juegos:", df["game_anon_id"].nunique())

print("\n" + "=" * 60)
print("TIPOS DE PITCHEO")
print("=" * 60)
print(df["AutoPitchType"].value_counts(dropna=False))

print("\n" + "=" * 60)
print("ALTITUD")
print("=" * 60)
print(df["altitude_category"].value_counts(dropna=False))

print("\n" + "=" * 60)
print("PITCH TYPE × ALTITUDE")
print("=" * 60)
print(
    pd.crosstab(
        df["AutoPitchType"],
        df["altitude_category"],
        margins=True
    )
)

print("\n" + "=" * 60)
print("MISSINGS MÁS IMPORTANTES")
print("=" * 60)

missing = (
    df.isna()
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

print(missing.head(30).round(2))

print("\n" + "=" * 60)
print("VARIABLES FÍSICAS")
print("=" * 60)

physical_cols = [
    "RelSpeed",
    "EffectiveVelo",
    "ZoneSpeed",
    "SpinRate",
    "SpinAxis",
    "RelHeight",
    "RelSide",
    "Extension",
    "VertBreak",
    "InducedVertBreak",
    "HorzBreak",
    "VertRelAngle",
    "HorzRelAngle",
    "VertApprAngle",
    "HorzApprAngle",
    "SpeedDrop",
    "ZoneTime"
]

print(df[physical_cols].describe().T)

print("\n" + "=" * 60)
print("OUTCOMES")
print("=" * 60)

target_cols = [
    "is_swing",
    "is_whiff",
    "is_contact",
    "is_called_strike",
    "is_swinging_strike",
    "is_ball_in_play",
    "is_batted",
    "is_hit",
    "home_run",
    "base_on_balls",
    "strikeout",
]

for col in target_cols:
    print(f"{col:25s}: {df[col].mean():.4f}")

print("\n" + "=" * 60)
print("ROLES DEL DICCIONARIO")
print("=" * 60)

print(diccionario["feature_role"].value_counts(dropna=False))

print("\nSTUFF FEATURES")
print(
    diccionario.loc[
        diccionario["feature_role"] == "stuff_feature",
        ["column_name", "use_as_model_feature"]
    ].to_string(index=False)
)

print("\nTARGET ONLY")
print(
    diccionario.loc[
        diccionario["feature_role"] == "target_only",
        "column_name"
    ].tolist()
)

summary_pitch_altitude = (
    df.groupby(
    ["AutoPitchType", "altitude_category"],
    observed=True
    ).agg(
        n=("PitchUID", "count"),
        pitchers=("pitcher_anon_id", "nunique"),
        velo=("RelSpeed", "mean"),
        spin=("SpinRate", "mean"),
        ivb=("InducedVertBreak", "mean"),
        hb=("HorzBreak", "mean"),
        whiff_rate=("is_whiff", "mean"),
        called_strike_rate=("is_called_strike", "mean"),
        hr_rate=("home_run", "mean"),
    )
    .reset_index()
)

print(summary_pitch_altitude.to_string(index=False))


print("\n" + "=" * 60)
print("PITCHERS POR NÚMERO DE CATEGORÍAS DE ALTITUD")
print("=" * 60)

pitcher_altitudes = (
    df.dropna(subset=["pitcher_anon_id", "altitude_category"])
    .groupby("pitcher_anon_id")["altitude_category"]
    .nunique()
)

print(pitcher_altitudes.value_counts().sort_index())

print("\nPitchers presentes en >= 2 altitudes:")
print((pitcher_altitudes >= 2).sum())

print("\nPitchers presentes en las 3 altitudes:")
print((pitcher_altitudes == 3).sum())


core_pitch_types = [
    "Four-Seam",
    "Slider",
    "Sinker",
    "Changeup",
    "Curveball",
    "Cutter",
    "Splitter",
]

df_core = df[
    df["AutoPitchType"].isin(core_pitch_types)
    & df["altitude_category"].isin(
        ["No Altitude", "Medium Altitude", "Extreme Altitude"]
    )
].copy()

pitcher_pitch_alt = (
    df_core
    .groupby([
        "pitcher_anon_id",
        "AutoPitchType",
        "altitude_category"
    ], observed=True)
    .agg(
        n=("PitchUID", "count"),
        velo=("RelSpeed", "mean"),
        spin=("SpinRate", "mean"),
        ivb=("InducedVertBreak", "mean"),
        hb=("HorzBreak", "mean"),
        whiff_rate=("is_whiff", "mean"),
        hr_rate=("home_run", "mean"),
    )
    .reset_index()
)

# Evitamos comparar pitcher-pitch con muestras ridículamente pequeñas
pitcher_pitch_alt = pitcher_pitch_alt[
    pitcher_pitch_alt["n"] >= 20
]

print(pitcher_pitch_alt.head(30))


print("\n" + "=" * 60)
print("DELTA WITHIN-PITCHER: EXTREME VS NO ALTITUDE")
print("=" * 60)

# Pivot: una fila por pitcher + pitch type
pivot = pitcher_pitch_alt.pivot(
    index=["pitcher_anon_id", "AutoPitchType"],
    columns="altitude_category",
    values=[
        "n",
        "velo",
        "spin",
        "ivb",
        "hb",
        "whiff_rate",
        "hr_rate"
    ]
)

# Aplanamos las columnas del pivot
pivot.columns = [
    f"{variable}_{altitude}"
    for variable, altitude in pivot.columns
]

# Volvemos pitcher y pitch type columnas normales
pivot = pivot.reset_index()

# Nos quedamos solo con casos que existen
# tanto en No Altitude como en Extreme Altitude
paired = pivot[
    pivot["n_No Altitude"].notna()
    & pivot["n_Extreme Altitude"].notna()
].copy()

print("Combinaciones pitcher-pitch comparables:", len(paired))

# Deltas = Extreme Altitude - No Altitude
paired["delta_velo"] = (
    paired["velo_Extreme Altitude"]
    - paired["velo_No Altitude"]
)

paired["delta_spin"] = (
    paired["spin_Extreme Altitude"]
    - paired["spin_No Altitude"]
)

paired["delta_ivb"] = (
    paired["ivb_Extreme Altitude"]
    - paired["ivb_No Altitude"]
)

paired["delta_hb"] = (
    paired["hb_Extreme Altitude"]
    - paired["hb_No Altitude"]
)

paired["delta_whiff"] = (
    paired["whiff_rate_Extreme Altitude"]
    - paired["whiff_rate_No Altitude"]
)

paired["delta_hr"] = (
    paired["hr_rate_Extreme Altitude"]
    - paired["hr_rate_No Altitude"]
)

print("\nDELTA PROMEDIO GLOBAL")

delta_cols = [
    "delta_velo",
    "delta_spin",
    "delta_ivb",
    "delta_hb",
    "delta_whiff",
    "delta_hr"
]

print(paired[delta_cols].mean())


delta_summary = (
    paired
    .groupby("AutoPitchType", observed=True)
    .agg(
        combinations=("pitcher_anon_id", "count"),
        delta_velo=("delta_velo", "mean"),
        delta_spin=("delta_spin", "mean"),
        delta_ivb=("delta_ivb", "mean"),
        delta_hb=("delta_hb", "mean"),
        delta_whiff=("delta_whiff", "mean"),
        delta_hr=("delta_hr", "mean"),
    )
    .sort_values("combinations", ascending=False)
)

print("\n" + "=" * 60)
print("DELTA EXTREME - NO ALTITUDE POR PITCH TYPE")
print("=" * 60)

print(delta_summary)


# ============================================================
# DELTA PONDERADO
# ============================================================

import numpy as np

# Peso = menor número de lanzamientos entre ambas altitudes
paired["weight"] = np.minimum(
    paired["n_Extreme Altitude"],
    paired["n_No Altitude"]
)

def weighted_mean(group, column):
    return np.average(
        group[column],
        weights=group["weight"]
    )

weighted_results = []

for pitch_type, group in paired.groupby(
    "AutoPitchType",
    observed=True
):

    weighted_results.append({
        "AutoPitchType": pitch_type,
        "n_pitcher_pitch_pairs": len(group),
        "delta_velo": weighted_mean(group, "delta_velo"),
        "delta_spin": weighted_mean(group, "delta_spin"),
        "delta_ivb": weighted_mean(group, "delta_ivb"),
        "delta_hb": weighted_mean(group, "delta_hb"),
        "delta_whiff": weighted_mean(group, "delta_whiff"),
        "delta_hr": weighted_mean(group, "delta_hr"),
    })

weighted_summary = pd.DataFrame(weighted_results)

print("\n" + "=" * 60)
print("DELTA PONDERADO EXTREME - NO ALTITUDE")
print("=" * 60)

print(
    weighted_summary
    .sort_values("n_pitcher_pitch_pairs", ascending=False)
    .to_string(index=False)
)


print("\n" + "=" * 60)
print("WITHIN-PITCHER + PITCH TYPE + YEAR")
print("=" * 60)

pitcher_pitch_year_alt = (
    df_core
    .groupby(
        [
            "pitcher_anon_id",
            "AutoPitchType",
            "year",
            "altitude_category"
        ],
        observed=True
    )
    .agg(
        n=("PitchUID", "count"),
        velo=("RelSpeed", "mean"),
        spin=("SpinRate", "mean"),
        ivb=("InducedVertBreak", "mean"),
        hb=("HorzBreak", "mean"),
        whiff_rate=("is_whiff", "mean"),
        hr_rate=("home_run", "mean"),
    )
    .reset_index()
)

# mínimo 20 pitches por ambiente
pitcher_pitch_year_alt = pitcher_pitch_year_alt[
    pitcher_pitch_year_alt["n"] >= 20
]

pivot_year = pitcher_pitch_year_alt.pivot(
    index=[
        "pitcher_anon_id",
        "AutoPitchType",
        "year"
    ],
    columns="altitude_category",
    values=[
        "n",
        "velo",
        "spin",
        "ivb",
        "hb",
        "whiff_rate",
        "hr_rate"
    ]
)

# Aplanamos columnas
pivot_year.columns = [
    f"{variable}_{altitude}"
    for variable, altitude in pivot_year.columns
]

# Convertimos índices en columnas normales
pivot_year = pivot_year.reset_index()

# Solo casos con datos en ambas altitudes
paired_year = pivot_year[
    pivot_year["n_No Altitude"].notna()
    & pivot_year["n_Extreme Altitude"].notna()
].copy()

print(
    "Combinaciones pitcher-pitch-year comparables:",
    len(paired_year)
)

# Deltas = Extreme - No Altitude
paired_year["delta_velo"] = (
    paired_year["velo_Extreme Altitude"]
    - paired_year["velo_No Altitude"]
)

paired_year["delta_spin"] = (
    paired_year["spin_Extreme Altitude"]
    - paired_year["spin_No Altitude"]
)

paired_year["delta_ivb"] = (
    paired_year["ivb_Extreme Altitude"]
    - paired_year["ivb_No Altitude"]
)

paired_year["delta_hb"] = (
    paired_year["hb_Extreme Altitude"]
    - paired_year["hb_No Altitude"]
)

paired_year["delta_whiff"] = (
    paired_year["whiff_rate_Extreme Altitude"]
    - paired_year["whiff_rate_No Altitude"]
)

paired_year["delta_hr"] = (
    paired_year["hr_rate_Extreme Altitude"]
    - paired_year["hr_rate_No Altitude"]
)

# Peso = menor muestra entre ambas altitudes
paired_year["weight"] = np.minimum(
    paired_year["n_Extreme Altitude"],
    paired_year["n_No Altitude"]
)

weighted_year_results = []

for pitch_type, group in paired_year.groupby(
    "AutoPitchType",
    observed=True
):

    weighted_year_results.append({
        "AutoPitchType": pitch_type,
        "n_pairs": len(group),
        "delta_velo": np.average(
            group["delta_velo"],
            weights=group["weight"]
        ),
        "delta_spin": np.average(
            group["delta_spin"],
            weights=group["weight"]
        ),
        "delta_ivb": np.average(
            group["delta_ivb"],
            weights=group["weight"]
        ),
        "delta_hb": np.average(
            group["delta_hb"],
            weights=group["weight"]
        ),
        "delta_whiff": np.average(
            group["delta_whiff"],
            weights=group["weight"]
        ),
        "delta_hr": np.average(
            group["delta_hr"],
            weights=group["weight"]
        ),
    })

weighted_year_summary = pd.DataFrame(weighted_year_results)

print("\n" + "=" * 60)
print("DELTA PONDERADO CONTROLANDO POR TEMPORADA")
print("=" * 60)

print(
    weighted_year_summary
    .sort_values("n_pairs", ascending=False)
    .to_string(index=False)
)