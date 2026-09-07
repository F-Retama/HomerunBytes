import pandas as pd
import numpy as np

# ============================================================
# 1. CARGA
# ============================================================

df = pd.read_parquet("stuff_model_df.parquet")

print("=" * 60)
print("DATASET ORIGINAL")
print("=" * 60)
print(df.shape)

CORE_PITCH_TYPES = [
    "Four-Seam",
    "Slider",
    "Sinker",
    "Changeup",
    "Curveball",
    "Cutter",
    "Splitter",
]

FASTBALL_TYPES = [
    "Four-Seam",
    "Sinker",
]

df_model = df[
    df["AutoPitchType"].isin(CORE_PITCH_TYPES)
].copy()

print("\n" + "=" * 60)
print("DATASET V1")
print("=" * 60)
print(df_model.shape)

print(df_model["AutoPitchType"].value_counts())

# ============================================================
# 2. PRIMARY FASTBALL POR PITCHER + YEAR
# ============================================================

fb_usage = (
    df_model[
        df_model["AutoPitchType"].isin(FASTBALL_TYPES)
    ]
    .groupby(
        ["pitcher_anon_id", "year", "AutoPitchType"],
        observed=True
    )
    .size()
    .reset_index(name="n_pitches")
)

print("\nFASTBALL USAGE:")
print(fb_usage.head(20))

primary_fb = (
    fb_usage
    .sort_values(
        ["pitcher_anon_id", "year", "n_pitches"],
        ascending=[True, True, False]
    )
    .drop_duplicates(
        ["pitcher_anon_id", "year"]
    )
    .rename(
        columns={
            "AutoPitchType": "primary_fastball_type",
            "n_pitches": "primary_fastball_n"
        }
    )
)


print("\n" + "=" * 60)
print("PRIMARY FASTBALL")
print("=" * 60)

print(primary_fb.head(20))

print("\nDistribución:")
print(
    primary_fb["primary_fastball_type"]
    .value_counts()
)

# ============================================================
# FILTRO DE ESTABILIDAD DE PRIMARY FASTBALL
# ============================================================

MIN_PRIMARY_FB_PITCHES = 20

primary_fb = primary_fb[
    primary_fb["primary_fastball_n"] >= MIN_PRIMARY_FB_PITCHES
].copy()

print("\n" + "=" * 60)
print("PRIMARY FASTBALL DESPUÉS DEL FILTRO")
print("=" * 60)

print("Mínimo de pitches:", MIN_PRIMARY_FB_PITCHES)
print("Pitcher-years válidos:", len(primary_fb))

print(
    primary_fb["primary_fastball_type"]
    .value_counts()
)

# ============================================================
# 3. PERFIL FÍSICO DE LA PRIMARY FASTBALL
# ============================================================

df_fb = df_model.merge(
    primary_fb[
        [
            "pitcher_anon_id",
            "year",
            "primary_fastball_type"
        ]
    ],
    on=["pitcher_anon_id", "year"],
    how="left"
)

primary_fb_pitches = df_fb[
    df_fb["AutoPitchType"]
    == df_fb["primary_fastball_type"]
].copy()

fb_profile = (
    primary_fb_pitches
    .groupby(
        ["pitcher_anon_id", "year"],
        observed=True
    )
    .agg(
        fb_rel_speed=("RelSpeed", "mean"),
        fb_spin_rate=("SpinRate", "mean"),
        fb_ivb=("InducedVertBreak", "mean"),
        fb_hb=("HorzBreak", "mean"),
        fb_rel_height=("RelHeight", "mean"),
        fb_rel_side=("RelSide", "mean"),
        fb_extension=("Extension", "mean"),
    )
    .reset_index()
)

print("\n" + "=" * 60)
print("FASTBALL PROFILE")
print("=" * 60)

print(fb_profile.head(20))

df_features = (
    df_fb
    .merge(
        fb_profile,
        on=["pitcher_anon_id", "year"],
        how="left"
    )
)

print("\n" + "=" * 60)
print("MERGE DEL PERFIL FB")
print("=" * 60)

print(df_features.shape)


# ============================================================
# 5. FEATURES RELACIONALES VS FASTBALL
# ============================================================

df_features["velo_diff_vs_fb"] = (
    df_features["RelSpeed"]
    - df_features["fb_rel_speed"]
)

df_features["ivb_diff_vs_fb"] = (
    df_features["InducedVertBreak"]
    - df_features["fb_ivb"]
)

df_features["hb_diff_vs_fb"] = (
    df_features["HorzBreak"]
    - df_features["fb_hb"]
)

df_features["spin_diff_vs_fb"] = (
    df_features["SpinRate"]
    - df_features["fb_spin_rate"]
)

# ============================================================
# 6. PITCH USAGE
# ============================================================

pitch_usage = (
    df_features
    .groupby(
        [
            "pitcher_anon_id",
            "year",
            "AutoPitchType"
        ],
        observed=True
    )
    .size()
    .reset_index(name="pitch_type_count")
)

pitcher_total = (
    df_features
    .groupby(
        ["pitcher_anon_id", "year"],
        observed=True
    )
    .size()
    .reset_index(name="pitcher_total_pitches")
)

pitch_usage = pitch_usage.merge(
    pitcher_total,
    on=["pitcher_anon_id", "year"],
    how="left"
)

pitch_usage["pitch_usage_pct"] = (
    pitch_usage["pitch_type_count"]
    / pitch_usage["pitcher_total_pitches"]
)

df_features = df_features.merge(
    pitch_usage[
        [
            "pitcher_anon_id",
            "year",
            "AutoPitchType",
            "pitch_usage_pct"
        ]
    ],
    on=[
        "pitcher_anon_id",
        "year",
        "AutoPitchType"
    ],
    how="left"
)

# ============================================================
# 7. RELEASE SIMILARITY VS FASTBALL
# ============================================================

df_features["release_distance_vs_fb"] = np.sqrt(
    (
        df_features["RelHeight"]
        - df_features["fb_rel_height"]
    ) ** 2
    +
    (
        df_features["RelSide"]
        - df_features["fb_rel_side"]
    ) ** 2
)

df_features["extension_diff_vs_fb"] = (
    df_features["Extension"]
    - df_features["fb_extension"]
)

# ============================================================
# DISTRIBUCIÓN DE FEATURES RELACIONALES
# ============================================================

arsenal_features = [
    "velo_diff_vs_fb",
    "ivb_diff_vs_fb",
    "hb_diff_vs_fb",
    "spin_diff_vs_fb",
    "pitch_usage_pct",
    "release_distance_vs_fb",
    "extension_diff_vs_fb",
]

print("\n" + "=" * 60)
print("DISTRIBUCIÓN FEATURES DE ARSENAL")
print("=" * 60)

print(
    df_features[arsenal_features]
    .describe(
        percentiles=[
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99
        ]
    )
    .T
)

print("\n" + "=" * 60)
print("FEATURES RELACIONALES POR PITCH TYPE")
print("=" * 60)

arsenal_by_pitch = (
    df_features
    .groupby("AutoPitchType", observed=True)
    .agg(
        n=("PitchUID", "count"),

        velo_diff_mean=("velo_diff_vs_fb", "mean"),
        velo_diff_std=("velo_diff_vs_fb", "std"),

        ivb_diff_mean=("ivb_diff_vs_fb", "mean"),
        ivb_diff_std=("ivb_diff_vs_fb", "std"),

        hb_diff_mean=("hb_diff_vs_fb", "mean"),
        hb_diff_std=("hb_diff_vs_fb", "std"),

        release_dist_mean=("release_distance_vs_fb", "mean"),
        usage_mean=("pitch_usage_pct", "mean"),
    )
    .reset_index()
)

print(
    arsenal_by_pitch
    .to_string(index=False)
)

# ============================================================
# 8. COBERTURA DE FEATURES DERIVADAS
# ============================================================

derived_features = [
    "primary_fastball_type",
    "velo_diff_vs_fb",
    "ivb_diff_vs_fb",
    "hb_diff_vs_fb",
    "spin_diff_vs_fb",
    "pitch_usage_pct",
    "release_distance_vs_fb",
    "extension_diff_vs_fb",
]

print("\n" + "=" * 60)
print("MISSINGNESS FEATURES DERIVADAS")
print("=" * 60)

print(
    (
        df_features[derived_features]
        .isna()
        .mean()
        * 100
    )
    .round(2)
)

print("\nPitchers sin fastball primaria identificada:")

no_fb = (
    df_features[
        df_features["primary_fastball_type"].isna()
    ]["pitcher_anon_id"]
    .nunique()
)

print(no_fb)

print("\n" + "=" * 60)
print("EJEMPLOS DE ARSENAL")
print("=" * 60)

arsenal_summary = (
    df_features
    .groupby(
        [
            "pitcher_anon_id",
            "year",
            "primary_fastball_type",
            "AutoPitchType"
        ],
        observed=True
    )
    .agg(
        n=("PitchUID", "count"),
        usage=("pitch_usage_pct", "mean"),
        velo=("RelSpeed", "mean"),
        velo_diff_vs_fb=("velo_diff_vs_fb", "mean"),
        ivb=("InducedVertBreak", "mean"),
        ivb_diff_vs_fb=("ivb_diff_vs_fb", "mean"),
        hb=("HorzBreak", "mean"),
        hb_diff_vs_fb=("hb_diff_vs_fb", "mean"),
        release_dist=("release_distance_vs_fb", "mean"),
    )
    .reset_index()
)

print(
    arsenal_summary
    .head(50)
    .to_string(index=False)
)

# ============================================================
# 9. AUDITORÍA DE ESTABILIDAD DE PRIMARY FASTBALL
# ============================================================

print("\n" + "=" * 60)
print("MUESTRA DE PRIMARY FASTBALL POR PITCHER-YEAR")
print("=" * 60)

print(
    primary_fb["primary_fastball_n"]
    .describe(
        percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90]
    )
)

thresholds = [1, 5, 10, 20, 30, 50, 100]

print("\nPitcher-years por debajo de cada umbral:")

for threshold in thresholds:
    n = (primary_fb["primary_fastball_n"] < threshold).sum()
    pct = 100 * n / len(primary_fb)

    print(
        f"< {threshold:3d} pitches: "
        f"{n:4d} pitcher-years ({pct:.1f}%)"
    )

threshold_coverage = []

for threshold in [10, 20, 30, 50]:

    valid_fb = primary_fb[
        primary_fb["primary_fastball_n"] >= threshold
    ][
        ["pitcher_anon_id", "year"]
    ]

    temp = df_model.merge(
        valid_fb,
        on=["pitcher_anon_id", "year"],
        how="inner"
    )

    threshold_coverage.append({
        "min_fb_pitches": threshold,
        "pitcher_years": len(valid_fb),
        "pitches_retained": len(temp),
        "pct_dataset_retained": 100 * len(temp) / len(df_model)
    })

threshold_coverage = pd.DataFrame(threshold_coverage)

print("\n" + "=" * 60)
print("COBERTURA SEGÚN MÍNIMO DE FASTBALLS")
print("=" * 60)

print(threshold_coverage.to_string(index=False))

# ============================================================
# 10. GUARDAR DATASET V1
# ============================================================

df_features.to_parquet(
    "stuff_model_features_v1.parquet",
    index=False
)

print("\n" + "=" * 60)
print("ARCHIVO GUARDADO")
print("=" * 60)

print("stuff_model_features_v1.parquet")
print(df_features.shape)