"""Referencias de arsenal aprendidas solo en train; auditoría sin modelos.

Convenciones heredadas de 02:
- Conteos de usage: pitch_type_count y pitcher_total_pitches (no se renombran).
- Desempate de FB: mayor conteo, mayor RelSpeed media válida del periodo
  permitido y, finalmente, Four-Seam. No se imputan velocidades faltantes.
- Los grupos con llaves nulas se excluyen del fit, como en 02.

03 no tiene helpers importables: importarlo ejecutaría el entrenamiento.
Se replica únicamente su carga, población y configuración de holdout.
Las columnas de arsenal del parquet se descartan antes de los merges.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_dtype_equal
from sklearn.model_selection import GroupShuffleSplit


MIN_PRIMARY_FB_PITCHES = 20
FASTBALL_TYPES = ["Four-Seam", "Sinker"]
TARGET = "is_whiff"
FB_KEYS = ["pitcher_anon_id", "year"]
USAGE_KEYS = FB_KEYS + ["AutoPitchType"]
FB_COLUMNS = {
    "fb_rel_speed": "RelSpeed",
    "fb_spin_rate": "SpinRate",
    "fb_ivb": "InducedVertBreak",
    "fb_hb": "HorzBreak",
    "fb_rel_height": "RelHeight",
    "fb_rel_side": "RelSide",
    "fb_extension": "Extension",
}
arsenal_features = [
    "velo_diff_vs_fb", "ivb_diff_vs_fb", "hb_diff_vs_fb",
    "spin_diff_vs_fb", "pitch_usage_pct", "release_distance_vs_fb",
    "extension_diff_vs_fb",
]
REFERENCE_COLUMNS = list(FB_COLUMNS) + [
    "primary_fastball_type", "primary_fastball_n",
    "pitch_type_count", "pitcher_total_pitches",
    "pitch_type_n", "total_pitches",
    "historical_pitch_type_n", "historical_total_pitches",
    "historical_pitch_usage_pct", "future_pitch_usage_pct",
] + arsenal_features


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_keys(frame, keys, name):
    require(not frame[keys].isna().any().any(), f"{name}: llaves nulas.")
    require(not frame.duplicated(keys).any(), f"{name}: llaves duplicadas.")


def check_merge_dtypes(left, right, keys):
    for key in keys:
        require(is_dtype_equal(left[key].dtype, right[key].dtype),
                f"Dtype incompatible en {key}: {left[key].dtype} vs {right[key].dtype}.")
        if isinstance(left[key].dtype, pd.CategoricalDtype):
            require(left[key].cat.categories.equals(right[key].cat.categories),
                    f"Orden de categorías incompatible en {key}.")


def fastball_counts(frame, group_keys=None):
    group_keys = FB_KEYS if group_keys is None else group_keys
    return (
        frame[frame["AutoPitchType"].isin(FASTBALL_TYPES)]
        .groupby(group_keys + ["AutoPitchType"], observed=True)
        .size().reset_index(name="n_pitches")
    )


def fastball_summary(frame, group_keys):
    """Resumen solo del periodo permitido; conteo de filas como en 02/04.

    Se conserva size en lugar de count(PitchUID): un ID faltante no elimina
    un lanzamiento. Ordenar velocidades antes de promediar hace reproducible
    la reducción flotante incluso al reordenar las filas de entrada.
    """
    candidates = frame.loc[frame["AutoPitchType"].isin(FASTBALL_TYPES),
                           group_keys + ["AutoPitchType", "RelSpeed"]].copy()
    candidates["RelSpeed"] = candidates["RelSpeed"].astype(float)
    candidates = candidates.sort_values(group_keys + ["AutoPitchType", "RelSpeed"], na_position="last")
    summary = candidates.groupby(group_keys + ["AutoPitchType"], observed=True).agg(
        n_pitches=("RelSpeed", "size"), mean_rel_speed=("RelSpeed", "mean")
    ).reset_index()
    summary["type_priority"] = summary["AutoPitchType"].map(
        {"Four-Seam": 1, "Sinker": 0}
    ).astype(int)
    return summary


def select_primary_fastball(frame, group_keys):
    """Primary fastball selection:
    1. highest pitch count among Four-Seam and Sinker
    2. if tied, highest mean RelSpeed within the allowed reference period
    3. if still tied, Four-Seam as deterministic final priority
    4. minimum 20 pitches required

    Valid mean speed beats NaN; two NaNs use Four-Seam. No imputation.
    Historical callers pass only the permitted prior seasons: RelSpeed for
    tie-breaking is also learned exclusively from that history.
    """
    selected = (fastball_summary(frame, group_keys)
                .sort_values(group_keys + ["n_pitches", "mean_rel_speed", "type_priority"],
                             ascending=[True] * len(group_keys) + [False, False, False],
                             na_position="last")
                .drop_duplicates(group_keys))
    selected = selected[selected["n_pitches"] >= MIN_PRIMARY_FB_PITCHES]
    return selected[group_keys + ["AutoPitchType", "n_pitches"]].rename(columns={
        "AutoPitchType": "primary_fastball_type", "n_pitches": "primary_fastball_n"
    })


def check_primary_determinism(frame, group_keys):
    columns = group_keys + ["primary_fastball_type", "primary_fastball_n"]
    expected = select_primary_fastball(frame, group_keys)[columns].sort_values(group_keys).reset_index(drop=True)
    for reordered in (frame.iloc[::-1], frame.sample(frac=1, random_state=42)):
        actual = select_primary_fastball(reordered, group_keys)[columns].sort_values(group_keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(expected, actual, check_exact=True)


def _fit_reference(train_df, group_keys):
    """Aprende perfil y usage únicamente de las filas recibidas de train."""
    usage_keys = group_keys + ["AutoPitchType"]
    primary_fb = select_primary_fastball(train_df, group_keys)
    raw = train_df.drop(columns=REFERENCE_COLUMNS, errors="ignore")
    selected = raw.merge(primary_fb, on=group_keys, how="inner", validate="many_to_one")
    selected = selected[selected["AutoPitchType"] == selected["primary_fastball_type"]]
    means = (
        selected.groupby(group_keys, observed=True)
        .agg(**{name: (source, "mean") for name, source in FB_COLUMNS.items()})
        .reset_index()
    )
    fb_profile = primary_fb.merge(means, on=group_keys, validate="one_to_one")
    usage = train_df.groupby(usage_keys, observed=True).size().reset_index(name="pitch_type_count")
    totals = train_df.groupby(group_keys, observed=True).size().reset_index(name="pitcher_total_pitches")
    usage = usage.merge(totals, on=group_keys, validate="many_to_one")
    usage["pitch_usage_pct"] = usage["pitch_type_count"] / usage["pitcher_total_pitches"]
    check_keys(fb_profile, group_keys, "fb_profile")
    check_keys(usage, usage_keys, "usage")
    require(fb_profile["primary_fastball_n"].ge(MIN_PRIMARY_FB_PITCHES).all(),
            "Fastball primaria con menos de 20 pitches.")
    require(fb_profile["primary_fastball_type"].isin(FASTBALL_TYPES).all(),
            "Tipo de fastball primaria inválido.")
    require(usage["pitch_usage_pct"].between(0, 1).all(), "Usage fuera de [0, 1].")
    return fb_profile, usage


def _apply_reference(df, fb_profile, usage, group_keys):
    """Aplica referencias sin aprender estadísticas ni completar faltantes."""
    usage_keys = group_keys + ["AutoPitchType"]
    check_keys(fb_profile, group_keys, "fb_profile")
    check_keys(usage, usage_keys, "usage")
    check_merge_dtypes(df, fb_profile, group_keys)
    check_merge_dtypes(df, usage, usage_keys)
    result = df.drop(columns=REFERENCE_COLUMNS, errors="ignore").copy()
    marker = "__arsenal_row_position__"
    require(marker not in result, f"Columna reservada ya presente: {marker}.")
    result[marker] = np.arange(len(result))
    result = result.merge(fb_profile, on=group_keys, how="left", sort=False, validate="many_to_one")
    result = result.merge(usage, on=usage_keys, how="left", sort=False, validate="many_to_one")
    require(len(result) == len(df), "Los merges cambiaron el número de filas.")
    require(np.array_equal(result.pop(marker).to_numpy(), np.arange(len(df))),
            "Los merges duplicaron, eliminaron o reordenaron pitches.")
    result.index = df.index.copy()
    for feature, raw, reference in [
        ("velo_diff_vs_fb", "RelSpeed", "fb_rel_speed"),
        ("ivb_diff_vs_fb", "InducedVertBreak", "fb_ivb"),
        ("hb_diff_vs_fb", "HorzBreak", "fb_hb"),
        ("spin_diff_vs_fb", "SpinRate", "fb_spin_rate"),
        ("extension_diff_vs_fb", "Extension", "fb_extension"),
    ]:
        result[feature] = result[raw] - result[reference]
    result["release_distance_vs_fb"] = np.sqrt(
        (result["RelHeight"] - result["fb_rel_height"]) ** 2
        + (result["RelSide"] - result["fb_rel_side"]) ** 2
    )
    present_usage = result["pitch_usage_pct"].dropna()
    require(present_usage.between(0, 1).all(), "Usage transformado fuera de [0, 1].")
    return result


def fit_arsenal_reference(train_df):
    """Referencia por pitcher-year con el desempate determinístico compartido."""
    return _fit_reference(train_df, FB_KEYS)


def apply_arsenal_reference(df, fb_profile, usage):
    return _apply_reference(df, fb_profile, usage, FB_KEYS)


def fit_historical_arsenal_reference(history_df):
    """Un perfil por pitcher; conteos y medias ponderados por pitches históricos.

    No promediar promedios anuales ni usar el año futuro para seleccionar FB.
    El llamador debe seleccionar explícitamente las temporadas permitidas.
    """
    require(not history_df.empty, "Historial vacío.")
    require(history_df["year"].notna().all(), "Año nulo en historial.")
    fb, usage = _fit_reference(history_df, ["pitcher_anon_id"])
    usage = usage.rename(columns={
        "pitch_type_count": "historical_pitch_type_n",
        "pitcher_total_pitches": "historical_total_pitches",
        "pitch_usage_pct": "historical_pitch_usage_pct",
    })
    # Procedencia temporal explícita para impedir aplicación al mismo año/pasado.
    for ref in (fb, usage):
        ref.attrs["history_years"] = sorted(int(y) for y in history_df["year"].unique())
    return fb, usage


def apply_historical_arsenal_reference(future_df, fb_profile_hist, usage_hist):
    require(not future_df.empty, "Evaluación temporal vacía.")
    require(future_df["year"].notna().all(), "Año nulo en evaluación.")
    for ref in (fb_profile_hist, usage_hist):
        years = ref.attrs.get("history_years", [])
        require(bool(years) and max(years) < future_df["year"].min(),
                "Leakage temporal: referencias sin procedencia o no anteriores a evaluación.")
    require(fb_profile_hist.attrs["history_years"] == usage_hist.attrs["history_years"],
            "FB y usage no pertenecen a la misma ventana histórica.")
    usage = usage_hist.rename(columns={
        "historical_pitch_type_n": "pitch_type_count",
        "historical_total_pitches": "pitcher_total_pitches",
        "historical_pitch_usage_pct": "pitch_usage_pct",
    })
    return _apply_reference(future_df, fb_profile_hist, usage, ["pitcher_anon_id"])


def audit_references(train_raw, fb_profile, usage):
    """Reconstruye conteos y medias para auditar las referencias de train."""
    train_pitchers = set(train_raw["pitcher_anon_id"])
    for name, ref in [("fb_profile", fb_profile), ("usage", usage)]:
        require(set(ref["pitcher_anon_id"]).issubset(train_pitchers),
                f"Leakage: {name} contiene pitchers ajenos a train.")
        train_keys = pd.MultiIndex.from_frame(train_raw[FB_KEYS].drop_duplicates())
        require(pd.MultiIndex.from_frame(ref[FB_KEYS]).isin(train_keys).all(),
                f"Leakage: {name} contiene pitcher-years ajenos a train.")
    # Auditar todos los grupos; excede la muestra mínima solicitada.
    total_counts = train_raw.groupby(FB_KEYS, observed=True).size()
    type_counts = train_raw.groupby(USAGE_KEYS, observed=True).size()
    type_index = pd.MultiIndex.from_frame(usage[USAGE_KEYS])
    total_index = pd.MultiIndex.from_frame(usage[FB_KEYS])
    np.testing.assert_array_equal(usage["pitch_type_count"], type_counts.reindex(type_index))
    np.testing.assert_array_equal(usage["pitcher_total_pitches"], total_counts.reindex(total_index))
    np.testing.assert_allclose(usage["pitch_usage_pct"],
                               type_counts.reindex(type_index).to_numpy()
                               / total_counts.reindex(total_index).to_numpy(), rtol=0, atol=0)
    groups = train_raw.groupby(FB_KEYS, observed=True)
    for row in fb_profile.itertuples(index=False):
        group = groups.get_group((row.pitcher_anon_id, row.year))
        counts = group.loc[group["AutoPitchType"].isin(FASTBALL_TYPES), "AutoPitchType"].value_counts()
        selected = group[group["AutoPitchType"] == row.primary_fastball_type]
        require(len(selected) == row.primary_fastball_n == counts.max(),
                "La primary fastball no coincide con el conteo máximo de train.")
        for name, source in FB_COLUMNS.items():
            np.testing.assert_allclose(getattr(row, name), selected[source].mean(),
                                       rtol=1e-6, atol=1e-6, equal_nan=True)


def get_fastball_ties(frame, group_keys=None):
    group_keys = FB_KEYS if group_keys is None else group_keys
    counts = fastball_counts(frame, group_keys)
    # Solo hay dos candidatos; ambos deben existir y tener igual conteo.
    stats = counts.groupby(group_keys, observed=True)["n_pitches"].agg(["size", "min", "max"])
    keys = stats[(stats["size"] == 2) & (stats["min"] == stats["max"])].reset_index()[group_keys]
    ties = counts.merge(keys, on=group_keys, validate="many_to_one")
    totals = frame.groupby(group_keys, observed=True).size().reset_index(name="total_pitches")
    ties = ties.merge(totals, on=group_keys, validate="many_to_one")
    ties["fastball_pct_total"] = 100 * ties["n_pitches"] / ties["total_pitches"]
    ties["fastball_pct_among_FB"] = 50.0
    return ties


def audit_ties(train_raw):
    fastball_ties = get_fastball_ties(train_raw)
    relevant = fastball_ties[fastball_ties["n_pitches"] >= MIN_PRIMARY_FB_PITCHES]
    summary = fastball_summary(train_raw, FB_KEYS)
    summary = summary.merge(relevant[FB_KEYS].drop_duplicates(), on=FB_KEYS, validate="many_to_one")
    selected = select_primary_fastball(train_raw, FB_KEYS).set_index(FB_KEYS)
    rows = []
    for keys, group in summary.groupby(FB_KEYS, observed=True):
        four = group[group["AutoPitchType"] == "Four-Seam"].iloc[0]
        sinker = group[group["AutoPitchType"] == "Sinker"].iloc[0]
        f, k = four["mean_rel_speed"], sinker["mean_rel_speed"]
        if pd.isna(f) and pd.isna(k) or pd.notna(f) and pd.notna(k) and f == k:
            reason = "four_seam_final_priority"
        elif pd.isna(f) or pd.isna(k):
            reason = "valid_mean_rel_speed_over_nan"
        else:
            reason = "higher_mean_rel_speed"
        rows.append({"pitcher_anon_id": keys[0], "year": keys[1],
                     "Four-Seam n": four["n_pitches"], "Sinker n": sinker["n_pitches"],
                     "Four-Seam mean RelSpeed": f, "Sinker mean RelSpeed": k,
                     "selected_primary_fastball": selected.loc[keys, "primary_fastball_type"],
                     "tie_break_reason": reason})
    print("\nFASTBALL TIE-BREAK AUDIT")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    return fastball_ties


def audit_historical_reference(history, fb, usage):
    """Verificación independiente de conteos/medias contra el historial permitido."""
    check_keys(fb, ["pitcher_anon_id"], "fb_profile_hist")
    check_keys(usage, ["pitcher_anon_id", "AutoPitchType"], "usage_hist")
    pitchers = set(history["pitcher_anon_id"])
    for ref in (fb, usage):
        require(set(ref["pitcher_anon_id"]).issubset(pitchers), "Referencia ajena al historial.")
    totals = history.groupby("pitcher_anon_id", observed=True).size()
    types = history.groupby(["pitcher_anon_id", "AutoPitchType"], observed=True).size()
    expected_n = types.reindex(pd.MultiIndex.from_frame(usage[["pitcher_anon_id", "AutoPitchType"]])).to_numpy()
    expected_total = usage["pitcher_anon_id"].map(totals).to_numpy()
    np.testing.assert_array_equal(usage["historical_pitch_type_n"], expected_n)
    np.testing.assert_array_equal(usage["historical_total_pitches"], expected_total)
    np.testing.assert_allclose(usage["historical_pitch_usage_pct"], expected_n / expected_total, rtol=0, atol=0)
    require(usage["historical_pitch_usage_pct"].between(0, 1).all(), "Usage histórico inválido.")
    groups = history.groupby("pitcher_anon_id", observed=True)
    for row in fb.itertuples(index=False):
        group = groups.get_group(row.pitcher_anon_id)
        candidates = group[group["AutoPitchType"].isin(FASTBALL_TYPES)]["AutoPitchType"].value_counts()
        selected = group[group["AutoPitchType"] == row.primary_fastball_type]
        require(row.primary_fastball_type in FASTBALL_TYPES and
                len(selected) == row.primary_fastball_n == candidates.max() and len(selected) >= 20,
                "Primary FB histórica inválida.")
        for name, source in FB_COLUMNS.items():
            np.testing.assert_allclose(getattr(row, name), selected[source].mean(), rtol=1e-6, atol=1e-6, equal_nan=True)


def usage_stability(future, usage_hist):
    """Resultado descriptivo aislado: nunca se devuelve al transformador."""
    counts = future.groupby(["pitcher_anon_id", "AutoPitchType"], observed=True).size().reset_index(name="n")
    totals = future.groupby("pitcher_anon_id", observed=True).size()
    counts["future_pitch_usage_pct"] = counts["n"] / counts["pitcher_anon_id"].map(totals)
    comparison = usage_hist.merge(counts, on=["pitcher_anon_id", "AutoPitchType"], validate="one_to_one")
    comparison["absolute_usage_change"] = (comparison["historical_pitch_usage_pct"] - comparison["future_pitch_usage_pct"]).abs()
    return comparison


def season_forward(df_model, years, evaluation_year):
    history = df_model[df_model["year"].isin(years)].copy()
    future = df_model[df_model["year"] == evaluation_year].copy()
    label = f"{'-'.join(map(str, years))} -> {evaluation_year}"
    require(set(history["year"].unique()) == set(years), "Ventana histórica incompleta.")
    require(set(future["year"].unique()) == {evaluation_year}, "Año de evaluación incorrecto.")
    require(history["year"].max() < future["year"].min(), "Leakage: solapamiento temporal.")
    require(history.index.intersection(future.index).empty, "Filas compartidas entre historial y futuro.")
    require(not set(history["PitchUID"]) & set(future["PitchUID"]), "PitchUID compartidos entre temporadas.")
    # Única llamada de fit histórico: recibe exclusivamente history.
    check_primary_determinism(history, ["pitcher_anon_id"])
    fb, usage = fit_historical_arsenal_reference(history)
    audit_historical_reference(history, fb, usage)
    future_fs = apply_historical_arsenal_reference(future, fb, usage)
    original = [c for c in future if c not in REFERENCE_COLUMNS]
    pd.testing.assert_frame_equal(future[original], future_fs[original])
    has_history = future["pitcher_anon_id"].isin(history["pitcher_anon_id"])
    require(future_fs.loc[~has_history, arsenal_features].isna().all().all(), "Fallback inesperado para pitcher nuevo.")
    historical_pairs = pd.MultiIndex.from_frame(usage[["pitcher_anon_id", "AutoPitchType"]])
    has_type = pd.MultiIndex.from_frame(future[["pitcher_anon_id", "AutoPitchType"]]).isin(historical_pairs)
    require(future_fs.loc[~has_type, "pitch_usage_pct"].isna().all(), "Fallback inesperado para pitch type nuevo.")
    total_pitchers = future["pitcher_anon_id"].nunique()
    known_pitchers = future.loc[has_history, "pitcher_anon_id"].nunique()
    print(f"\nSEASON-FORWARD {label}")
    print("Evaluation year:", evaluation_year)
    print("Total evaluation pitchers:", total_pitchers)
    print("Pitchers with prior history:", known_pitchers)
    print("Pitchers without prior history:", total_pitchers - known_pitchers)
    print(f"% pitchers with history: {100 * known_pitchers / total_pitchers:.2f}%")
    print("Evaluation rows:", len(future))
    print("Rows with FB reference:", future_fs["fb_rel_speed"].notna().sum())
    print(f"FB reference coverage: {future_fs['fb_rel_speed'].notna().mean():.6f}")
    print("ARSENAL FEATURE MISSINGNESS (all evaluation rows)")
    print((future_fs[arsenal_features].isna().mean() * 100).to_string(float_format=lambda x: f"{x:.2f}%"))
    # Universo futuro completo para cobertura; cohorte elegible separada explícitamente.
    eligible = future_fs.loc[has_history]
    print("Rows with prior history (eligible cohort):", len(eligible))
    print(f"FB reference coverage within eligible cohort: {eligible['fb_rel_speed'].notna().mean():.6f}")
    print("ARSENAL FEATURE MISSINGNESS (prior-history cohort only)")
    print((eligible[arsenal_features].isna().mean() * 100).to_string(float_format=lambda x: f"{x:.2f}%"))
    ties = get_fastball_ties(history, ["pitcher_anon_id"])
    print("Historical tied primary FB candidates >=20 (pitchers):",
          ties.loc[ties["n_pitches"].ge(20), "pitcher_anon_id"].nunique())
    print("Tie handling: count, valid mean RelSpeed, Four-Seam final priority.")
    checks = [
        f"{label}: exact history years {years}, evaluation year {evaluation_year}: PASS",
        f"{label}: max(history year) < min(evaluation year), no shared rows/PitchUID: PASS",
        f"{label}: FB/usage sources only history; unique keys; dtypes; counts/means/usage bounds: PASS",
        f"{label}: rows {len(future)} -> {len(future_fs)}, original order/values preserved: PASS",
        f"{label}: merge-created duplicate pitches 0; unknown pitchers/types remain NaN: PASS",
    ]
    # Solo después de construir features se calcula esta tabla descriptiva futura.
    return usage_stability(future, usage), checks


def primary_stability(df_model):
    # Análisis descriptivo anual separado, jamás aplicado a features futuras.
    check_primary_determinism(df_model, FB_KEYS)
    annual, _ = fit_arsenal_reference(df_model)
    years = sorted(int(y) for y in df_model["year"].dropna().unique())
    print("\nPRIMARY FASTBALL STABILITY")
    changed_tables = []
    for previous, future in zip(years, years[1:]):
        if future != previous + 1:
            continue
        a = annual[annual["year"] == previous].copy()
        b = annual[annual["year"] == future].copy()
        c = a.merge(b, on="pitcher_anon_id", suffixes=("_previous", "_future"), validate="one_to_one")
        same = c["primary_fastball_type_previous"] == c["primary_fastball_type_future"]
        print(f"{previous} -> {future} | comparable pitcher-years: {len(c)} | same: {same.sum()} | "
              f"changed: {(~same).sum()} | stable: {100 * same.mean():.2f}%")
        changes = c.loc[~same, ["pitcher_anon_id", "year_previous", "year_future",
                             "primary_fastball_type_previous", "primary_fastball_type_future",
                             "primary_fastball_n_previous", "primary_fastball_n_future"]].copy()
        changes.columns = ["pitcher_anon_id", "previous_year", "future_year", "previous_primary_fb",
                           "future_primary_fb", "previous_primary_fb_n", "future_primary_fb_n"]
        changed_tables.append(changes)
    result = pd.concat(changed_tables, ignore_index=True) if changed_tables else pd.DataFrame()
    print("Changed cases (first 10; all cases saved in 04_primary_fastball_changes.csv):")
    print(result.head(10).to_string(index=False))
    print("Comparable = valid >=20-pitch primary FB in both consecutive years; deterministic ties included.")
    return result


def main():
    # Misma población, orden y holdout de 03; no importar 03 porque entrena.
    df = pd.read_parquet(Path(__file__).resolve().parent / "stuff_model_features_v1.parquet")
    df_model = df.dropna(subset=[TARGET, "pitcher_anon_id"]).copy()
    features = [
        "RelSpeed", "SpinRate", "SpinAxis", "RelHeight", "RelSide", "Extension",
        "InducedVertBreak", "HorzBreak", "VertRelAngle", "HorzRelAngle",
        "VertApprAngle", "HorzApprAngle", "AutoPitchType", "PitcherThrows",
    ]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(
        df_model[features], df_model[TARGET], groups=df_model["pitcher_anon_id"]
    ))
    train_raw = df_model.iloc[train_idx].copy()
    test_raw = df_model.iloc[test_idx].copy()
    train_pitchers = set(train_raw["pitcher_anon_id"])
    test_pitchers = set(test_raw["pitcher_anon_id"])
    require(not train_pitchers & test_pitchers, "Leakage: pitchers compartidos en el holdout.")
    audit_ties(train_raw)
    check_primary_determinism(train_raw, FB_KEYS)
    fb_profile, usage = fit_arsenal_reference(train_raw)
    train_fs = apply_arsenal_reference(train_raw, fb_profile, usage)
    test_fs = apply_arsenal_reference(test_raw, fb_profile, usage)
    audit_references(train_raw, fb_profile, usage)
    require(test_fs[arsenal_features].isna().all().all(),
            "Leakage: un pitcher no visto recibió features de arsenal.")
    for name, raw, transformed in [("Train", train_raw, train_fs), ("Test", test_raw, test_fs)]:
        # Incluye PitchUID y todas las columnas originales independientes del arsenal.
        columns = [c for c in raw if c not in REFERENCE_COLUMNS]
        pd.testing.assert_frame_equal(raw[columns], transformed[columns])

    for label, frame in [("TRAIN MISSINGNESS", train_fs), ("TEST MISSINGNESS", test_fs)]:
        print("\n" + label)
        print((frame[arsenal_features].isna().mean() * 100).to_string(float_format=lambda x: f"{x:.2f}%"))
    print("\nFB REFERENCES")
    print("Train pitcher-years with FB profile:", len(fb_profile))
    for name, frame in [("Train", train_fs), ("Test", test_fs)]:
        print(f"{name} rows with FB reference:", frame["fb_rel_speed"].notna().sum())
        print(f"{name} coverage: {frame['fb_rel_speed'].notna().mean():.6f}")

    print("\nLEAKAGE AUDIT")
    print("Train pitchers:", len(train_pitchers))
    print("Test pitchers:", len(test_pitchers))
    print("Pitchers compartidos:", len(train_pitchers & test_pitchers))
    print("fb_profile pitcher-years:", len(fb_profile))
    print("fb_profile y usage: todos los pitchers y pitcher-years pertenecen a train: OK")
    print("fb_profile duplicate keys:", fb_profile.duplicated(FB_KEYS).sum())
    print("usage duplicate keys:", usage.duplicated(USAGE_KEYS).sum())
    for name, raw, transformed in [("Train", train_raw, train_fs), ("Test", test_raw, test_fs)]:
        print(f"{name} rows: {len(raw)} -> {len(transformed)}; orden y valores originales: OK")
        if "PitchUID" in raw:
            print(f"{name} duplicate PitchUID: {raw['PitchUID'].duplicated().sum()} -> "
                  f"{transformed['PitchUID'].duplicated().sum()}")
    print("Merges many_to_one: OK; filas añadidas por merges: 0")
    print("Usage en [0, 1], conteos y cocientes (todos los grupos): OK")
    print("FB tipo, umbral, conteos y medias (todos los perfiles): OK")
    print("Test no visto: siete features 100% NaN, sin fallback: OK")
    print("\nMERGE KEY DTYPES")
    for key in USAGE_KEYS:
        print(f"{key}: train={train_raw[key].dtype}, test={test_raw[key].dtype}, "
              f"usage={usage[key].dtype}" + (f", fb_profile={fb_profile[key].dtype}" if key in FB_KEYS else ""))
    counts = fastball_counts(train_raw)
    maxima = counts.groupby(FB_KEYS, observed=True)["n_pitches"].transform("max")
    tied = counts[counts["n_pitches"].eq(maxima)]
    tied = tied[tied.duplicated(FB_KEYS, keep=False)]
    print("\nPRIMARY FASTBALL TIES")
    print("Pitcher-years con empate máximo:", len(tied[FB_KEYS].drop_duplicates()))
    print("Con al menos 20 pitches:", len(tied.loc[tied["n_pitches"].ge(20), FB_KEYS].drop_duplicates()))
    print("Desempate determinístico: conteo, velocidad media válida, prioridad final Four-Seam.")
    available = set(int(y) for y in df_model["year"].dropna().unique())
    usage_results = []
    temporal_checks = []
    for years, evaluation_year in [([2024], 2025), ([2024, 2025], 2026)]:
        if not set(years + [evaluation_year]).issubset(available):
            print(f"Season-forward {years} -> {evaluation_year}: SKIP (años no disponibles)")
            continue
        comparison, checks = season_forward(df_model, years, evaluation_year)
        usage_results.append((years, evaluation_year, comparison))
        temporal_checks.extend(checks)
    changes = primary_stability(df_model)
    changes.to_csv(Path(__file__).resolve().parent / "04_primary_fastball_changes.csv", index=False)
    print("\nPITCH USAGE STABILITY")
    print("Units: proportion [0,1]; unweighted pitcher-pitchtype pairs observed in both periods.")
    for years, evaluation_year, comparison in usage_results:
        print(f"{'-'.join(map(str, years))} -> {evaluation_year}")
        delta = comparison["absolute_usage_change"]
        print("n comparable pitcher-pitchtype pairs:", len(comparison))
        print(f"mean absolute usage change: {delta.mean():.6f}")
        print(f"median absolute usage change: {delta.median():.6f}")
        print(comparison.groupby("AutoPitchType", observed=True)["absolute_usage_change"]
              .agg(["count", "mean", "median"]).to_string(float_format=lambda x: f"{x:.6f}"))
    print("\nPRIMARY FASTBALL DETERMINISM CHECK")
    print("PASS")
    print("Contexts: holdout train, historical 2024, historical 2024-2025, annual descriptive audit.")
    print("\nLEAKAGE AUDIT")
    print("Pitcher holdout unchanged; overlap 0; test arsenal 100% NaN: PASS")
    print("Pitcher holdout reference provenance, unique keys, original row counts/values: PASS")
    for check in temporal_checks:
        print(check)
    print("Future primary FB and future usage used only in descriptive audit: PASS")
    print("No models, no fallbacks, no NaN filling: PASS")
    print("\nAuditoría completada. No se entrenaron modelos ni se modificaron datasets.")


if __name__ == "__main__":
    main()
