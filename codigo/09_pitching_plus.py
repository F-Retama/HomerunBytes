"""Pitching+/Out+ V1 y escala Stuff+: stuff + ubicacion + arsenal + contexto.

Reentrena los 5 targets V1 (whiff_per_pitch, whiff_given_swing, groundball,
called_strike, chase) con el conjunto de features COMPLETO que el brief
define para esta capa: fisicas + ubicacion + arsenal historico (fold-safe,
via 04_fold_safe_arsenal.py) + altitude_category + Balls/Strikes +
BatterSide. A diferencia de 07/08 (que aislaban Stuff+ puro y Location+ puro
para medir robustez de cada capa por separado), aqui las cinco sub-señales
comparten el mismo set de features completo.

Validacion: season-forward (04/06), no pitcher holdout puro, porque el
arsenal historico requiere temporadas previas y queda 100% ausente para
pitchers sin historial. Esto tambien cumple el season holdout que pide el
criterio de robustez del reto. Se corren las dos ventanas de 04/06
(2024->2025, 2024-2025->2026); la segunda (mas historial) es la ventana
"primaria" para las columnas finales guardadas.

Problema resuelto antes de combinar (discutido 24/09/2026): whiff_given_swing,
chase y groundball tienen poblaciones elegibles mutuamente excluyentes por
fila (is_swing==1, outside_strike_zone==1, is_ball_in_play==1
respectivamente). Un promedio literal de las 5 probabilidades por fila esta
indefinido para casi todos los pitches si nos limitamos a la poblacion
observada de cada target. Resolucion: cada sub-modelo se ENTRENA sobre su
propia poblacion condicional (igual que en 07/08, para aprender la
probabilidad condicional correcta), pero para ANOTAR el score se aplica el
modelo ya entrenado a TODO pitch de la poblacion comun elegible (arsenal
completo + altitude_category valida), sin importar si ese pitch especifico
fue swing/BIP/etc. Es la pregunta contrafactual estandar ("si hubiera
swing, que tan probable es el whiff") y es exactamente lo que un modelo
entrenado sobre esa poblacion permite responder para cualquier fila con las
features necesarias.

Correccion semantica (24/09/2026): aplicar un modelo fuera de su dominio de
entrenamiento produce un numero, no necesariamente la cantidad nombrada. El
modelo de chase se aprende solo sobre pitches fuera de zona; sobre un pitch
dentro de zona (con su PlateLocHeight/Side intacto) es extrapolacion. groundball
se aprende sobre contacto elegible: es "propension a groundball condicionada a
contacto". Por eso las columnas finales llevan la poblacion en el nombre
(p_whiff_per_pitch, p_whiff_given_swing, p_groundball_given_contact,
p_called_strike, p_chase_given_outside_zone) y se calcula ademas una version
de sensibilidad del score SIN chase (stuff_plus_no_chase).

Los modelos entrenados, las referencias de estandarizacion y la cohorte de
evaluacion se guardan en 09_pitching_plus_bundle.pkl (ignorado por git, como
todo .pkl) para que 10_altitude_counterfactual.py use exactamente estos
modelos sin reentrenar.

Combinacion en un solo score (decision del equipo, 24/09/2026): NO se usan
pesos de valor de carrera (The Book) porque no hay traduccion limpia de esos
pesos a probabilidades con poblaciones elegibles distintas. Tampoco se
ajustan pesos por regresion contra un proxy de valor observado todavia: eso
reabre el problema de poblacion de PA-terminal que ya bloqueo xRunValue en
05_target_definitions.py. V1 adoptado: estandarizar cada probabilidad
predicha (z-score) contra su propia media/SD en TRAIN (nunca en test/eval,
para no filtrar la referencia "liga promedio"), y promediar con pesos
iguales. Las cinco probabilidades ya estan orientadas "mayor = mejor para
el pitcher" sin invertir signos (groundball tambien, consistente con H3).
Se guardan los cinco componentes por separado ademas del score combinado.
Pendiente para V2, explicitamente NO resuelto aqui: sensibilidad de pesos y
version con pesos aprendidos contra valor de carrera cuando exista una
definicion reproducible de xRunValue/run expectancy.

Escala final: Stuff+ = 100 + 10 * z(Pitching+/Out+ crudo), estandarizado
tambien contra la media/SD de train, siguiendo la definicion formal de
escala del brief.
"""

import importlib.util
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

BASE_DIR = Path(__file__).resolve().parent

STUFF_NUMERIC = [
    "RelSpeed", "SpinRate", "SpinAxis", "RelHeight", "RelSide", "Extension",
    "InducedVertBreak", "HorzBreak", "VertRelAngle", "HorzRelAngle",
    "VertApprAngle", "HorzApprAngle",
]
LOCATION_NUMERIC = ["PlateLocHeight", "PlateLocSide", "Balls", "Strikes"]
ARSENAL_NUMERIC = [
    "velo_diff_vs_fb", "ivb_diff_vs_fb", "hb_diff_vs_fb", "spin_diff_vs_fb",
    "pitch_usage_pct", "release_distance_vs_fb", "extension_diff_vs_fb",
]
FULL_NUMERIC = STUFF_NUMERIC + LOCATION_NUMERIC + ARSENAL_NUMERIC
FULL_CATEGORICAL = ["AutoPitchType", "PitcherThrows", "BatterSide", "altitude_category"]
FULL_FEATURES = FULL_NUMERIC + FULL_CATEGORICAL

COMPONENTS = [
    "whiff_per_pitch", "whiff_given_swing", "groundball", "called_strike", "chase",
]
COMPONENT_LABELS = {
    "whiff_per_pitch": "p_whiff_per_pitch",
    "whiff_given_swing": "p_whiff_given_swing",
    "groundball": "p_groundball_given_contact",
    "called_strike": "p_called_strike",
    "chase": "p_chase_given_outside_zone",
}
SEASON_FORWARD_WINDOWS = [([2024], 2025), ([2024, 2025], 2026)]
PRIMARY_WINDOW_LABEL = "2024-2025 -> 2026"


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, BASE_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_full_features(df_model, arsenal_module):
    for col in FULL_CATEGORICAL:
        if col in df_model.columns and not isinstance(df_model[col].dtype, pd.CategoricalDtype):
            df_model[col] = df_model[col].astype("category")
    return df_model


def common_eligible_mask(df, arsenal_features):
    return df[arsenal_features].notna().all(axis=1) & df["altitude_category"].notna()


def train_component(name, target_fn, history_fs, future_fs, hist_common, fut_common, boosting):
    train_eligible, train_event = target_fn(history_fs)
    train_eligible = train_eligible & hist_common
    X_train = history_fs.loc[train_eligible, FULL_FEATURES]
    y_train = train_event.loc[train_eligible]

    model = boosting.build_model()
    model.fit(X_train, y_train)

    # Anotacion: se aplica a TODA la poblacion comun elegible, no solo a la
    # poblacion condicional del target (ver docstring: pregunta contrafactual).
    X_train_score = history_fs.loc[hist_common, FULL_FEATURES]
    X_future_score = future_fs.loc[fut_common, FULL_FEATURES]
    train_scores = pd.Series(model.predict_proba(X_train_score)[:, 1], index=X_train_score.index)
    future_scores = pd.Series(model.predict_proba(X_future_score)[:, 1], index=X_future_score.index)

    # Discriminacion propia del componente: solo sobre su poblacion condicional real,
    # igual que 07/08, para poder comparar contra esos resultados.
    test_eligible, test_event = target_fn(future_fs)
    test_eligible = test_eligible & fut_common
    metrics = None
    if test_eligible.sum() > 0 and test_event.loc[test_eligible].nunique() > 1:
        y_test = test_event.loc[test_eligible]
        pred_test = model.predict_proba(future_fs.loc[test_eligible, FULL_FEATURES])[:, 1]
        metrics = {
            "n_train_label": int(train_eligible.sum()),
            "n_test_label": int(test_eligible.sum()),
            "auc": roc_auc_score(y_test, pred_test),
            "log_loss": log_loss(y_test, pred_test),
            "brier": brier_score_loss(y_test, pred_test),
        }
    return train_scores, future_scores, metrics, model


def standardize(train_scores, future_scores):
    mean, std = train_scores.mean(), train_scores.std()
    return (train_scores - mean) / std, (future_scores - mean) / std, mean, std


def run_window(df_model, years, evaluation_year, arsenal, six, boosting):
    label = f"{'-'.join(map(str, years))} -> {evaluation_year}"
    history = df_model[df_model["year"].isin(years)].copy()
    future = df_model[df_model["year"] == evaluation_year].copy()

    fb, usage = arsenal.fit_historical_arsenal_reference(history)
    arsenal.audit_historical_reference(history, fb, usage)
    future_fs = arsenal.apply_historical_arsenal_reference(future, fb, usage)
    history_fs = six.apply_reference_in_sample(arsenal, history, fb, usage)

    hist_common = common_eligible_mask(history_fs, ARSENAL_NUMERIC)
    fut_common = common_eligible_mask(future_fs, ARSENAL_NUMERIC)

    print(f"\n{'=' * 60}\nPITCHING+ SEASON-FORWARD {label}\n{'=' * 60}")
    print(f"Train (historial) filas comunes elegibles: {int(hist_common.sum()):,} / {len(history_fs):,}")
    print(f"Future (evaluacion) filas comunes elegibles: {int(fut_common.sum()):,} / {len(future_fs):,}")

    z_train_cols, z_future_cols = {}, {}
    metrics_rows = []
    # boosting.TARGETS es una lista de tuplas (name, family, fn); indexar por nombre.
    target_fn_by_name = {name: fn for name, _family, fn in boosting.TARGETS}

    raw_train, raw_future = {}, {}
    models, component_stats = {}, {}
    for name in COMPONENTS:
        target_fn = target_fn_by_name[name]
        train_scores, future_scores, metrics, model = train_component(
            name, target_fn, history_fs, future_fs, hist_common, fut_common, boosting
        )
        raw_train[name] = train_scores
        raw_future[name] = future_scores
        z_tr, z_fu, mean, std = standardize(train_scores, future_scores)
        models[name] = model
        component_stats[name] = (mean, std)
        z_train_cols[name] = z_tr
        z_future_cols[name] = z_fu
        print(f"\n{name}: train_mean_prob={mean:.4f} train_std_prob={std:.4f}")
        if metrics:
            print(f"  Discriminacion propia (poblacion condicional real): "
                  f"AUC={metrics['auc']:.4f} LogLoss={metrics['log_loss']:.5f} "
                  f"Brier={metrics['brier']:.5f} (n_test_label={metrics['n_test_label']:,})")
            metrics_rows.append({"window": label, "component": name, **metrics})
        else:
            print("  Discriminacion propia: no evaluable (denominador insuficiente).")

    z_train_df = pd.DataFrame(z_train_cols)
    z_future_df = pd.DataFrame(z_future_cols)
    pitching_plus_train = z_train_df.mean(axis=1)
    pitching_plus_future = z_future_df.mean(axis=1)

    pp_mean, pp_std = pitching_plus_train.mean(), pitching_plus_train.std()
    no_chase = [k for k in COMPONENTS if k != "chase"]
    pp4_train = z_train_df[no_chase].mean(axis=1)
    pp4_future = z_future_df[no_chase].mean(axis=1)
    pp4_mean, pp4_std = pp4_train.mean(), pp4_train.std()
    stuff4_future = 100 + 10 * (pp4_future - pp4_mean) / pp4_std
    stuff_plus_train = 100 + 10 * (pitching_plus_train - pp_mean) / pp_std
    stuff_plus_future = 100 + 10 * (pitching_plus_future - pp_mean) / pp_std

    print(f"\nPitching+/Out+ crudo (train): mean={pitching_plus_train.mean():.4f} "
          f"std={pitching_plus_train.std():.4f}")
    print(f"Stuff+ (future, escala 100+10z): mean={stuff_plus_future.mean():.2f} "
          f"std={stuff_plus_future.std():.2f} min={stuff_plus_future.min():.1f} "
          f"max={stuff_plus_future.max():.1f}")

    future_out = pd.DataFrame({COMPONENT_LABELS[k]: v for k, v in raw_future.items()})
    for k in COMPONENTS:
        future_out["z_" + COMPONENT_LABELS[k][2:]] = z_future_cols[k]
    future_out["pitching_plus_raw"] = pitching_plus_future
    future_out["stuff_plus_score"] = stuff_plus_future
    future_out["stuff_plus_no_chase"] = stuff4_future
    future_out["window"] = label
    future_out["AutoPitchType"] = future_fs.loc[fut_common, "AutoPitchType"].values
    future_out["altitude_category"] = future_fs.loc[fut_common, "altitude_category"].values
    future_out["pitcher_anon_id"] = future_fs.loc[fut_common, "pitcher_anon_id"].values
    future_out["year"] = future_fs.loc[fut_common, "year"].values

    sanity = (
        future_out.groupby("AutoPitchType", observed=True)["stuff_plus_score"]
        .agg(["count", "mean", "std"]).sort_values("mean", ascending=False)
    )
    print("\nSanity check: Stuff+ medio por pitch type (esperado: Splitter/Slider/Curveball "
          "por encima de Four-Seam/Sinker, consistente con 03_stuff_baseline.py):")
    print(sanity.to_string(float_format=lambda x: f"{x:.2f}"))

    bundle_entry = {
        "label": label,
        "models": models,
        "component_stats": component_stats,
        "pp5": (pp_mean, pp_std),
        "pp4_no_chase": (pp4_mean, pp4_std),
        "features": FULL_FEATURES,
        "future_frame": future_fs.loc[
            fut_common,
            FULL_FEATURES + ["pitcher_anon_id", "year", "outside_strike_zone", "PitchUID",
                             "primary_fastball_type"],
        ].reset_index(drop=True),
    }
    return future_out, pd.DataFrame(metrics_rows), bundle_entry


def main():
    arsenal = _load_module("fold_safe_arsenal", "04_fold_safe_arsenal.py")
    six = _load_module("arsenal_incremental_value", "06_arsenal_incremental_value.py")
    boosting = _load_module("boosting_models", "07_boosting_models.py")

    df = pd.read_parquet(BASE_DIR / "stuff_model_features_v1.parquet")
    df_model = df.dropna(subset=["is_whiff", "pitcher_anon_id"]).copy()
    df_model = build_full_features(df_model, arsenal)

    available_years = set(int(y) for y in df_model["year"].dropna().unique())
    all_scores, all_metrics, bundle = [], [], {}
    for years, evaluation_year in SEASON_FORWARD_WINDOWS:
        if not set(years + [evaluation_year]).issubset(available_years):
            print(f"Season-forward {years} -> {evaluation_year}: SKIP (anos no disponibles)")
            continue
        scores, metrics, bundle_entry = run_window(df_model, years, evaluation_year, arsenal, six, boosting)
        all_scores.append(scores)
        all_metrics.append(metrics)
        bundle[bundle_entry["label"]] = bundle_entry

    scores_df = pd.concat(all_scores, ignore_index=True)
    metrics_df = pd.concat(all_metrics, ignore_index=True)
    scores_path = BASE_DIR / "09_pitching_plus_scores.csv"
    metrics_path = BASE_DIR / "09_pitching_plus_component_metrics.csv"
    scores_df.to_csv(scores_path, index=False)
    joblib.dump(bundle, BASE_DIR / "09_pitching_plus_bundle.pkl")
    metrics_df.to_csv(metrics_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"Ventana primaria para uso posterior (contrafactual de altitud): {PRIMARY_WINDOW_LABEL}")
    print("=" * 60)
    print(f"Scores guardados: {scores_path.name} ({len(scores_df):,} filas)")
    print(f"Metricas por componente guardadas: {metrics_path.name}")
    print("\nComparacion de discriminacion por componente vs 07 (Stuff/Location puros, pitcher holdout)"
          " y 08 (park holdout), donde aplique: revisar manualmente los tres CSV, mismos targets.")


if __name__ == "__main__":
    main()
