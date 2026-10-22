"""Modelos de boosting (HistGradientBoostingClassifier) para los targets V1.

LightGBM y XGBoost no pudieron cargarse en este entorno: ambos requieren la
libreria nativa libomp.dylib, y `brew install libomp` fallo porque las
Command Line Tools de este host no pueden compilar (xcrun no encuentra un
compilador compatible con la arquitectura). Ese es un problema de setup del
host, no del proyecto; se documenta aqui en vez de resolverse en silencio.
En su lugar se usa sklearn.ensemble.HistGradientBoostingClassifier: es
tambien gradient boosting por histogramas (mismo principio que LightGBM),
viene con scikit-learn (ya instalado) y soporta NaN y columnas categoricas
de forma nativa, sin imputer ni one-hot. Si se resuelve el problema de
libomp mas adelante, LightGBM/XGBoost son swaps directos de build_model().

Reutiliza exactamente el pitcher holdout de 03_stuff_baseline.py (misma
poblacion, mismo GroupShuffleSplit, mismo random_state) para que los
resultados sean comparables con el AUC=0.6136 ya reportado ahi.

Arsenal historico deliberadamente EXCLUIDO: 04_fold_safe_arsenal.py ya
demostro que bajo pitcher holdout puro (pitchers no vistos) las siete
features de arsenal quedan 100% ausentes en test, por diseno. Su valor
incremental solo es medible en el escenario season-forward de
06_arsenal_incremental_value.py (pitcher conocido, temporada futura), no
aqui.

Dos familias de targets, siguiendo la separacion Stuff+ / Location+ de la
guia (seccion 1 y 4): Stuff+ usa solo features fisicas puras; Location+ usa
ubicacion, count y BatterSide, sin velocidad ni movimiento. called_strike
se define exactamente como lo cerro 05_target_definitions.py: is_called_strike
sobre todos los pitches, sin restringir a takes (no se inventa una
poblacion nueva no ratificada).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

BASE_DIR = Path(__file__).resolve().parent

STUFF_NUMERIC = [
    "RelSpeed", "SpinRate", "SpinAxis", "RelHeight", "RelSide", "Extension",
    "InducedVertBreak", "HorzBreak", "VertRelAngle", "HorzRelAngle",
    "VertApprAngle", "HorzApprAngle",
]
STUFF_CATEGORICAL = ["AutoPitchType", "PitcherThrows"]

LOCATION_NUMERIC = ["PlateLocHeight", "PlateLocSide", "Balls", "Strikes"]
LOCATION_CATEGORICAL = ["AutoPitchType", "PitcherThrows", "BatterSide"]

FAMILY_FEATURES = {
    "stuff": (STUFF_NUMERIC, STUFF_CATEGORICAL),
    "location": (LOCATION_NUMERIC, LOCATION_CATEGORICAL),
}
CATEGORICAL_COLUMNS = sorted(set(STUFF_CATEGORICAL) | set(LOCATION_CATEGORICAL))


def target_whiff_per_pitch(df):
    eligible = df["is_whiff"].notna()
    return eligible, df["is_whiff"].astype(int)


def target_whiff_given_swing(df):
    eligible = df["is_swing"].eq(1) & df["is_whiff"].notna()
    return eligible, df["is_whiff"].astype(int)


def target_groundball(df):
    eligible = (
        df["is_ball_in_play"].eq(1)
        & df["hit_type"].notna()
        & df["hit_type"].ne("Bunt")
    )
    event = (df["hit_type"] == "GroundBall").astype(int)
    return eligible, event


def target_called_strike(df):
    eligible = df["is_called_strike"].notna()
    return eligible, df["is_called_strike"].astype(int)


def target_chase(df):
    eligible = df["outside_strike_zone"].eq(1) & df["swung_outside_strike_zone"].notna()
    return eligible, df["swung_outside_strike_zone"].astype(int)


TARGETS = [
    ("whiff_per_pitch", "stuff", target_whiff_per_pitch),
    ("whiff_given_swing", "stuff", target_whiff_given_swing),
    ("groundball", "stuff", target_groundball),
    ("called_strike", "location", target_called_strike),
    ("chase", "location", target_chase),
]


def build_model():
    return HistGradientBoostingClassifier(
        categorical_features="from_dtype",
        random_state=42,
        max_iter=300,
        learning_rate=0.05,
        validation_fraction=0.1,
        early_stopping=True,
        n_iter_no_change=15,
    )


def top_permutation_importance(model, X_test, y_test, n_repeats=5, sample_size=20000, seed=42):
    """Sustituto de SHAP: HistGradientBoostingClassifier no expone
    feature_importances_ (no es un ensamble de arboles con conteos de split
    simples). Se corre sobre una muestra de test para mantener el tiempo de
    ejecucion razonable; no altera el modelo ni las metricas reportadas.
    """
    if len(X_test) > sample_size:
        X_sample = X_test.sample(sample_size, random_state=seed)
        y_sample = y_test.loc[X_sample.index]
    else:
        X_sample, y_sample = X_test, y_test
    result = permutation_importance(
        model, X_sample, y_sample, scoring="roc_auc",
        n_repeats=n_repeats, random_state=seed, n_jobs=-1,
    )
    order = np.argsort(result.importances_mean)[::-1]
    return pd.Series(result.importances_mean[order], index=X_sample.columns[order])


def main():
    df = pd.read_parquet(BASE_DIR / "stuff_model_features_v1.parquet")
    df_model = df.dropna(subset=["is_whiff", "pitcher_anon_id"]).copy()

    for col in CATEGORICAL_COLUMNS:
        if col in df_model.columns and not isinstance(df_model[col].dtype, pd.CategoricalDtype):
            df_model[col] = df_model[col].astype("category")

    base_features = STUFF_NUMERIC + STUFF_CATEGORICAL
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(
        df_model[base_features], df_model["is_whiff"], groups=df_model["pitcher_anon_id"]
    ))
    train_df = df_model.iloc[train_idx]
    test_df = df_model.iloc[test_idx]
    train_pitchers = set(train_df["pitcher_anon_id"])
    test_pitchers = set(test_df["pitcher_anon_id"])
    if train_pitchers & test_pitchers:
        raise ValueError("Leakage: pitchers compartidos en el holdout.")

    print("=" * 60)
    print("PITCHER HOLDOUT (identico a 03_stuff_baseline.py / 04_fold_safe_arsenal.py)")
    print("=" * 60)
    print(f"Train pitchers: {len(train_pitchers)} | Test pitchers: {len(test_pitchers)} "
          f"| Compartidos: {len(train_pitchers & test_pitchers)}")
    print("Arsenal historico NO incluido: bajo pitcher holdout puro queda 100% ausente en")
    print("test (ver 04_fold_safe_arsenal.py). Su valor incremental se mide en")
    print("06_arsenal_incremental_value.py bajo season-forward con pitchers conocidos.")

    rows = []
    for name, family, target_fn in TARGETS:
        numeric_features, categorical_features = FAMILY_FEATURES[family]
        features = numeric_features + categorical_features

        train_eligible, train_event = target_fn(train_df)
        test_eligible, test_event = target_fn(test_df)

        X_train = train_df.loc[train_eligible, features]
        y_train = train_event.loc[train_eligible]
        X_test = test_df.loc[test_eligible, features]
        y_test = test_event.loc[test_eligible]

        model = build_model()
        model.fit(X_train, y_train)
        pred = model.predict_proba(X_test)[:, 1]
        null_pred = np.full(len(y_test), y_train.mean())

        row = {
            "target": name, "family": family,
            "n_train": len(y_train), "n_test": len(y_test),
            "base_rate_train": y_train.mean(), "base_rate_test": y_test.mean(),
            "null_auc": roc_auc_score(y_test, null_pred),
            "null_log_loss": log_loss(y_test, null_pred),
            "null_brier": brier_score_loss(y_test, null_pred),
            "boosting_auc": roc_auc_score(y_test, pred),
            "boosting_log_loss": log_loss(y_test, pred),
            "boosting_brier": brier_score_loss(y_test, pred),
        }
        rows.append(row)

        print(f"\n{'=' * 60}")
        print(f"{name} ({family}) | n_train={row['n_train']:,} n_test={row['n_test']:,} "
              f"base_rate_test={row['base_rate_test']:.4f}")
        print("=" * 60)
        print(f"Null:     AUC={row['null_auc']:.4f}  LogLoss={row['null_log_loss']:.5f}  "
              f"Brier={row['null_brier']:.5f}")
        print(f"Boosting: AUC={row['boosting_auc']:.4f}  LogLoss={row['boosting_log_loss']:.5f}  "
              f"Brier={row['boosting_brier']:.5f}")
        print(f"Delta AUC vs Null: {row['boosting_auc'] - row['null_auc']:+.4f}")

        importances = top_permutation_importance(model, X_test, y_test)
        print("Top features (permutation importance, delta AUC; sustituto de SHAP, ver docstring):")
        print(importances.head(8).to_string(float_format=lambda x: f"{x:.5f}"))

    results_df = pd.DataFrame(rows)
    output_path = BASE_DIR / "07_boosting_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResultados guardados: {output_path.name}")
    print("Referencia externa para whiff_per_pitch: regresion logistica fisica de")
    print("03_stuff_baseline.py sobre el mismo split (ver su propia salida para el AUC).")


if __name__ == "__main__":
    main()
