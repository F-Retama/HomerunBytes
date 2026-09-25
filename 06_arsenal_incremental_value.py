"""Valor incremental del arsenal historico sobre Physical Stuff para is_whiff.

Compara Null, Physical Stuff y Physical Stuff + Historical Arsenal sobre
exactamente las mismas filas elegibles de cada ventana season-forward
(2024 -> 2025, 2024-2025 -> 2026). Reutiliza las referencias fold-safe de
04_fold_safe_arsenal.py; no reimplementa el pipeline de arsenal ni sus
auditorias de leakage.

Las features de arsenal de las filas de ENTRENAMIENTO (dentro del propio
historial) se calculan aplicando la referencia del historial completo sobre
si mismo. Esto introduce un sesgo conservador y deliberado: cada fila
contribuye levemente a su propio promedio de referencia, lo que comprime
los diferenciales hacia cero y dificulta -no facilita- mostrar valor
incremental. La aplicacion sobre la temporada de evaluacion futura no
tiene ese sesgo: usa exclusivamente apply_historical_arsenal_reference de
04, que exige que el historial sea estrictamente anterior al ano evaluado
y no imputa referencias faltantes.

04 no se importa con "import 04_fold_safe_arsenal" porque el nombre del
archivo empieza con un digito; se carga por ruta con importlib. El modulo
esta protegido por "if __name__ == '__main__'", asi que cargarlo no
ejecuta su auditoria completa.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
TARGET = "is_whiff"
PHYSICAL_NUMERIC = [
    "RelSpeed", "SpinRate", "SpinAxis", "RelHeight", "RelSide", "Extension",
    "InducedVertBreak", "HorzBreak", "VertRelAngle", "HorzRelAngle",
    "VertApprAngle", "HorzApprAngle",
]
PHYSICAL_CATEGORICAL = ["AutoPitchType", "PitcherThrows"]
ARSENAL_NUMERIC = [
    "velo_diff_vs_fb", "ivb_diff_vs_fb", "hb_diff_vs_fb", "spin_diff_vs_fb",
    "pitch_usage_pct", "release_distance_vs_fb", "extension_diff_vs_fb",
]
SEASON_FORWARD_WINDOWS = [([2024], 2025), ([2024, 2025], 2026)]


def _load_arsenal_module():
    spec = importlib.util.spec_from_file_location(
        "fold_safe_arsenal", BASE_DIR / "04_fold_safe_arsenal.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_pipeline(numeric_features, categorical_features):
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        sparse_threshold=0,
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=2000, C=0.1, solver="saga", random_state=42
        )),
    ])


def evaluate(model, X_train, y_train, X_eval, y_eval):
    model.fit(X_train, y_train)
    pred = model.predict_proba(X_eval)[:, 1]
    return {
        "auc": roc_auc_score(y_eval, pred),
        "log_loss": log_loss(y_eval, pred),
        "brier": brier_score_loss(y_eval, pred),
    }


def null_metrics(y_train, y_eval):
    pred = np.full(len(y_eval), y_train.mean())
    return {
        "auc": roc_auc_score(y_eval, pred),
        "log_loss": log_loss(y_eval, pred),
        "brier": brier_score_loss(y_eval, pred),
    }


def apply_reference_in_sample(arsenal, history, fb, usage):
    """Aplica la referencia historica sobre el propio historial (entrenamiento).

    apply_historical_arsenal_reference exige ano futuro > max(historial), por
    lo que no puede usarse aqui a proposito. _apply_reference es la funcion
    interna que si lo permite; se le pasa el usage renombrado exactamente
    como lo hace 04 antes de aplicarlo a un periodo futuro.
    """
    usage_renamed = usage.rename(columns={
        "historical_pitch_type_n": "pitch_type_count",
        "historical_total_pitches": "pitcher_total_pitches",
        "historical_pitch_usage_pct": "pitch_usage_pct",
    })
    return arsenal._apply_reference(history, fb, usage_renamed, ["pitcher_anon_id"])


def run_window(df_model, arsenal, years, evaluation_year):
    label = f"{'-'.join(map(str, years))} -> {evaluation_year}"
    history = df_model[df_model["year"].isin(years)].copy()
    future = df_model[df_model["year"] == evaluation_year].copy()

    arsenal.check_primary_determinism(history, ["pitcher_anon_id"])
    fb, usage = arsenal.fit_historical_arsenal_reference(history)
    arsenal.audit_historical_reference(history, fb, usage)

    future_fs = arsenal.apply_historical_arsenal_reference(future, fb, usage)
    history_fs = apply_reference_in_sample(arsenal, history, fb, usage)

    eligible_train = history_fs[ARSENAL_NUMERIC].notna().all(axis=1)
    eligible_eval = future_fs[ARSENAL_NUMERIC].notna().all(axis=1)

    train_df = history_fs.loc[eligible_train]
    eval_df = future_fs.loc[eligible_eval]

    print(f"\n{'=' * 60}")
    print(f"SEASON-FORWARD {label}")
    print("=" * 60)
    print(f"Filas de historial con arsenal completo (entrenamiento): "
          f"{len(train_df)} / {len(history_fs)}")
    print(f"Filas de evaluacion elegibles (identicas para las 3 alternativas): "
          f"{len(eval_df)} / {len(future_fs)}")

    y_train = train_df[TARGET]
    y_eval = eval_df[TARGET]

    results = {"Null": null_metrics(y_train, y_eval)}

    physical_cols = PHYSICAL_NUMERIC + PHYSICAL_CATEGORICAL
    physical_model = build_pipeline(PHYSICAL_NUMERIC, PHYSICAL_CATEGORICAL)
    results["Physical Stuff"] = evaluate(
        physical_model, train_df[physical_cols], y_train,
        eval_df[physical_cols], y_eval,
    )

    arsenal_cols = PHYSICAL_NUMERIC + ARSENAL_NUMERIC + PHYSICAL_CATEGORICAL
    arsenal_model = build_pipeline(PHYSICAL_NUMERIC + ARSENAL_NUMERIC, PHYSICAL_CATEGORICAL)
    results["Physical Stuff + Historical Arsenal"] = evaluate(
        arsenal_model, train_df[arsenal_cols], y_train,
        eval_df[arsenal_cols], y_eval,
    )

    for name, metrics in results.items():
        print(f"{name:35s} AUC={metrics['auc']:.4f}  "
              f"LogLoss={metrics['log_loss']:.5f}  Brier={metrics['brier']:.5f}")

    delta_auc = (results["Physical Stuff + Historical Arsenal"]["auc"]
                 - results["Physical Stuff"]["auc"])
    delta_logloss = (results["Physical Stuff"]["log_loss"]
                      - results["Physical Stuff + Historical Arsenal"]["log_loss"])
    delta_brier = (results["Physical Stuff"]["brier"]
                   - results["Physical Stuff + Historical Arsenal"]["brier"])
    print(f"Delta AUC (Arsenal vs Physical):     {delta_auc:+.4f}")
    print(f"Delta Log Loss (positivo = mejora):  {delta_logloss:+.5f}")
    print(f"Delta Brier (positivo = mejora):     {delta_brier:+.5f}")

    return [
        {"window": label, "model": name, "n_train": len(train_df),
         "n_eval": len(eval_df), **metrics}
        for name, metrics in results.items()
    ]


def main():
    df = pd.read_parquet(BASE_DIR / "stuff_model_features_v1.parquet")
    df_model = df.dropna(subset=[TARGET, "pitcher_anon_id"]).copy()

    arsenal = _load_arsenal_module()

    available_years = set(int(y) for y in df_model["year"].dropna().unique())
    all_rows = []
    for years, evaluation_year in SEASON_FORWARD_WINDOWS:
        if not set(years + [evaluation_year]).issubset(available_years):
            print(f"Season-forward {years} -> {evaluation_year}: SKIP (anos no disponibles)")
            continue
        all_rows.extend(run_window(df_model, arsenal, years, evaluation_year))

    results_df = pd.DataFrame(all_rows)
    output_path = BASE_DIR / "06_arsenal_incremental_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResultados guardados: {output_path.name}")
    print("Modelo: regresion logistica (misma familia que 03_stuff_baseline.py).")
    print("No se entreno boosting aqui; esta comparacion aisla el efecto del arsenal")
    print("historico manteniendo fija la arquitectura del modelo.")


if __name__ == "__main__":
    main()
