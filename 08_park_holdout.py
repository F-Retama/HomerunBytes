"""Park holdout: generaliza el modelo a un entorno de altitud no visto.

Entrena en No Altitude + Medium Altitude, evalua en Extreme Altitude (Harp
Helu y estadios comparables). Complementa -no reemplaza- el pitcher holdout
de 07_boosting_models.py: ese prueba generalizacion a PITCHERS no vistos
dentro de la misma mezcla de entornos; este prueba generalizacion a un
ENTORNO no visto. Los pitchers compartidos entre train y test son esperados
y correctos aqui (no es leakage): la pregunta es ambiental, no de pitcher.

Decision explicita (24/09/2026): altitude_category NO se incluye como
feature en esta primera version, por dos razones:
1. Entrenar sin Extreme Altitude deja esa categoria enteramente no vista
   para el codificador categorico al momento de predecir -un artefacto
   tecnico de codificacion, no una medida de generalizacion real.
2. Incluirla mezclaria dos preguntas distintas: "¿el modelo generaliza su
   entendimiento fisico/de ubicacion a un entorno diferente?" vs. "¿usar el
   entorno como insumo explicito ayuda?". Esta version responde solo la
   primera. La segunda queda para altitude_counterfactual.py (planeado):
   entrenar CON altitude_category sobre todo el dataset y comparar
   predicciones al cambiar la categoria, con las demas features fijas.

Se verifico antes de escribir este script que ninguna otra columna
categorica usada (AutoPitchType, PitcherThrows, BatterSide) tiene niveles
presentes en Extreme Altitude ausentes de {No Altitude, Medium Altitude}.
El script igual lo re-verifica en tiempo de ejecucion y falla si encuentra
alguno: no se asume, se comprueba.

Reutiliza integramente las definiciones de targets, familias de features y
build_model()/top_permutation_importance() de 07_boosting_models.py; no se
reimplementan. 07 se carga por ruta con importlib porque su nombre empieza
con un digito; esta protegido por "if __name__ == '__main__'", asi que
cargarlo no vuelve a correr su propio entrenamiento.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

BASE_DIR = Path(__file__).resolve().parent
TRAIN_ALTITUDES = ["No Altitude", "Medium Altitude"]
TEST_ALTITUDE = "Extreme Altitude"


def _load_boosting_module():
    spec = importlib.util.spec_from_file_location(
        "boosting_models", BASE_DIR / "07_boosting_models.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_no_unseen_categories(train_df, test_df, categorical_columns):
    for col in categorical_columns:
        train_cats = set(train_df[col].dropna().unique())
        test_cats = set(test_df[col].dropna().unique())
        unseen = test_cats - train_cats
        if unseen:
            raise ValueError(f"Categorias no vistas en {col} bajo park holdout: {unseen}")


def main():
    boosting = _load_boosting_module()

    df = pd.read_parquet(BASE_DIR / "stuff_model_features_v1.parquet")
    df_model = df.dropna(subset=["is_whiff", "pitcher_anon_id"]).copy()

    for col in boosting.CATEGORICAL_COLUMNS:
        if col in df_model.columns and not isinstance(df_model[col].dtype, pd.CategoricalDtype):
            df_model[col] = df_model[col].astype("category")

    excluded = df_model["altitude_category"].isna().sum()
    train_df = df_model[df_model["altitude_category"].isin(TRAIN_ALTITUDES)]
    test_df = df_model[df_model["altitude_category"].eq(TEST_ALTITUDE)]
    check_no_unseen_categories(train_df, test_df, boosting.CATEGORICAL_COLUMNS)

    train_pitchers = set(train_df["pitcher_anon_id"])
    test_pitchers = set(test_df["pitcher_anon_id"])
    shared = train_pitchers & test_pitchers

    print("=" * 60)
    print(f"PARK HOLDOUT: train = {TRAIN_ALTITUDES}, test = {TEST_ALTITUDE}")
    print("=" * 60)
    print(f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,} | "
          f"Excluidas (altitude_category ausente/invalida): {excluded:,}")
    print(f"Train pitchers: {len(train_pitchers)} | Test pitchers: {len(test_pitchers)} | "
          f"Compartidos: {len(shared)} ({100 * len(shared) / len(test_pitchers):.1f}% de test)")
    print("Pitchers compartidos son esperados y correctos aqui: se prueba generalizacion")
    print("a un ENTORNO no visto, no a un pitcher no visto (eso ya lo prueba 07).")
    print("altitude_category EXCLUIDA como feature (ver docstring).")
    print("Sin categorias no vistas en AutoPitchType/PitcherThrows/BatterSide: verificado arriba.")

    rows = []
    for name, family, target_fn in boosting.TARGETS:
        numeric_features, categorical_features = boosting.FAMILY_FEATURES[family]
        features = numeric_features + categorical_features

        train_eligible, train_event = target_fn(train_df)
        test_eligible, test_event = target_fn(test_df)

        X_train = train_df.loc[train_eligible, features]
        y_train = train_event.loc[train_eligible]
        X_test = test_df.loc[test_eligible, features]
        y_test = test_event.loc[test_eligible]

        model = boosting.build_model()
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

        importances = boosting.top_permutation_importance(model, X_test, y_test)
        print("Top features (permutation importance):")
        print(importances.head(8).to_string(float_format=lambda x: f"{x:.5f}"))

    results_df = pd.DataFrame(rows)
    output_path = BASE_DIR / "08_park_holdout_results.csv"
    results_df.to_csv(output_path, index=False)

    pitcher_holdout_path = BASE_DIR / "07_boosting_results.csv"
    if pitcher_holdout_path.exists():
        pitcher_holdout = pd.read_csv(pitcher_holdout_path).set_index("target")["boosting_auc"]
        print(f"\n{'=' * 60}")
        print("COMPARACION: PITCHER HOLDOUT (07) vs PARK HOLDOUT (08)")
        print("=" * 60)
        for row in rows:
            name = row["target"]
            if name in pitcher_holdout.index:
                delta = row["boosting_auc"] - pitcher_holdout[name]
                print(f"{name:20s} pitcher_holdout_AUC={pitcher_holdout[name]:.4f}  "
                      f"park_holdout_AUC={row['boosting_auc']:.4f}  delta={delta:+.4f}")
    else:
        print("\n07_boosting_results.csv no encontrado: correr 07_boosting_models.py primero "
              "para tener la comparacion pitcher holdout vs park holdout.")

    print(f"\nResultados guardados: {output_path.name}")


if __name__ == "__main__":
    main()
