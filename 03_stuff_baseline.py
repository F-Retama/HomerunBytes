import warnings
import traceback

import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    brier_score_loss,
    accuracy_score
)

# ============================================================
# 1. CARGA
# ============================================================

df = pd.read_parquet("stuff_model_features_v1.parquet")

print("=" * 60)
print("DATASET")
print("=" * 60)
print(df.shape)

# ============================================================
# 2. TARGET
# ============================================================

# Baseline V1: probabilidad de whiff por pitch total, no condicionada a swing.
TARGET = "is_whiff"

df_model = df.dropna(subset=[TARGET, "pitcher_anon_id"]).copy()

print("\nTarget mean:")
print(df_model[TARGET].mean())

# ============================================================
# 3. FEATURES FÍSICAS BASE
# ============================================================

numeric_features = [
    "RelSpeed",
    "SpinRate",
    "SpinAxis",
    "RelHeight",
    "RelSide",
    "Extension",
    "InducedVertBreak",
    "HorzBreak",
    "VertRelAngle",
    "HorzRelAngle",
    "VertApprAngle",
    "HorzApprAngle",
]

categorical_features = [
    "AutoPitchType",
    "PitcherThrows",
]

# Diagnóstico previo al entrenamiento; los NaN los maneja el imputador.
print("\n" + "=" * 60)
print("CHECK INFINITOS")
print("=" * 60)
inf_counts = {
    col: int(np.isinf(pd.to_numeric(df_model[col], errors="coerce")).sum())
    for col in numeric_features
}
for col, count in inf_counts.items():
    print(f"{col}: {count}")
if any(inf_counts.values()):
    raise ValueError("Hay infinitos en las features físicas; revisar datos antes de entrenar.")

features = numeric_features + categorical_features

X = df_model[features]
y = df_model[TARGET]
groups = df_model["pitcher_anon_id"]

# ============================================================
# 4. SPLIT POR PITCHER
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

train_pitchers = set(groups.iloc[train_idx])
test_pitchers = set(groups.iloc[test_idx])

print("\n" + "=" * 60)
print("SPLIT")
print("=" * 60)

print("Train rows:", len(X_train))
print("Test rows:", len(X_test))
print("Train pitchers:", len(train_pitchers))
print("Test pitchers:", len(test_pitchers))
print("Pitchers compartidos:", len(train_pitchers & test_pitchers))

# ============================================================
# 5. BASELINE NULO
# ============================================================

null_prob = y_train.mean()

null_predictions = np.full(
    len(y_test),
    null_prob
)

null_auc = roc_auc_score(
    y_test,
    null_predictions
)

null_logloss = log_loss(
    y_test,
    null_predictions
)

null_brier = brier_score_loss(
    y_test,
    null_predictions
)

print("\n" + "=" * 60)
print("BASELINE NULO")
print("=" * 60)

print("Train whiff rate:", null_prob)
print("Test whiff rate:", y_test.mean())

print("AUC:", null_auc)
print("Log Loss:", null_logloss)
print("Brier:", null_brier)

# ============================================================
# BASELINE POR PITCH TYPE (tasas calculadas únicamente en train)
# ============================================================

pitch_type_rates = (
    df_model.iloc[train_idx]
    .groupby("AutoPitchType", observed=True)[TARGET]
    .mean()
)
# Convertir a float antes de fillna también admite columnas categóricas.
pitch_type_pred = (
    df_model.iloc[test_idx]["AutoPitchType"]
    .map(pitch_type_rates)
    .astype(float)
    .fillna(null_prob)
    .to_numpy()
)

pitch_type_auc = roc_auc_score(y_test, pitch_type_pred)
pitch_type_logloss = log_loss(y_test, pitch_type_pred)
pitch_type_brier = brier_score_loss(y_test, pitch_type_pred)

print("\n" + "=" * 60)
print("BASELINE POR PITCH TYPE")
print("=" * 60)
print("AUC:", pitch_type_auc)
print("Log Loss:", pitch_type_logloss)
print("Brier:", pitch_type_brier)

# ============================================================
# 6. PREPROCESAMIENTO
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ],
    sparse_threshold=0
)

# ============================================================
# 7. REGRESIÓN LOGÍSTICA
# ============================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                C=0.1,
                solver="saga",
                random_state=42,
                class_weight=None
            )
        )
    ]
)

print("\nEntrenando regresión logística...")

model.fit(
    X_train,
    y_train
)

# Inspección sin modificar datos, solver ni límites de las variables.
print("\n" + "=" * 60)
print("DIAGNÓSTICO NUMÉRICO")
print("=" * 60)

X_train_transformed = model.named_steps["preprocessor"].transform(X_train)
X_test_transformed = model.named_steps["preprocessor"].transform(X_test)
print("Train transformed shape:", X_train_transformed.shape)
print("Test transformed shape:", X_test_transformed.shape)
print("\nTIPO DE MATRIZ TRANSFORMADA")
print(type(X_test_transformed))
print("dtype:", X_test_transformed.dtype)

# En matrices dispersas inspeccionar solo los valores almacenados evita densificar.
for label, transformed in [("TRAIN", X_train_transformed), ("TEST", X_test_transformed)]:
    values = transformed.data if hasattr(transformed, "toarray") else transformed.ravel()
    print("\n" + label)
    print("NaN:", np.isnan(values).sum())
    print("Inf:", np.isinf(values).sum())
    print("Max abs:", np.max(np.abs(values)) if values.size else 0.0)

clf = model.named_steps["classifier"]
print("\nCOEFICIENTES")
print("NaN:", np.isnan(clf.coef_).sum())
print("Inf:", np.isinf(clf.coef_).sum())
print("Max abs coefficient:", np.max(np.abs(clf.coef_)))
print("Intercept:", clf.intercept_)

# Supresión limitada a esta inspección; predict_proba mantiene los warnings visibles.
with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
    decision = clf.decision_function(X_test_transformed)
print("\nDECISION FUNCTION")
print("NaN:", np.isnan(decision).sum())
print("Inf:", np.isinf(decision).sum())
print("Min:", np.nanmin(decision))
print("Max:", np.nanmax(decision))

print("\n" + "=" * 60)
print("RANGOS EXTREMOS DE FEATURES")
print("=" * 60)
for col in numeric_features:
    x = pd.to_numeric(df_model[col], errors="coerce")
    print(
        f"{col:20s} "
        f"min={x.min():10.3f} "
        f"p0.1={x.quantile(0.001):10.3f} "
        f"p1={x.quantile(0.01):10.3f} "
        f"p99={x.quantile(0.99):10.3f} "
        f"p99.9={x.quantile(0.999):10.3f} "
        f"max={x.max():10.3f}"
    )

pred_prob = model.predict_proba(
    X_test
)[:, 1]

print("\nPREDICCIONES")
print(pd.Series(pred_prob).describe(percentiles=[.001, .01, .05, .5, .95, .99, .999]))
print("Probabilidades exactamente 0:", (pred_prob == 0).sum())
print("Probabilidades exactamente 1:", (pred_prob == 1).sum())


# ============================================================
# CHECK FINAL: PREDICT_PROBA VS SIGMOID MANUAL
# ============================================================

manual_prob = 1 / (1 + np.exp(-decision))
max_diff = np.max(np.abs(pred_prob - manual_prob))

print("\n" + "=" * 60)
print("CHECK PREDICT_PROBA")
print("=" * 60)
print("Máxima diferencia predict_proba vs sigmoid manual:", max_diff)
print("Predicciones manuales NaN:", np.isnan(manual_prob).sum())
print("Predicciones manuales Inf:", np.isinf(manual_prob).sum())

print("\n" + "=" * 60)
print("LOCALIZACIÓN DEL WARNING")
print("=" * 60)
try:
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        test_pred = model.predict_proba(X_test)[:, 1]
    print("predict_proba terminó sin RuntimeWarning.")
except RuntimeWarning as error:
    print("RuntimeWarning capturado en predict_proba:")
    print(error)
    # Incluir el traceback: imprimir solo el mensaje no localiza la operación.
    print(traceback.format_exc())

# ============================================================
# AISLAMIENTO DE LA OPERACIÓN NUMÉRICA
# ============================================================

print("\n" + "=" * 60)
print("MATMUL DIRECTO")
print("=" * 60)
Xtr = model.named_steps["preprocessor"].transform(X_test)
clf = model.named_steps["classifier"]
beta = clf.coef_[0]
intercept = clf.intercept_[0]
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", RuntimeWarning)
    raw_matmul = Xtr @ beta + intercept
print("Warnings:", len(caught))
for w in caught:
    print(type(w.message).__name__, ":", w.message)
print("NaN:", np.isnan(raw_matmul).sum())
print("Inf:", np.isinf(raw_matmul).sum())
print("Max diff vs decision_function:", np.max(np.abs(raw_matmul - decision)))

print("\n" + "=" * 60)
print("COMPARACIÓN OPERACIONES")
print("=" * 60)
operations = {}
for name, func in {
    "matmul": lambda: Xtr @ beta,
    "dot": lambda: np.dot(Xtr, beta),
    "einsum": lambda: np.einsum("ij,j->i", Xtr, beta),
}.items():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        result = func()
    operations[name] = result
    print(name, "| warnings:", len(caught), "| NaN:", np.isnan(result).sum(),
          "| Inf:", np.isinf(result).sum())
    for w in caught:
        print("   ", w.message)
print("matmul vs dot:", np.max(np.abs(operations["matmul"] - operations["dot"])))
print("matmul vs einsum:", np.max(np.abs(operations["matmul"] - operations["einsum"])))

print("\nARRAY INFO")
print("X dtype:", Xtr.dtype)
print("beta dtype:", beta.dtype)
print("X C contiguous:", Xtr.flags["C_CONTIGUOUS"])
print("X F contiguous:", Xtr.flags["F_CONTIGUOUS"])
print("beta contiguous:", beta.flags["C_CONTIGUOUS"])
print("\nNP ERROR SETTINGS")
print(np.geterr())
print("\nNUMPY CONFIG")
np.show_config()

# ============================================================
# 8. MÉTRICAS
# ============================================================

auc = roc_auc_score(
    y_test,
    pred_prob
)

ll = log_loss(
    y_test,
    pred_prob
)

brier = brier_score_loss(
    y_test,
    pred_prob
)

print("\n" + "=" * 60)
print("REGRESIÓN LOGÍSTICA — STUFF FÍSICO")
print("=" * 60)

print("AUC:", auc)
print("Log Loss:", ll)
print("Brier:", brier)

print("\nMEJORA VS BASELINE")

print(
    "Delta AUC:",
    auc - null_auc
)

print(
    "Delta Log Loss:",
    null_logloss - ll
)

print(
    "Delta Brier:",
    null_brier - brier
)

# ============================================================
# MEJORA MODELO FÍSICO VS PITCH TYPE
# ============================================================

print("\nMEJORA MODELO FÍSICO VS PITCH TYPE")
print("Delta AUC:", auc - pitch_type_auc)
print("Delta Log Loss:", pitch_type_logloss - ll)
print("Delta Brier:", pitch_type_brier - brier)

# ============================================================
# 9. PREDICCIÓN MEDIA POR PITCH TYPE
# ============================================================

results = df_model.iloc[test_idx][
    [
        "AutoPitchType",
        "is_whiff"
    ]
].copy()

results["pred_whiff"] = pred_prob

pitch_type_results = (
    results
    .groupby(
        "AutoPitchType",
        observed=True
    )
    .agg(
        n=("is_whiff", "count"),
        actual_whiff=("is_whiff", "mean"),
        predicted_whiff=("pred_whiff", "mean"),
    )
    .reset_index()
)

print("\n" + "=" * 60)
print("WHIFF POR PITCH TYPE")
print("=" * 60)

print(
    pitch_type_results
    .sort_values("actual_whiff", ascending=False)
    .to_string(index=False)
)

# ============================================================
# 10. COEFICIENTES
# ============================================================

feature_names = (
    model
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

coefficients = (
    model
    .named_steps["classifier"]
    .coef_[0]
)

coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients
})

coef_df["abs_coefficient"] = (
    coef_df["coefficient"].abs()
)

print("\n" + "=" * 60)
print("COEFICIENTES MÁS IMPORTANTES")
print("=" * 60)

print(
    coef_df
    .sort_values(
        "abs_coefficient",
        ascending=False
    )
    .head(25)
    .to_string(index=False)
)

