"""Stuff+ / Location+ / Pitching+ V2 (Bloque B): targets completos y valor LMB.

Reutiliza el pipeline de 16_three_scores.py (mismas familias, espejo de
lateralidad, variantes base/arsenal, pesos OOF del historial, escala 100+10z) con
tres cambios, todos construidos en 17_sequence_and_contact.py:

1. Componentes: los 5 de V1 + weak_contact (EV < 60 mph | BIP medido),
   hard_hit (EV >= 95 mph | BIP medido), hr_given_contact (HR | BIP) y
   out_within_PA (la PA del pitch termina en out). hard_hit y HR se orientan con
   signo negativo (mas alto = peor para el pitcher) antes de ponderar, para que los
   pesos no negativos tengan sentido. Barrel no se incluye (17).
2. Pitching+ agrega contexto de secuencia y juego: Outs, Inning y prev_pitch_type
   (pitch anterior de la misma PA). Stuff+ sigue siendo fisica pura; Location+ no
   cambia.
3. Valor realizado para aprender pesos y evaluar: run expectancy LMB por conteo x
   outs (36 estados) aprendido solo con el historial de cada ventana, sin pagos
   externos (reemplaza The Book de 13). Nulo = valor medio por conteo x outs en el
   historial (~0), el baseline 'xRunValue promedio por conteo' del criterio 1.

4. Pesos de Stuff+ (decision tras la primera corrida, 26/09/2026): los pesos
   aprendidos regresando el valor sobre predicciones SOLO fisicas estan mal
   identificados: el called strike 'fisico' mide la tendencia del tipo de pitch a
   ir a la zona (intencion/uso), no stuff. Resultado de esa corrida: peso ~0.55-0.60
   en called strike, orden por tipo implausible (Changeup/Splitter ~88) y cero
   validez a nivel pitcher (corr con el valor 2026 = 0.017, IC incluye 0). Regla
   adoptada: Stuff+ usa el valor por evento aprendido en Pitching+ (donde ubicacion
   y conteo estan controlados), aplicado a las probabilidades fisicas, SIN called
   strike. Los pesos propios de Stuff+ se conservan como columna de sensibilidad
   (stuff_plus_own_learned).

5. Rama de contacto (26/09/2026, ultima modificacion de arquitectura antes del Bloque
   C). Componente continuo contact_value = P(BIP | X) x E[valor realizado | BIP, X], en
   runs desde la perspectiva del pitcher: monetiza directamente el contacto en vez de
   sumar HR/hard-hit/GB, que se traslapan. hard_hit, HR, weak y groundball quedan
   como columnas explicativas. Fouls fuera: ya se valoran via el cambio de conteo.
   Neutralidad: Stuff+ (y Location+) usan el RE general y no ven altitude_category,
   asi que su valor de contacto no cambia con el entorno por construccion. Pitching+
   usa el RE de su entorno (shrinkage hacia el general, k = 500 pitches por estado,
   solo historial) y tiene altitude_category como feature: el mismo contacto puede
   costar distinto en Extreme. Ablacion en la misma corrida: pesos sin la rama
   (columnas *_sin_contact_value), con el mismo valor objetivo. Chequeo de coherencia
   ambiental: contrafactual No -> Extreme del contact_value de Pitching+.
   --sin-rama-contacto reproduce la configuracion de la seccion 33 con prefijo 18A.

Requiere 17_pitch_sequence.parquet (correr 17 primero).
"""

import importlib.util
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, BASE_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # para que joblib pueda serializar ContactValueModel
    spec.loader.exec_module(mod)
    return mod


t16 = _load("three_scores", "16_three_scores.py")
t17 = _load("sequence_and_contact", "17_sequence_and_contact.py")

SEQ_COLS = ["prev_pitch_type", "out_within_PA", "hard_hit", "weak_contact", "hr_given_contact",
            "next_b", "next_s", "next_o", "value_defined", "half_usable_for_re", "runs_rest_of_half"]
COMPONENTS = ["whiff_per_pitch", "whiff_given_swing", "called_strike", "chase", "groundball",
              "weak_contact", "hard_hit", "hr_given_contact", "out_within_PA"]
SIGNS = {"hard_hit": -1.0, "hr_given_contact": -1.0}
CONTEXT = ["Outs", "Inning", "prev_pitch_type"]


def flag_target(col):
    def fn(df):
        eligible = df[col].notna()
        return eligible, df[col].fillna(0).astype(int)
    return fn


def re_value(H, F):
    re = t17.fit_re(H)
    y_hist = t17.realized_value_re(H, re)
    y_fut = t17.realized_value_re(F, re)
    state = pd.Series(y_hist).groupby([H["Balls"].astype(int).to_numpy(), H["Strikes"].astype(int).to_numpy(),
                                        H["Outs"].astype(int).clip(upper=2).to_numpy()]).mean()
    keys = list(zip(F["Balls"].astype(int), F["Strikes"].astype(int), F["Outs"].astype(int).clip(upper=2)))
    null_fut = np.array([state.get(k, 0.0) for k in keys])
    return y_hist, y_fut, null_fut, re


def re_value_env(H, F):
    re_env, pool = t17.fit_re_env(H)
    y_hist = t17.realized_value_re_env(H, re_env, pool)
    y_fut = t17.realized_value_re_env(F, re_env, pool)
    key = lambda fr: [fr["altitude_category"].astype(str).to_numpy(), fr["Balls"].astype(int).to_numpy(),
                      fr["Strikes"].astype(int).to_numpy(), fr["Outs"].astype(int).clip(upper=2).to_numpy()]
    state = pd.Series(y_hist).groupby(key(H)).mean()
    null_fut = np.array([state.get(k, 0.0) for k in zip(*key(F))])
    return y_hist, y_fut, null_fut, re_env


def contact_value_target(df):
    y = df["_y_value"]
    return y.notna(), (y * df["is_ball_in_play"].eq(1)).fillna(0.0)


def prep(df_model):
    seq = pd.read_parquet(BASE_DIR / "17_pitch_sequence.parquet", columns=["PitchUID"] + SEQ_COLS)
    n = len(df_model)
    out = df_model.drop(columns=[c for c in SEQ_COLS if c in df_model]).merge(seq, on="PitchUID", how="left", validate="one_to_one")
    if len(out) != n:
        raise ValueError("El merge con 17 cambio el numero de filas.")
    out["prev_pitch_type"] = out["prev_pitch_type"].astype(object)
    out["value_defined"] = out["value_defined"].fillna(False).astype(bool)
    out["half_usable_for_re"] = out["half_usable_for_re"].fillna(False).astype(bool)
    return out


def build_cfg(mods):
    fam = t16.FAMILIES
    # Orden: pitching antes que stuff, porque Stuff+ toma sus pesos (ver docstring).
    families = {
        "location": fam["location"],
        "pitching": {k: v + CONTEXT for k, v in fam["pitching"].items()},
        "stuff": fam["stuff"],
    }
    targets = {n: fn for n, _f, fn in mods["boosting"].TARGETS}
    for c in ["weak_contact", "hard_hit", "hr_given_contact", "out_within_PA"]:
        targets[c] = flag_target(c)
    cfg = {"families": families, "components": COMPONENTS, "signs": SIGNS, "targets": targets,
           "categorical": t16.CATEGORICAL + ["prev_pitch_type"], "value": re_value,
           "weight_override": {"stuff": {"from": "pitching", "exclude": ["called_strike"]}}}
    if not NO_CONTACT_BRANCH:
        targets["contact_value"] = contact_value_target
        cfg.update({"components": COMPONENTS + ["contact_value"],
                    "builders": {"contact_value": t16.make_contact_builder(mods["boosting"])},
                    "value_env": re_value_env, "family_value": {"pitching": "env"},
                    "ablation": "contact_value"})
    return cfg


def env_coherence(bundle, prefix="18", window="2024-2025 -> 2026"):
    """Mismos pitches observados en No Altitude: contact_value de Pitching+ con la etiqueta No vs Extreme.
    Stuff+ no tiene altitude_category entre sus features (neutral por construccion)."""
    e = bundle[window]
    fam = e["families"]["pitching"]
    assert all("altitude_category" not in cols for cols in e["families"]["stuff"]["variants"].values())
    F = e["future_frame"]
    m = F["altitude_category"].astype(str).eq("No Altitude").to_numpy()
    X = F[m].copy()
    ars = e["f_ars"][m]
    out = {}
    for lvl in ["No Altitude", "Extreme Altitude"]:
        X["altitude_category"] = pd.Categorical([lvl] * len(X), categories=F["altitude_category"].cat.categories)
        out[lvl] = t16.family_predict(fam["models"]["contact_value"], X, ars, fam["variants"])
        out["gb_" + lvl] = t16.family_predict(fam["models"]["groundball"], X, ars, fam["variants"])
    d = pd.DataFrame({"AutoPitchType": X["AutoPitchType"].astype(str).to_numpy(),
                      "delta_runs_x100": 100 * (out["Extreme Altitude"] - out["No Altitude"]),
                      "p_gb": out["gb_No Altitude"]})
    d["tercil_propension_gb"] = pd.qcut(d["p_gb"], 3, labels=["GB bajo", "GB medio", "GB alto"])
    by_type = d.groupby("AutoPitchType")["delta_runs_x100"].agg(["count", "mean"]).reset_index().assign(grupo="pitch_type")
    by_gb = d.groupby("tercil_propension_gb", observed=True)["delta_runs_x100"].agg(["count", "mean"]).reset_index().assign(grupo="propension_gb")
    res = pd.concat([by_type.rename(columns={"AutoPitchType": "nivel"}), by_gb.rename(columns={"tercil_propension_gb": "nivel"})])
    res.to_csv(BASE_DIR / f"{prefix}_contact_env_coherence.csv", index=False)
    print("\nCoherencia ambiental: contact_value de Pitching+, mismo pitch con etiqueta Extreme - No "
          "(runs por 100 pitches, perspectiva del pitcher; negativo = mas dano esperado en altitud):")
    print(res.round(3).to_string(index=False))


def pitcher_level_check(prefix="18", min_pitches=150, B=1000, seed=0):
    """Exploratorio (la version formal es el Bloque C): la media por pitcher de cada score
    en 2025 vs el valor realizado por pitch del mismo pitcher en 2026, con IC bootstrap por
    pitcher, frente al propio valor realizado de 2025 como predictor."""
    s = pd.read_csv(BASE_DIR / f"{prefix}_three_scores.csv")
    a, b = s[s.window == "2024 -> 2025"], s[s.window == "2024-2025 -> 2026"]
    cols = ["realized_value_pitcher"] + [c for c in s.columns if c.endswith(("_plus", "_v1_equal", "_own_learned", "_sin_contact_value", "_con_contacto"))]
    na, nb = a.groupby("pitcher_anon_id").size(), b.groupby("pitcher_anon_id").size()
    keep = na[na >= min_pitches].index.intersection(nb[nb >= min_pitches].index)
    M = a[a.pitcher_anon_id.isin(keep)].groupby("pitcher_anon_id")[cols].mean()
    y = b[b.pitcher_anon_id.isin(keep)].groupby("pitcher_anon_id")["realized_value_pitcher"].mean().loc[M.index]
    h = a[a.pitcher_anon_id.isin(keep)].assign(half=lambda x: x.groupby("pitcher_anon_id").cumcount() % 2)
    rng = np.random.default_rng(seed)
    rows = []
    for c in cols:
        pt = np.corrcoef(M[c], y)[0, 1]
        idx = [rng.integers(0, len(M), len(M)) for _ in range(B)]
        bs = [np.corrcoef(M[c].to_numpy()[i], y.to_numpy()[i])[0, 1] for i in idx]
        hh = h.groupby(["pitcher_anon_id", "half"])[c].mean().unstack()
        r = np.corrcoef(hh[0], hh[1])[0, 1]
        rows.append({"predictor_2025": c, "n_pitchers": len(M), "corr_valor_2026": pt,
                     "ci95_lo": np.percentile(bs, 2.5), "ci95_hi": np.percentile(bs, 97.5),
                     "confiabilidad_split_half_2025": 2 * r / (1 + r)})
    out = pd.DataFrame(rows)
    out.to_csv(BASE_DIR / f"{prefix}_pitcher_level_check.csv", index=False)
    print(f"\nChequeo a nivel pitcher (>= {min_pitches} pitches en 2025 y 2026; exploratorio):")
    print(out.round(3).to_string(index=False))


def finalize_official(prefix="18"):
    """Criterio fijado antes de la corrida: la rama de contacto entra al score oficial solo si
    mejora el valor fuera de muestra. No mejoro (guia 34), asi que el score oficial es el de
    pesos sin la rama (A'), y la version con rama queda como *_con_contacto. contact_value
    sigue disponible como columna explicativa (p_contact_value, runs por pitch)."""
    s = pd.read_csv(BASE_DIR / f"{prefix}_three_scores.csv")
    if "stuff_plus_con_contacto" in s:
        return
    ren = {}
    for fam in ["stuff", "location", "pitching"]:
        ren[f"{fam}_plus"] = f"{fam}_plus_con_contacto"
        ren[f"{fam}_plus_sin_contact_value"] = f"{fam}_plus"
    s.rename(columns=ren).to_csv(BASE_DIR / f"{prefix}_three_scores.csv", index=False)
    w = pd.read_csv(BASE_DIR / f"{prefix}_weights.csv")
    w["weight_oficial"] = w["weight_sin_ablacion"]
    w.to_csv(BASE_DIR / f"{prefix}_weights.csv", index=False)
    b = joblib.load(BASE_DIR / f"{prefix}_three_scores_bundle.pkl")
    for entry in b.values():
        for f in entry["families"].values():
            f["weights_con_contacto"], f["weights_raw_con_contacto"] = f["weights"], f["weights_raw"]
            f["weights"], f["weights_raw"] = f["weights_abl"], f["weights_raw_abl"]
        entry["official_scheme"] = "sin rama de contacto (A')"
    joblib.dump(b, BASE_DIR / f"{prefix}_three_scores_bundle.pkl")
    print("Score oficial = pesos sin rama de contacto (A'); version con rama en *_con_contacto.")


NO_CONTACT_BRANCH = "--sin-rama-contacto" in sys.argv

if __name__ == "__main__":
    if NO_CONTACT_BRANCH:
        sys.argv.remove("--sin-rama-contacto")
    prefix = "18A" if NO_CONTACT_BRANCH else "18"
    _scores, bundle = t16.run(prefix=prefix, cfg_fn=build_cfg, prep_fn=prep)
    if "--sample-pitchers" not in " ".join(sys.argv):
        if not NO_CONTACT_BRANCH:
            finalize_official(prefix)
            env_coherence(bundle, prefix)
        pitcher_level_check(prefix)
    elif not NO_CONTACT_BRANCH:
        env_coherence(bundle, prefix + "_SMOKE")
