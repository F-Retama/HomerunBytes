"""Stuff+ / Location+ / Pitching+ separados (Bloque A de la correccion post-auditoria).

Motivo (auditoria 25/09/2026, guia seccion 32): las 12 features "fisicas" de
03/07/09 incluian VertRelAngle, HorzRelAngle, VertApprAngle y HorzApprAngle, que
junto con movimiento y release reconstruyen la ubicacion en el plato (R2 ~0.997
en pitchers no vistos). Por eso el "Stuff+ puro" no era puro y el score de 09,
llamado Stuff+, era en realidad un Pitching+.

Tres scores por pitch, cada uno en escala 100 + 10z:
  - Stuff+    : solo fisica en release (brief): RelSpeed, SpinRate, SpinAxis,
                RelHeight, RelSide, Extension, InducedVertBreak, HorzBreak,
                AutoPitchType, PitcherThrows; + arsenal historico cuando existe.
                SIN angulos de release/aproximacion ni ubicacion.
  - Location+ : PlateLocHeight, PlateLocSide, Balls, Strikes, BatterSide,
                AutoPitchType (cluster), PitcherThrows. Sin fisica.
  - Pitching+ : fisica de Stuff+ + ubicacion + count + BatterSide +
                altitude_category + arsenal. Sin angulos: con la ubicacion
                explicita aportan ~0.0006 AUC (ablacion de la auditoria) y
                meterlos reintroduce la contradiccion angulo/ubicacion en los
                contrafactuales de altitud.
Pitching+ usa las features crudas en un solo modelo (no stacking de las
salidas de Stuff+/Location+): misma informacion, sin requerir out-of-fold
en cascada para evitar fuga.

Horizontales en convencion de diestro: HorzBreak, RelSide y hb_diff_vs_fb se
multiplican por -1 en zurdos y SpinAxis -> 360 - SpinAxis (verificado: Four-Seam
zurdo 151 -> 209 vs diestro 210). Lateralidad no Right/Left se deja sin espejar
(PitcherThrows sigue como feature).

Cobertura (degradacion): cada familia con arsenal tiene dos variantes. Si el pitch
tiene las 7 features de arsenal historico -> variante 'arsenal'; si no (pitcher
nuevo o tipo sin historial) -> variante 'base' sin arsenal. Ya no se excluye a
nadie; information_level lo indica.

Componentes: los mismos 5 targets V1 de 05/07 (whiff por pitch, whiff|swing,
groundball|contacto, called strike, chase|fuera de zona), entrenados cada uno
en su poblacion condicional y aplicados a todos los pitches. En Stuff+ esto no
es extrapolacion (no hay ubicacion entre las features). En Location+/Pitching+
chase fuera de zona sigue siendo extrapolacion para pitches en zona (27.1).

Pesos (post-auditoria): V1 (pesos iguales) queda como sensibilidad porque 13
mostro que V2 lo supera fuera de muestra en ambas direcciones. Pesos oficiales:
no negativos + ridge leve (funciones de 13) contra el valor realizado por conteo
(cadena de Markov de 13 aprendida con el historial), usando predicciones
OUT-OF-FOLD del historial (GroupKFold por pitcher). Nada de la temporada
evaluada entra al entrenamiento, a los pesos ni a la estandarizacion.

Validacion en la temporada evaluada (season-forward): AUC/LogLoss/Brier por
componente (total, pitchers nuevos = holdout de pitcher, pitchers con historial),
calibracion global por altitude_category y RMSE del compuesto contra el nulo por
conteo. La suite completa (GroupKFold, park holdout, reliability, estabilidad por
pitcher) es el Bloque C.
"""

import argparse
import importlib.util
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss, log_loss, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

BASE_DIR = Path(__file__).resolve().parent
PHYS_NUM = ["RelSpeed", "SpinRate", "SpinAxis", "RelHeight", "RelSide", "Extension",
            "InducedVertBreak", "HorzBreak"]
PHYS_CAT = ["AutoPitchType", "PitcherThrows"]
ARSENAL = ["velo_diff_vs_fb", "ivb_diff_vs_fb", "hb_diff_vs_fb", "spin_diff_vs_fb",
           "pitch_usage_pct", "release_distance_vs_fb", "extension_diff_vs_fb"]
LOC_NUM = ["PlateLocHeight", "PlateLocSide", "Balls", "Strikes"]
LOC_CAT = ["AutoPitchType", "PitcherThrows", "BatterSide"]
DROPPED_ANGLES = ["VertRelAngle", "HorzRelAngle", "VertApprAngle", "HorzApprAngle"]
FAMILIES = {
    "stuff": {"base": PHYS_NUM + PHYS_CAT, "arsenal": PHYS_NUM + ARSENAL + PHYS_CAT},
    "location": {"base": LOC_NUM + LOC_CAT},
    "pitching": {"base": PHYS_NUM + LOC_NUM + PHYS_CAT + ["BatterSide", "altitude_category"],
                 "arsenal": PHYS_NUM + LOC_NUM + ARSENAL + PHYS_CAT + ["BatterSide", "altitude_category"]},
}
SCORE_NAME = {"stuff": "stuff_plus", "location": "location_plus", "pitching": "pitching_plus"}
COMPONENTS = ["whiff_per_pitch", "whiff_given_swing", "groundball", "called_strike", "chase"]
CATEGORICAL = ["AutoPitchType", "PitcherThrows", "BatterSide", "altitude_category"]
WINDOWS = [([2024], 2025), ([2024, 2025], 2026)]
N_FOLDS = 2


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, BASE_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def mirror(frame):
    """Convencion de diestro para variables con signo absoluto."""
    f = frame.copy()
    left = f["PitcherThrows"].astype(str).eq("Left").to_numpy()
    for col in ["HorzBreak", "RelSide", "hb_diff_vs_fb"]:
        if col in f:
            f.loc[left, col] = -f.loc[left, col]
    f.loc[left, "SpinAxis"] = 360 - f.loc[left, "SpinAxis"]
    return f


def location_leak_check(df, cols, n=150_000, seed=0):
    """R2 de PlateLoc* a partir de un set de features, en pitchers no vistos."""
    d = df.dropna(subset=["PlateLocHeight", "PlateLocSide"]).sample(min(n, len(df)), random_state=seed)
    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=seed).split(d, groups=d["pitcher_anon_id"]))
    out = {}
    for y in ["PlateLocHeight", "PlateLocSide"]:
        m = HistGradientBoostingRegressor(max_iter=300, categorical_features="from_dtype", random_state=seed)
        m.fit(d.iloc[tr][cols], d.iloc[tr][y])
        out[y] = r2_score(d.iloc[te][y], m.predict(d.iloc[te][cols]))
    return out


class ContactValueModel:
    """Rama de contacto: P(BIP | X) x E[valor realizado | BIP, X] (runs, perspectiva del pitcher).
    Expone predict_proba para encajar en el pipeline; la columna 1 es el valor esperado."""

    def __init__(self, p_bip, reg):
        self.p_bip, self.reg = p_bip, reg

    def predict_proba(self, X):
        x = self.p_bip.predict_proba(X)[:, 1] * self.reg.predict(X)
        return np.column_stack([-x, x])


def make_contact_builder(boosting):
    def build(train, cols):
        elig = train["is_ball_in_play"].notna()
        clf = boosting.build_model().fit(train.loc[elig, cols], train.loc[elig, "is_ball_in_play"].astype(int))
        bip = train["is_ball_in_play"].eq(1) & train["_y_value"].notna()
        reg = HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=42, max_iter=300,
                                            learning_rate=0.05, early_stopping=True, n_iter_no_change=15)
        reg.fit(train.loc[bip, cols], train.loc[bip, "_y_value"])
        return ContactValueModel(clf, reg)
    return build


def fit_predict(boosting, target_fn, train, cols, score_frames, builder=None):
    if builder is not None:
        model = builder(train, cols)
        return model, [model.predict_proba(f[cols])[:, 1] if len(f) else np.array([]) for f in score_frames]
    elig, event = target_fn(train)
    model = boosting.build_model().fit(train.loc[elig, cols], event.loc[elig])
    return model, [model.predict_proba(f[cols])[:, 1] if len(f) else np.array([]) for f in score_frames]


def family_predict(models, frame, has_ars, variants):
    """Prediccion por fila con la variante que corresponde a su nivel de informacion."""
    p = np.full(len(frame), np.nan)
    if "arsenal" in variants and has_ars.any():
        p[has_ars] = models["arsenal"].predict_proba(frame.loc[has_ars, variants["arsenal"]])[:, 1]
    rest = ~has_ars if "arsenal" in variants else np.ones(len(frame), bool)
    if rest.any():
        p[rest] = models["base"].predict_proba(frame.loc[rest, variants["base"]])[:, 1]
    return p


def pitcher_stats(codes, Z, y, n_p):
    """Estadisticos suficientes por pitcher (igual que 13, para K componentes)."""
    K = Z.shape[1]
    G = np.zeros((n_p, K, K)); c = np.zeros((n_p, K))
    np.add.at(G, codes, Z[:, :, None] * Z[:, None, :])
    np.add.at(c, codes, Z * y[:, None])
    return G, c, np.bincount(codes, minlength=n_p).astype(float)


def component_metrics(y, p):
    if len(y) and not np.isin(np.unique(y), [0, 1]).all():
        # Componente continuo (rama de contacto, runs): RMSE y R2 contra la media.
        return {"n": len(y), "auc": np.nan, "log_loss": np.nan, "brier": np.nan, "mean_pred": float(np.mean(p)),
                "obs_rate": float(np.mean(y)), "rmse": float(np.sqrt(np.mean((y - p) ** 2))), "r2": r2_score(y, p)}
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {"n": len(y), "auc": np.nan, "log_loss": np.nan, "brier": np.nan, "mean_pred": np.nan, "obs_rate": np.nan}
    return {"n": len(y), "auc": roc_auc_score(y, p), "log_loss": log_loss(y, p), "brier": brier_score_loss(y, p),
            "mean_pred": float(np.mean(p)), "obs_rate": float(np.mean(y))}


def count13_value(cv):
    """Valor realizado de 13 (pagos de The Book); nulo por conteo = 0 por construccion."""
    def value(H, F):
        V = cv.fit_count_value(cv.classify(H))
        y_fut = cv.realized_value(cv.classify(F), V)
        return cv.realized_value(cv.classify(H), V), y_fut, np.zeros(len(F)), V
    return value


def default_cfg(mods):
    return {"families": FAMILIES, "components": COMPONENTS, "signs": {},
            "targets": {n: fn for n, _f, fn in mods["boosting"].TARGETS},
            "categorical": CATEGORICAL, "value": count13_value(mods["cv"])}


def run_window(df_model, years, eval_year, mods, n_boot, seed, cfg=None):
    arsenal, six, boosting, cv = mods["arsenal"], mods["six"], mods["boosting"], mods["cv"]
    cfg = cfg or default_cfg(mods)
    comps_all = cfg["components"]
    K = len(comps_all)
    label = f"{'-'.join(map(str, years))} -> {eval_year}"
    history = df_model[df_model["year"].isin(years)].copy()
    future = df_model[df_model["year"] == eval_year].copy()
    fb, usage = arsenal.fit_historical_arsenal_reference(history)
    arsenal.audit_historical_reference(history, fb, usage)
    H = mirror(six.apply_reference_in_sample(arsenal, history, fb, usage)).reset_index(drop=True)
    F = mirror(arsenal.apply_historical_arsenal_reference(future, fb, usage)).reset_index(drop=True)
    for col in cfg["categorical"]:
        cats = df_model[col].cat.categories
        H[col] = pd.Categorical(H[col].astype(object), categories=cats)
        F[col] = pd.Categorical(F[col].astype(object), categories=cats)
    h_ars = H[ARSENAL].notna().all(axis=1).to_numpy()
    f_ars = F[ARSENAL].notna().all(axis=1).to_numpy()
    new_pitcher = ~F["pitcher_anon_id"].isin(set(H["pitcher_anon_id"])).to_numpy()
    targets = cfg["targets"]
    print(f"\n{'=' * 80}\n{label} | historial {len(H):,} (arsenal completo {h_ars.mean():.1%}) | "
          f"evaluacion {len(F):,} (arsenal {f_ars.mean():.1%}, pitchers nuevos {new_pitcher.mean():.1%} de filas)\n{'=' * 80}", flush=True)

    # Valor realizado (por defecto el de 13); se aprende solo con el historial.
    y_hist, y_fut, null_fut, V = cfg["value"](H, F)
    values = {"pooled": (y_hist, y_fut, null_fut)}
    if cfg.get("value_env"):
        values["env"] = cfg["value_env"](H, F)[:3]
    builders = cfg.get("builders", {})

    groups = H["pitcher_anon_id"].to_numpy()
    folds = list(GroupKFold(n_splits=N_FOLDS).split(H, groups=groups))
    out = {"label": label, "families": {}, "V": V}
    scores = pd.DataFrame({"window": label, "PitchUID": F["PitchUID"].to_numpy(),
                           "pitcher_anon_id": F["pitcher_anon_id"].to_numpy(), "year": F["year"].to_numpy(),
                           "AutoPitchType": F["AutoPitchType"].astype(str).to_numpy(),
                           "PitcherThrows": F["PitcherThrows"].astype(str).to_numpy(),
                           "altitude_category": F["altitude_category"].astype(str).to_numpy(),
                           "information_level": np.where(f_ars, "arsenal_historico", "sin_arsenal"),
                           "new_pitcher": new_pitcher, "realized_value_pitcher": y_fut})
    if "env" in values:
        scores["realized_value_env"] = values["env"][1]
    metric_rows, calib_rows, weight_rows, perf_rows = [], [], [], []
    for fam, variants in cfg["families"].items():
        vname = cfg.get("family_value", {}).get(fam, "pooled")
        y_hist, y_fut, null_fut = values[vname]
        H["_y_value"], F["_y_value"] = y_hist, y_fut
        primary = "arsenal" if "arsenal" in variants else "base"
        models, stats = {}, {}
        Z_hist, Z_fut, Z_oof = [], [], []
        for comp in comps_all:
            fn = targets[comp]
            sign = cfg["signs"].get(comp, 1.0)
            models[comp] = {}
            for var, cols in variants.items():
                train = H[h_ars] if var == "arsenal" else H
                models[comp][var], _ = fit_predict(boosting, fn, train, cols, [], builders.get(comp))
            p_hist = family_predict(models[comp], H, h_ars, variants)
            p_fut = family_predict(models[comp], F, f_ars, variants)
            mean, std = p_hist.mean(), p_hist.std()
            stats[comp] = (mean, std)
            Z_hist.append(sign * (p_hist - mean) / std)
            Z_fut.append(sign * (p_fut - mean) / std)
            scores[f"{SCORE_NAME[fam]}__p_{comp}"] = p_fut
            # OOF sobre el historial para aprender pesos (variante principal; filas con arsenal si aplica).
            p_oof = np.full(len(H), np.nan)
            cols = variants[primary]
            for tr, te in folds:
                trm = np.zeros(len(H), bool); trm[tr] = True
                if primary == "arsenal":
                    trm &= h_ars
                m, (pte,) = fit_predict(boosting, fn, H[trm], cols, [H.iloc[te][cols]], builders.get(comp))
                p_oof[te] = pte
            Z_oof.append(sign * (p_oof - np.nanmean(p_oof)) / np.nanstd(p_oof))
            # Metricas en la temporada evaluada (poblacion condicional real del componente).
            elig, event = fn(F)
            elig = elig.to_numpy()
            for subset, mask in [("todos", elig), ("pitchers_nuevos", elig & new_pitcher),
                                 ("con_historial", elig & ~new_pitcher)]:
                metric_rows.append({"window": label, "family": fam, "component": comp, "subset": subset,
                                    **component_metrics(event.to_numpy()[mask], p_fut[mask])})
            for alt in ["No Altitude", "Medium Altitude", "Extreme Altitude"]:
                mask = elig & (F["altitude_category"].astype(str).to_numpy() == alt)
                if mask.sum():
                    calib_rows.append({"window": label, "family": fam, "component": comp, "altitude_category": alt,
                                       "n": int(mask.sum()), "mean_pred": float(p_fut[mask].mean()),
                                       "obs_rate": float(event.to_numpy()[mask].mean()),
                                       "calib_in_large": float(p_fut[mask].mean() - event.to_numpy()[mask].mean())})
            key = "r2" if comp in builders else "auc"
            print(f"  {fam:9s} {comp:18s} {key.upper()} eval={metric_rows[-3][key]:.4f} "
                  f"(nuevos {metric_rows[-2][key]:.4f}, historial {metric_rows[-1][key]:.4f})", flush=True)
        Zh, Zf, Zo = (np.column_stack(z) for z in (Z_hist, Z_fut, Z_oof))

        # Pesos contra valor realizado, con predicciones OOF del historial.
        ok = ~np.isnan(y_hist) & ~np.isnan(Zo).any(axis=1)
        if primary == "arsenal":
            ok &= h_ars
        pit = pd.Index(pd.unique(groups[ok])); codes = pit.get_indexer(groups[ok])
        zbar, ybar = Zo[ok].mean(axis=0), y_hist[ok].mean()
        G, c, _ = pitcher_stats(codes, Zo[ok] - zbar, y_hist[ok] - ybar, len(pit))
        w_raw = cv.nn_ridge(G.sum(0), c.sum(0))
        if w_raw.sum() <= 0:
            raise ValueError(f"{fam}: pesos aprendidos todos en cero.")
        w = w_raw / w_raw.sum()
        e = np.full(K, 1.0 / K)
        # Override opcional: tomar los pesos (valor por evento) de otra familia ya calculada.
        ov = cfg.get("weight_override", {}).get(fam)
        own_w, own_w_raw = w.copy(), w_raw.copy()
        if ov:
            src = out["families"][ov["from"]]
            w_raw = src["weights_raw"].copy()
            w_raw[[c_ in ov["exclude"] for c_ in comps_all]] = 0.0
            w = w_raw / w_raw.sum()
            # Direccion prestada; magnitud reescalada con el OOF de esta familia (como a1 en V1).
            w_raw = w * (w @ c.sum(0)) / (w @ G.sum(0) @ w)
        # Ablacion opcional en la misma corrida: pesos sin un componente (p.ej. la rama de contacto).
        ab = cfg.get("ablation")
        w_abl = w_abl_raw = None
        if ab:
            if ov:
                wr = out["families"][ov["from"]]["weights_raw_abl"].copy()
                wr[[c_ in ov["exclude"] for c_ in comps_all]] = 0.0
                w_abl = wr / wr.sum()
                w_abl_raw = w_abl * (w_abl @ c.sum(0)) / (w_abl @ G.sum(0) @ w_abl)
            else:
                idx = np.array([i for i, c_ in enumerate(comps_all) if c_ != ab])
                w_abl_raw = np.zeros(K)
                w_abl_raw[idx] = cv.nn_ridge(G.sum(0)[np.ix_(idx, idx)], c.sum(0)[idx])
                w_abl = w_abl_raw / w_abl_raw.sum()
        a1 = (e @ c.sum(0)) / (e @ G.sum(0) @ e)
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(n_boot):
            mult = np.bincount(rng.integers(0, len(pit), len(pit)), minlength=len(pit)).astype(float)
            wb = cv.nn_ridge(np.tensordot(mult, G, 1), mult @ c)
            draws.append(wb / wb.sum() if wb.sum() > 0 else np.full(K, np.nan))
        lo, hi = np.nanpercentile(draws, [2.5, 97.5], axis=0)
        for k, comp in enumerate(comps_all):
            weight_rows.append({"window": label, "family": fam, "component": comp, "weight_v1": 1.0 / K, "sign": cfg["signs"].get(comp, 1.0),
                                "weight_learned": own_w[k], "ci95_lo": lo[k], "ci95_hi": hi[k], "weight_used": w[k],
                                "weight_source": f"{ov['from']} sin {','.join(ov['exclude'])}" if ov else "propios",
                                "weight_sin_ablacion": w_abl[k] if ab else np.nan, "valor_objetivo": vname})

        # Escala 100+10z contra la distribucion del historial (nunca la temporada evaluada).
        schemes = ([("", w), ("_v1_equal", e)] + ([("_own_learned", own_w)] if ov else [])
                   + ([(f"_sin_{ab}", w_abl)] if ab else []))
        for scheme, wv in schemes:
            raw_h, raw_f = Zh @ wv, Zf @ wv
            scores[SCORE_NAME[fam] + scheme] = 100 + 10 * (raw_f - raw_h.mean()) / raw_h.std()
            stats["scale" + scheme] = (raw_h.mean(), raw_h.std())

        # RMSE fuera de muestra contra el nulo por conteo (E[y|conteo]=0 por construccion).
        okf = ~np.isnan(y_fut)
        fp = pd.Index(pd.unique(scores["pitcher_anon_id"][okf])); fcodes = fp.get_indexer(scores["pitcher_anon_id"][okf])
        nul = null_fut[okf]
        preds = {"nulo_por_conteo": nul, "V1_pesos_iguales": nul + a1 * (Zf[okf] - zbar) @ e,
                 "pesos_aprendidos": nul + (Zf[okf] - zbar) @ w_raw}
        if ov:
            preds["pesos_propios_de_la_familia"] = nul + (Zf[okf] - zbar) @ own_w_raw
        if ab:
            preds[f"pesos_sin_{ab}"] = nul + (Zf[okf] - zbar) @ w_abl_raw
        sse = {k: np.bincount(fcodes, weights=(y_fut[okf] - p) ** 2, minlength=len(fp)) for k, p in preds.items()}
        base = sse["nulo_por_conteo"]
        for k in preds:
            gains = []
            for _ in range(n_boot):
                m = np.bincount(rng.integers(0, len(fp), len(fp)), minlength=len(fp))
                gains.append(1 - (m @ sse[k]) / (m @ base))
            ci = np.percentile(gains, [2.5, 97.5]) if k != "nulo_por_conteo" else (0.0, 0.0)
            perf_rows.append({"window": label, "family": fam, "model": k, "valor_objetivo": vname,
                              "rmse": float(np.sqrt(sse[k].sum() / okf.sum())),
                              "mse_reduction_vs_null": float(1 - sse[k].sum() / base.sum()),
                              "ci95_lo": ci[0], "ci95_hi": ci[1]})
        print(f"  -> {fam}: pesos aprendidos " + ", ".join(f"{c_}={v:.3f}" for c_, v in zip(comps_all, w))
              + f" | reduccion MSE vs nulo: " + ", ".join(f"{r_['model']} {r_['mse_reduction_vs_null']:.4%}" for r_ in perf_rows if r_['family'] == fam and r_['window'] == label and r_['model'] != 'nulo_por_conteo'), flush=True)
        out["families"][fam] = {"variants": variants, "models": models, "stats": stats,
                                "weights": w, "weights_raw": w_raw, "zbar_oof": zbar,
                                "own_weights": own_w, "weight_override": ov,
                                "weights_abl": w_abl, "weights_raw_abl": w_abl_raw, "value": vname}
    out["future_frame"] = F
    out["f_ars"] = f_ars
    return scores, pd.DataFrame(metric_rows), pd.DataFrame(calib_rows), pd.DataFrame(weight_rows), pd.DataFrame(perf_rows), out


def main():
    run(prefix="16")


def load_modules():
    return {"arsenal": _load("fold_safe_arsenal", "04_fold_safe_arsenal.py"),
            "six": _load("arsenal_incremental_value", "06_arsenal_incremental_value.py"),
            "boosting": _load("boosting_models", "07_boosting_models.py"),
            "cv": _load("count_value", "13_count_value_weights.py")}


def run(prefix, cfg_fn=None, prep_fn=None):
    """prefix: prefijo de archivos; cfg_fn(mods) -> cfg; prep_fn(df_model) -> df_model (antes de categorizar)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-pitchers", type=float, default=1.0, help="fraccion de pitchers (solo smoke test)")
    args = ap.parse_args()
    pd.set_option("display.width", 220)
    mods = load_modules()
    cfg = cfg_fn(mods) if cfg_fn else default_cfg(mods)
    df = pd.read_parquet(BASE_DIR / "stuff_model_features_v1.parquet")
    df_model = df.dropna(subset=["is_whiff", "pitcher_anon_id"]).copy()
    if args.sample_pitchers < 1.0:
        keep = pd.Series(df_model["pitcher_anon_id"].unique()).sample(frac=args.sample_pitchers, random_state=0)
        df_model = df_model[df_model["pitcher_anon_id"].isin(set(keep))].copy()
        prefix = prefix + "_SMOKE"
    if prep_fn:
        df_model = prep_fn(df_model)
    for col in cfg["categorical"]:
        df_model[col] = df_model[col].astype("category")

    print("Chequeo de fuga de ubicacion (R2 de PlateLoc* en pitchers no vistos):")
    m = mirror(df_model)
    leak_rows = []
    for name, cols in [("stuff_v1 (con angulos, 03/07/09)", PHYS_NUM + DROPPED_ANGLES + PHYS_CAT),
                       ("stuff_plus (este script)", PHYS_NUM + PHYS_CAT)]:
        r = location_leak_check(m, cols)
        leak_rows.append({"feature_set": name, **{f"r2_{k}": v for k, v in r.items()}})
        print(f"  {name:36s} R2 altura {r['PlateLocHeight']:.3f} | R2 lado {r['PlateLocSide']:.3f}", flush=True)

    all_scores, all_m, all_c, all_w, all_p, bundle = [], [], [], [], [], {}
    for years, eval_year in WINDOWS:
        s, mt, cb, wt, pf, entry = run_window(df_model, years, eval_year, mods, args.bootstrap, args.seed, cfg)
        all_scores.append(s); all_m.append(mt); all_c.append(cb); all_w.append(wt); all_p.append(pf)
        bundle[entry["label"]] = entry
    scores = pd.concat(all_scores, ignore_index=True)
    metrics = pd.concat(all_m, ignore_index=True)
    weights = pd.concat(all_w, ignore_index=True)
    perf = pd.concat(all_p, ignore_index=True)
    calib = pd.concat(all_c, ignore_index=True)

    scores.to_csv(BASE_DIR / f"{prefix}_three_scores.csv", index=False)
    metrics.to_csv(BASE_DIR / f"{prefix}_component_metrics.csv", index=False)
    calib.to_csv(BASE_DIR / f"{prefix}_calibration_by_altitude.csv", index=False)
    weights.to_csv(BASE_DIR / f"{prefix}_weights.csv", index=False)
    perf.to_csv(BASE_DIR / f"{prefix}_out_of_sample_value.csv", index=False)
    pd.DataFrame(leak_rows).to_csv(BASE_DIR / f"{prefix}_location_leak_check.csv", index=False)
    joblib.dump(bundle, BASE_DIR / f"{prefix}_three_scores_bundle.pkl")
    run.last_prefix = prefix

    f2 = lambda x: f"{x:.3f}"
    print(f"\n{'=' * 80}\nRESUMEN\n{'=' * 80}")
    print("AUC por familia y componente (temporada evaluada):")
    print(metrics[metrics.subset == "todos"].pivot_table(index=["family", "component"], columns="window", values="auc").to_string(float_format=f2))
    print("\nAUC en pitchers nuevos (holdout de pitcher dentro de la temporada futura):")
    print(metrics[metrics.subset == "pitchers_nuevos"].pivot_table(index=["family", "component"], columns="window", values="auc").to_string(float_format=f2))
    print("\nPesos aprendidos (OOF del historial, normalizados):")
    print(weights.pivot_table(index=["family", "component"], columns="window", values="weight_learned").to_string(float_format=f2))
    print("\nReduccion de MSE del valor realizado vs nulo por conteo:")
    print(perf.pivot_table(index=["family", "model"], columns="window", values="mse_reduction_vs_null").to_string(float_format=lambda x: f"{x:.4%}"))
    print("\nCalibracion global (media predicha - tasa observada), puntos porcentuales:")
    print((calib.pivot_table(index=["family", "component"], columns=["window", "altitude_category"], values="calib_in_large") * 100).to_string(float_format=lambda x: f"{x:+.2f}"))
    prim = scores[scores.window == "2024-2025 -> 2026"]
    print("\nMedias por pitch type (ventana 2024-2025 -> 2026):")
    print(prim.groupby("AutoPitchType")[["stuff_plus", "location_plus", "pitching_plus"]].mean().sort_values("stuff_plus", ascending=False).to_string(float_format=lambda x: f"{x:.1f}"))
    print("\nCobertura por nivel de informacion:")
    print(scores.groupby(["window", "information_level"]).size().to_string())
    print("\nCorrelacion entre scores (ventana primaria):")
    print(prim[["stuff_plus", "location_plus", "pitching_plus"]].corr().round(3).to_string())
    print(f"\nArchivos con prefijo {prefix}_: three_scores.csv, component_metrics.csv, calibration_by_altitude.csv, "
          "weights.csv, out_of_sample_value.csv, location_leak_check.csv, three_scores_bundle.pkl")
    return scores, bundle


if __name__ == "__main__":
    main()
