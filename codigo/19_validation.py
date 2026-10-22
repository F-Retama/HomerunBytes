"""Bloque C: validacion formal de la arquitectura congelada (18, score oficial A').

Regla: durante la validacion no se cambian features, targets, pesos ni arquitectura.
Los fallos se documentan como hallazgos (guia 35).

Etapas (python 19_validation.py --stage X; 'all' corre todas en orden):
  cv          Validacion cruzada por pitcher (GroupKFold 5, las tres temporadas).
              Variantes SIN arsenal: un pitcher no visto no tiene historial, y es la
              version que reciben prospectos. Pesos del compuesto = oficiales de 18
              (ventana 2024-25 -> 2026), congelados. Salvedad: esos 10 numeros se
              aprendieron con 2024-25, fuga menor en los folds que contienen esas
              temporadas. RE, estandarizacion y modelos se aprenden solo con el
              train de cada fold. Guarda las predicciones out-of-fold por pitch.
  park        Park holdout con la arquitectura corregida: train en dos categorias de
              altitud y test en la tercera (Extreme como resultado oficial; Medium y No
              como checks). Pitching+ sin altitude_category (categoria no vista).
  stability   Con los scores out-of-fold de 'cv': split-half dentro de temporada,
              correlacion 2024->25 y 2025->26 por pitcher y por pitcher x tipo, y
              prediccion del valor de la temporada siguiente contra el valor observado
              (Pearson, Spearman, IC bootstrap por pitcher, RMSE cruzado entre pares
              de temporadas y valor incremental de sumar el score al valor observado).
  calibration Sobre las predicciones season-forward oficiales de 18: Brier, log loss,
              calibracion global, intercepto/pendiente (logit) y deciles por temporada
              x altitud. Recalibracion sin fuga: Platt aprendido con 2025 (modelo de la
              ventana 1) y aplicado a 2026 (ventana 2). Stuff+ y Location+ con
              calibrador global (neutrales); Pitching+ por altitud. Los scores oficiales
              no cambian: lo recalibrado son las probabilidades.
  summary     Tabla de salida del Bloque C.

Valor por pitch para evaluar a nivel pitcher (stability): RE LMB por conteo x outs
estimado con todas las temporadas. Es una definicion de medida, no un parametro de
los predictores.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

BASE_DIR = Path(__file__).resolve().parent
PRIMARY = "2024-2025 -> 2026"
ALTS = ["No Altitude", "Medium Altitude", "Extreme Altitude"]
FAMS = ["stuff", "location", "pitching"]
MIN_PITCHER, MIN_PITCHER_TYPE = 150, 50


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, BASE_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


t18 = _load("three_scores_v2", "18_three_scores_v2.py")
t16, t17 = t18.t16, t18.t17


def setup():
    mods = t16.load_modules()
    cfg = t18.build_cfg(mods)
    comps = [c for c in cfg["components"] if c != "contact_value"]
    w = pd.read_csv(BASE_DIR / "18_weights.csv")
    w = w[w.window == PRIMARY]
    weights = {f: w[w.family == f].set_index("component").loc[comps, "weight_oficial"].to_numpy(float) for f in FAMS}
    return mods, cfg, comps, weights


def load_data(cfg):
    df = pd.read_parquet(BASE_DIR / "stuff_model_features_v1.parquet")
    df = df.dropna(subset=["is_whiff", "pitcher_anon_id"]).copy()
    df = t18.prep(df)
    df = t16.mirror(df).reset_index(drop=True)
    for col in cfg["categorical"]:
        df[col] = df[col].astype("category")
    seq = pd.read_parquet(BASE_DIR / "17_pitch_sequence.parquet", columns=["PitchUID", "uuid_time"])
    df = df.merge(seq, on="PitchUID", how="left", validate="one_to_one")
    df["value_eval"] = t17.realized_value_re(df, t17.fit_re(df))
    return df


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def platt(p, y):
    m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
    return float(m.intercept_[0]), float(m.coef_[0, 0])


def calib_stats(y, p):
    out = t16.component_metrics(y, p)
    if len(y) > 50 and np.isin(np.unique(y), [0, 1]).all() and len(np.unique(y)) == 2:
        a, b = platt(p, y)
        out.update({"calib_intercept": a, "calib_slope": b})
    out["calib_in_large"] = out.get("mean_pred", np.nan) - out.get("obs_rate", np.nan)
    return out


def value_frames(train, test, env):
    if env:
        re_env, pool = t17.fit_re_env(train)
        y_tr, y_te = t17.realized_value_re_env(train, re_env, pool), t17.realized_value_re_env(test, re_env, pool)
        key = lambda f: [f["altitude_category"].astype(str).to_numpy(), f["Balls"].astype(int).to_numpy(),
                         f["Strikes"].astype(int).to_numpy(), f["Outs"].astype(int).clip(upper=2).to_numpy()]
    else:
        re = t17.fit_re(train)
        y_tr, y_te = t17.realized_value_re(train, re), t17.realized_value_re(test, re)
        key = lambda f: [f["Balls"].astype(int).to_numpy(), f["Strikes"].astype(int).to_numpy(),
                         f["Outs"].astype(int).clip(upper=2).to_numpy()]
    state = pd.Series(y_tr).groupby(key(train)).mean()
    null_tr = np.array([state.get(k, 0.0) for k in zip(*key(train))])
    null_te = np.array([state.get(k, 0.0) for k in zip(*key(test))])
    return y_tr, y_te, null_tr, null_te


def fit_family(mods, cfg, comps, weights, fam, cols, train, test, env_value):
    """Entrena los componentes de una familia en train, califica test y evalua el compuesto."""
    boosting = mods["boosting"]
    P_tr, P_te, comp_rows = {}, {}, []
    for comp in comps:
        fn = cfg["targets"][comp]
        _m, (ptr, pte) = t16.fit_predict(boosting, fn, train, cols, [train[cols], test[cols]])
        P_tr[comp], P_te[comp] = ptr, pte
        elig, event = fn(test)
        elig = elig.to_numpy()
        comp_rows.append({"family": fam, "component": comp, **calib_stats(event.to_numpy()[elig], pte[elig])})
    sign = np.array([cfg["signs"].get(c, 1.0) for c in comps])
    mu = np.array([P_tr[c].mean() for c in comps]); sd = np.array([P_tr[c].std() for c in comps])
    Z_tr = sign * (np.column_stack([P_tr[c] for c in comps]) - mu) / sd
    Z_te = sign * (np.column_stack([P_te[c] for c in comps]) - mu) / sd
    w = weights[fam]
    raw_tr, raw_te = Z_tr @ w, Z_te @ w
    score_te = 100 + 10 * (raw_te - raw_tr.mean()) / raw_tr.std()
    y_tr, y_te, null_tr, null_te = value_frames(train, test, env_value)
    ok_tr, ok_te = ~np.isnan(y_tr), ~np.isnan(y_te)
    x_tr = raw_tr[ok_tr] - raw_tr.mean()
    a = float(x_tr @ (y_tr[ok_tr] - null_tr[ok_tr]) / (x_tr @ x_tr))
    pred = null_te[ok_te] + a * (raw_te[ok_te] - raw_tr.mean())
    mse, mse0 = np.mean((y_te[ok_te] - pred) ** 2), np.mean((y_te[ok_te] - null_te[ok_te]) ** 2)
    comp_summary = {"family": fam, "mse_reduction_vs_null": 1 - mse / mse0,
                    "spearman_score_vs_value_minus_null": spearmanr(raw_te[ok_te], y_te[ok_te] - null_te[ok_te])[0],
                    "n_value": int(ok_te.sum())}
    return score_te, P_te, comp_rows, comp_summary


def stage_cv(args):
    mods, cfg, comps, weights = setup()
    df = load_data(cfg)
    groups = df["pitcher_anon_id"].to_numpy()
    out = df[["PitchUID", "pitcher_anon_id", "year", "AutoPitchType", "altitude_category", "uuid_time", "value_eval"]].copy()
    out["altitude_category"] = out["altitude_category"].astype(str)
    out["AutoPitchType"] = out["AutoPitchType"].astype(str)
    out["fold"] = -1
    comp_rows, sum_rows = [], []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=args.folds).split(df, groups=groups)):
        out.loc[te, "fold"] = k
        train, test = df.iloc[tr], df.iloc[te]
        for fam in FAMS:
            cols = cfg["families"][fam]["base"]
            score, P, crow, csum = fit_family(mods, cfg, comps, weights, fam, cols, train, test, env_value=(fam == "pitching"))
            out.loc[te, f"{fam}_plus"] = score
            for c in comps:
                out.loc[te, f"{fam}__p_{c}"] = P[c]
            for r in crow:
                comp_rows.append({"fold": k, **r})
            sum_rows.append({"fold": k, **csum})
            print(f"fold {k} {fam:9s} MSE red {csum['mse_reduction_vs_null']:+.4%} | AUC whiff "
                  f"{[r['auc'] for r in crow if r['component'] == 'whiff_per_pitch'][0]:.4f}", flush=True)
    out.to_parquet(BASE_DIR / "19_cv_oof.parquet", index=False)
    cm, cs = pd.DataFrame(comp_rows), pd.DataFrame(sum_rows)
    cm.to_csv(BASE_DIR / "19_cv_component_metrics.csv", index=False)
    cs.to_csv(BASE_DIR / "19_cv_composite_metrics.csv", index=False)
    agg = cm.groupby(["family", "component"])[["auc", "brier", "calib_in_large", "calib_slope"]].agg(["mean", "std"])
    print("\nCV por pitcher (media y SD entre folds):\n" + agg.round(4).to_string())
    print("\n" + cs.groupby("family")[["mse_reduction_vs_null", "spearman_score_vs_value_minus_null"]].agg(["mean", "std"]).round(5).to_string())


def stage_park(args):
    mods, cfg, comps, weights = setup()
    df = load_data(cfg)
    alt = df["altitude_category"].astype(str)
    comp_rows, sum_rows = [], []
    oof = pd.read_parquet(BASE_DIR / "19_cv_oof.parquet") if (BASE_DIR / "19_cv_oof.parquet").exists() else None
    for train_alts, test_alt in [(["No Altitude", "Medium Altitude"], "Extreme Altitude"),
                                 (["No Altitude", "Extreme Altitude"], "Medium Altitude"),
                                 (["Medium Altitude", "Extreme Altitude"], "No Altitude")]:
        train, test = df[alt.isin(train_alts)], df[alt.eq(test_alt)]
        label = f"{'+'.join(a.split()[0] for a in train_alts)} -> {test_alt.split()[0]}"
        for fam in FAMS:
            cols = [c for c in cfg["families"][fam]["base"] if c != "altitude_category"]
            score, P, crow, csum = fit_family(mods, cfg, comps, weights, fam, cols, train, test, env_value=False)
            for r in crow:
                ref = np.nan
                if oof is not None:
                    fn = cfg["targets"][r["component"]]
                    elig, event = fn(test)
                    o = oof.set_index("PitchUID").loc[test["PitchUID"], f"{fam}__p_{r['component']}"].to_numpy()
                    e = elig.to_numpy()
                    ref = t16.component_metrics(event.to_numpy()[e], o[e])["auc"]
                comp_rows.append({"direccion": label, **r, "auc_cv_por_pitcher_mismas_filas": ref})
            sum_rows.append({"direccion": label, **csum})
            print(f"{label} {fam:9s} MSE red {csum['mse_reduction_vs_null']:+.4%}", flush=True)
    cm, cs = pd.DataFrame(comp_rows), pd.DataFrame(sum_rows)
    cm.to_csv(BASE_DIR / "19_park_component_metrics.csv", index=False)
    cs.to_csv(BASE_DIR / "19_park_composite_metrics.csv", index=False)
    show = cm[["direccion", "family", "component", "auc", "auc_cv_por_pitcher_mismas_filas", "calib_in_large", "calib_slope"]]
    print("\nPark holdout:\n" + show.round(4).to_string(index=False))
    print("\n" + cs.round(5).to_string(index=False))


def boot_corr(x, y, B, rng):
    pt_p, pt_s = pearsonr(x, y)[0], spearmanr(x, y)[0]
    idx = [rng.integers(0, len(x), len(x)) for _ in range(B)]
    bp = [pearsonr(x[i], y[i])[0] for i in idx]
    bs = [spearmanr(x[i], y[i])[0] for i in idx]
    return pt_p, *np.percentile(bp, [2.5, 97.5]), pt_s, *np.percentile(bs, [2.5, 97.5])


def split_half(frame, keys, col):
    f = frame.sort_values(keys + ["uuid_time"]).copy()
    f["half"] = f.groupby(keys).cumcount() % 2
    h = f.groupby(keys + ["half"])[col].mean().unstack()
    r = np.corrcoef(h[0], h[1])[0, 1]
    return 2 * r / (1 + r), len(h)


def stage_stability(args):
    rng = np.random.default_rng(args.seed)
    o = pd.read_parquet(BASE_DIR / "19_cv_oof.parquet")
    cols = ["value_eval", "stuff_plus", "location_plus", "pitching_plus"]
    rows = []
    for level, keys, mn in [("pitcher", ["pitcher_anon_id"], MIN_PITCHER), ("pitcher_x_tipo", ["pitcher_anon_id", "AutoPitchType"], MIN_PITCHER_TYPE)]:
        for year in [2024, 2025, 2026]:
            f = o[o.year == year]
            n = f.groupby(keys).size()
            f = f.set_index(keys).loc[n[n >= mn].index].reset_index()
            for c in cols:
                rel, ng = split_half(f, keys, c)
                rows.append({"nivel": level, "prueba": f"split-half {year}", "score": c, "n_grupos": ng, "valor": rel})
        g = o.groupby(keys + ["year"]).agg(n=("PitchUID", "size"), **{c: (c, "mean") for c in cols}).reset_index()
        g = g[g.n >= mn]
        for y0, y1 in [(2024, 2025), (2025, 2026)]:
            a = g[g.year == y0].set_index(keys); b = g[g.year == y1].set_index(keys)
            ix = a.index.intersection(b.index)
            for c in cols:
                r = boot_corr(a.loc[ix, c].to_numpy(), b.loc[ix, c].to_numpy(), args.boot, rng)
                rows.append({"nivel": level, "prueba": f"año a año {y0}->{y1}", "score": c, "n_grupos": len(ix),
                             "valor": r[0], "ci95_lo": r[1], "ci95_hi": r[2], "spearman": r[3]})
    st = pd.DataFrame(rows)
    st.to_csv(BASE_DIR / "19_stability.csv", index=False)
    print("Estabilidad (scores out-of-fold por pitcher):")
    print(st.pivot_table(index=["nivel", "score"], columns="prueba", values="valor").round(3).to_string())

    # Prediccion del valor de la temporada siguiente (nivel pitcher).
    g = o.groupby(["pitcher_anon_id", "year"]).agg(n=("PitchUID", "size"), **{c: (c, "mean") for c in cols}).reset_index()
    g = g[g.n >= MIN_PITCHER]
    pairs = {}
    for y0, y1 in [(2024, 2025), (2025, 2026)]:
        a = g[g.year == y0].set_index("pitcher_anon_id"); b = g[g.year == y1].set_index("pitcher_anon_id")
        ix = a.index.intersection(b.index)
        pairs[(y0, y1)] = (a.loc[ix], b.loc[ix, "value_eval"])
    preds = {"valor observado t": ["value_eval"], "Stuff+ t": ["stuff_plus"], "Location+ t": ["location_plus"],
             "Pitching+ t": ["pitching_plus"], "valor observado + Stuff+": ["value_eval", "stuff_plus"],
             "valor observado + Pitching+": ["value_eval", "pitching_plus"]}
    prow = []
    for name, feats in preds.items():
        for (p_tr, p_te) in [((2024, 2025), (2025, 2026)), ((2025, 2026), (2024, 2025))]:
            Xa, ya = pairs[p_tr]; Xb, yb = pairs[p_te]
            A = np.column_stack([np.ones(len(Xa))] + [Xa[f].to_numpy() for f in feats])
            beta = np.linalg.lstsq(A, ya.to_numpy(), rcond=None)[0]
            B_ = np.column_stack([np.ones(len(Xb))] + [Xb[f].to_numpy() for f in feats])
            pred = B_ @ beta
            rmse = float(np.sqrt(np.mean((yb.to_numpy() - pred) ** 2)))
            rmse0 = float(np.sqrt(np.mean((yb.to_numpy() - ya.mean()) ** 2)))
            r = boot_corr(pred, yb.to_numpy(), args.boot, rng)
            prow.append({"predictor": name, "ajustado_en": f"{p_tr[0]}->{p_tr[1]}", "evaluado_en": f"{p_te[0]}->{p_te[1]}",
                         "n_pitchers": len(yb), "pearson": r[0], "pearson_lo": r[1], "pearson_hi": r[2],
                         "spearman": r[3], "spearman_lo": r[4], "spearman_hi": r[5], "rmse": rmse, "rmse_nulo": rmse0,
                         "reduccion_mse_vs_nulo": 1 - (rmse / rmse0) ** 2})
    pr = pd.DataFrame(prow)
    pr.to_csv(BASE_DIR / "19_future_value_prediction.csv", index=False)
    print("\nPrediccion del valor por pitch de la temporada siguiente (nivel pitcher, >= 150 pitches ambas temporadas):")
    print(pr.round(4).to_string(index=False))


def stage_calibration(args):
    mods, cfg, comps, weights = setup()
    s = pd.read_csv(BASE_DIR / "18_three_scores.csv")
    raw = pd.read_parquet(BASE_DIR / "stuff_model_features_v1.parquet").dropna(subset=["is_whiff", "pitcher_anon_id"])
    raw = t18.prep(raw)
    lab = raw[["PitchUID"]].copy()
    for c in comps:
        elig, ev = cfg["targets"][c](raw)
        lab[f"e_{c}"], lab[f"y_{c}"] = elig.to_numpy(), ev.to_numpy()
    lab["bip"] = raw["is_ball_in_play"].eq(1).to_numpy()
    s = s.merge(lab, on="PitchUID", how="left", validate="many_to_one")
    rows, dec_rows, rec_rows, recal = [], [], [], {}
    all_comps = comps + ["contact_value"]
    for fam in FAMS:
        for c in all_comps:
            col = f"{fam}_plus__p_{c}"
            if c == "contact_value":
                yv = s["realized_value_env" if fam == "pitching" else "realized_value_pitcher"]
                s[f"e_{c}_{fam}"] = yv.notna(); s[f"y_{c}_{fam}"] = (yv * s["bip"]).fillna(0.0)
                ecol, ycol = f"e_{c}_{fam}", f"y_{c}_{fam}"
            else:
                ecol, ycol = f"e_{c}", f"y_{c}"
            for win in s.window.unique():
                for a in ALTS + ["todas"]:
                    m = (s.window == win) & s[ecol].fillna(False).astype(bool)
                    if a != "todas":
                        m &= s.altitude_category == a
                    if m.sum() < 200:
                        continue
                    y, p = s.loc[m, ycol].to_numpy(float), s.loc[m, col].to_numpy(float)
                    rows.append({"window": win, "family": fam, "component": c, "altitude_category": a, **calib_stats(y, p)})
                    if c != "contact_value" and a == "todas":
                        q = pd.qcut(p, 10, labels=False, duplicates="drop")
                        d = pd.DataFrame({"q": q, "p": p, "y": y}).groupby("q").agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
                        for qi, r in d.iterrows():
                            dec_rows.append({"window": win, "family": fam, "component": c, "decil": int(qi), **r.to_dict()})
            # Recalibracion: aprendida con 2025 (ventana 1), aplicada a 2026 (ventana 2).
            groups = ALTS if fam == "pitching" else ["todas"]
            for a in groups:
                m1 = (s.window == "2024 -> 2025") & s[ecol].fillna(False).astype(bool)
                m2 = (s.window == PRIMARY) & s[ecol].fillna(False).astype(bool)
                if a != "todas":
                    m1 &= s.altitude_category == a; m2 &= s.altitude_category == a
                y1, p1 = s.loc[m1, ycol].to_numpy(float), s.loc[m1, col].to_numpy(float)
                y2, p2 = s.loc[m2, ycol].to_numpy(float), s.loc[m2, col].to_numpy(float)
                if c == "contact_value":
                    A = np.column_stack([np.ones(len(p1)), p1]); ab = np.linalg.lstsq(A, y1, rcond=None)[0]
                    p2r = ab[0] + ab[1] * p2
                else:
                    ab = platt(p1, y1)
                    p2r = 1 / (1 + np.exp(-(ab[0] + ab[1] * logit(p2))))
                recal[(fam, c, a)] = ab
                for alt_eval in ([a] if a != "todas" else ALTS + ["todas"]):
                    mm = np.ones(len(p2), bool) if alt_eval == "todas" else (s.loc[m2, "altitude_category"] == alt_eval).to_numpy()
                    r0, r1 = calib_stats(y2[mm], p2[mm]), calib_stats(y2[mm], p2r[mm])
                    rec_rows.append({"family": fam, "component": c, "calibrador": a, "altitude_category": alt_eval, "n": int(mm.sum()),
                                     "brier_raw": r0.get("brier"), "brier_recal": r1.get("brier"),
                                     "rmse_raw": r0.get("rmse"), "rmse_recal": r1.get("rmse"),
                                     "cil_raw": r0["calib_in_large"], "cil_recal": r1["calib_in_large"],
                                     "slope_raw": r0.get("calib_slope"), "slope_recal": r1.get("calib_slope"),
                                     "auc_raw": r0.get("auc"), "auc_recal": r1.get("auc")})
    pd.DataFrame(rows).to_csv(BASE_DIR / "19_calibration_raw.csv", index=False)
    pd.DataFrame(dec_rows).to_csv(BASE_DIR / "19_reliability_deciles.csv", index=False)
    rec = pd.DataFrame(rec_rows)
    rec.to_csv(BASE_DIR / "19_recalibration_2026.csv", index=False)
    joblib.dump(recal, BASE_DIR / "19_recalibrators.pkl")
    cr = pd.DataFrame(rows)
    print("Calibracion cruda (temporada evaluada, todas las altitudes): calibracion global (pp) y pendiente")
    print(cr[cr.altitude_category == "todas"].pivot_table(index=["family", "component"], columns="window",
          values=["calib_in_large", "calib_slope"]).round(4).to_string())
    print("\nRecalibracion (aprendida con 2025, evaluada en 2026):")
    print(rec[rec.altitude_category != "todas"].round(4).to_string(index=False))


def stage_summary(args):
    cvm = pd.read_csv(BASE_DIR / "19_cv_component_metrics.csv"); cvs = pd.read_csv(BASE_DIR / "19_cv_composite_metrics.csv")
    pk = pd.read_csv(BASE_DIR / "19_park_composite_metrics.csv"); pkc = pd.read_csv(BASE_DIR / "19_park_component_metrics.csv")
    st = pd.read_csv(BASE_DIR / "19_stability.csv"); fp = pd.read_csv(BASE_DIR / "19_future_value_prediction.csv")
    sf = pd.read_csv(BASE_DIR / "18_out_of_sample_value.csv"); rec = pd.read_csv(BASE_DIR / "19_recalibration_2026.csv")
    rows = []
    for fam in FAMS:
        cv = cvs[cvs.family == fam]["mse_reduction_vs_null"]
        s = sf[(sf.family == fam) & (sf.model == "pesos_sin_contact_value")]
        e = pk[(pk.family == fam) & pk.direccion.str.endswith("Extreme")]["mse_reduction_vs_null"].iloc[0]
        auc_cv = cvm[(cvm.family == fam) & (cvm.component == "whiff_per_pitch")]["auc"]
        auc_pk = pkc[(pkc.family == fam) & (pkc.component == "whiff_per_pitch") & pkc.direccion.str.endswith("Extreme")]["auc"].iloc[0]
        sh = st[(st.nivel == "pitcher") & (st.score == f"{fam}_plus") & st.prueba.str.startswith("split")]["valor"]
        yy = st[(st.nivel == "pitcher") & (st.score == f"{fam}_plus") & st.prueba.str.startswith("año")]["valor"]
        fpp = fp[fp.predictor == f"{fam.capitalize()}+ t"]["pearson"]
        rc = rec[(rec.family == fam) & (rec.altitude_category != "todas") & rec.brier_raw.notna()]
        rows.append({"score": fam, "CV pitcher red. MSE (media±SD)": f"{cv.mean():.3%} ± {cv.std():.3%}",
                     "CV AUC whiff (media±SD)": f"{auc_cv.mean():.4f} ± {auc_cv.std():.4f}",
                     "season-forward red. MSE 2025/2026": f"{s.iloc[0].mse_reduction_vs_null:.3%} / {s.iloc[1].mse_reduction_vs_null:.3%}",
                     "park -> Extreme red. MSE / AUC whiff": f"{e:.3%} / {auc_pk:.4f}",
                     "split-half pitcher (rango)": f"{sh.min():.3f}-{sh.max():.3f}",
                     "año a año pitcher (24-25, 25-26)": ", ".join(f"{v:.3f}" for v in yy),
                     "corr valor t+1 (dos pares)": ", ".join(f"{v:.3f}" for v in fpp),
                     "Brier raw->recal 2026 (media componentes)": f"{rc.brier_raw.mean():.5f} -> {rc.brier_recal.mean():.5f}"})
    rows.append({"score": "valor observado",
                 "split-half pitcher (rango)": "{:.3f}-{:.3f}".format(*st[(st.nivel == "pitcher") & (st.score == "value_eval") & st.prueba.str.startswith("split")]["valor"].agg(["min", "max"])),
                 "año a año pitcher (24-25, 25-26)": ", ".join(f"{v:.3f}" for v in st[(st.nivel == "pitcher") & (st.score == "value_eval") & st.prueba.str.startswith("año")]["valor"]),
                 "corr valor t+1 (dos pares)": ", ".join(f"{v:.3f}" for v in fp[fp.predictor == "valor observado t"]["pearson"])})
    t = pd.DataFrame(rows)
    t.to_csv(BASE_DIR / "19_exit_table.csv", index=False)
    print(t.T.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["cv", "park", "stability", "calibration", "summary", "all"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    pd.set_option("display.width", 240)
    stages = ["cv", "park", "stability", "calibration", "summary"] if args.stage == "all" else [args.stage]
    for st in stages:
        print(f"\n{'#' * 30} ETAPA {st} {'#' * 30}", flush=True)
        {"cv": stage_cv, "park": stage_park, "stability": stage_stability,
         "calibration": stage_calibration, "summary": stage_summary}[st](args)


if __name__ == "__main__":
    main()
