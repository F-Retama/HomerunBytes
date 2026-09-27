"""Bloque D (D3-D4): efecto de altitud sobre el Stuff+ limpio y tabla de hipotesis H1-H4.

delta = Stuff+(X transformado a Extreme) - Stuff+(X observado en No Altitude), con el Stuff+
oficial de 18: fisica pura, sin ubicacion, sin count y sin altitude_category. El cambio viene
solo de como la atmosfera transforma el shape. Transformacion elegida en 20
(mult_global_var): IVB x r_IVB y HB x r_HB, aprendidos con el historial de cada ventana. Con
una transformacion multiplicativa comun, los diferenciales contra la recta (ivb_diff_vs_fb,
hb_diff_vs_fb) escalan con el mismo factor. Horizontales en convencion de diestro (el frame de
18 ya esta espejado).

Incertidumbre: bootstrap por pitcher (B por defecto 200) que remuestrea a la vez las celdas que
estiman r y la cohorte evaluada. Chequeo inverso: pitches observados en Extreme llevados a No
(X / r). Sensibilidad: formas aditiva_tipo y mult_global_magnus (puntuales). Niveles de
evidencia: Robusto = IC excluye 0 con el mismo signo en ambas ventanas, chequeo inverso con el
mismo signo y >= 30 celdas del tipo; Condicional = mismo signo en ambas ventanas pero algun IC
cruza 0; Incierto = < 30 celdas o signo distinto entre ventanas. Relevancia practica: |delta|
>= 1 punto (umbral interno).

H1-H4: H1 = transformacion + gradiente (20) + delta por tipo. H2 = changeup/splitter frente a
tipos basados en Magnus (diferencia de deltas con IC). H3 = evidencia de contacto de 17 y 18
(no se fuerza en Stuff+). H4 = pendiente IVB~SpinRate por entorno en Four-Seam y ablacion de
whiff (spin sin movimiento vs movimiento sin spin), evaluada por entorno.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE_DIR = Path(__file__).resolve().parent
WINDOWS = {"2024 -> 2025": [2024], "2024-2025 -> 2026": [2024, 2025]}
PRIMARY = "2024-2025 -> 2026"
SCALE_KEY = "scale_sin_contact_value"
MIN_CELLS, RELEVANT, MIN_PITCHER = 30, 1.0, 100
MAGNUS = ["Four-Seam", "Sinker", "Slider", "Curveball", "Cutter"]
SPEED_DIFF = ["Changeup", "Splitter"]


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, BASE_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


t18 = _load("three_scores_v2", "18_three_scores_v2.py")
t16, t17 = t18.t16, t18.t17
t20 = _load("altitude_physics", "20_altitude_physics.py")


class StuffScorer:
    """Stuff+ oficial de 18 para una ventana, solo con los componentes de peso no nulo."""

    def __init__(self, entry, cfg):
        fam = entry["families"]["stuff"]
        self.fam, self.variants = fam, fam["variants"]
        comps = cfg["components"]
        w = np.asarray(fam["weights"], float)
        keep = np.abs(w) > 1e-9
        self.comps = [c for c, k in zip(comps, keep) if k]
        self.w = w[keep]
        self.sign = np.array([cfg["signs"].get(c, 1.0) for c in self.comps])
        self.stats = [fam["stats"][c] for c in self.comps]
        self.m, self.s = fam["stats"][SCALE_KEY]

    def score(self, frame, has_ars):
        Z = np.column_stack([sg * (t16.family_predict(self.fam["models"][c], frame, has_ars, self.variants) - mu) / sd
                             for c, sg, (mu, sd) in zip(self.comps, self.sign, self.stats)])
        return 100 + 10 * (Z @ self.w - self.m) / self.s


def transform(frame, form, p, inverse=False):
    f = frame.copy()
    if form in ("mult_global_var", "mult_global_magnus"):
        rI = p["InducedVertBreak"] if form == "mult_global_var" else p["r"]
        rH = p["HorzBreak"] if form == "mult_global_var" else p["r"]
        if inverse:
            rI, rH = 1 / rI, 1 / rH
        for col, r in [("InducedVertBreak", rI), ("ivb_diff_vs_fb", rI), ("HorzBreak", rH), ("hb_diff_vs_fb", rH)]:
            f[col] = f[col] * r
    else:  # aditiva_tipo
        sg = -1.0 if inverse else 1.0
        t = f["AutoPitchType"].astype(str).to_numpy(); fb = f["primary_fastball_type"].astype(str).to_numpy()
        for v, dcol in [("InducedVertBreak", "ivb_diff_vs_fb"), ("HorzBreak", "hb_diff_vs_fb")]:
            own = np.array([p.get((x, v), 0.0) for x in t]); ref = np.array([p.get((x, v), 0.0) for x in fb])
            f[v] = f[v] + sg * own
            f[dcol] = f[dcol] + sg * (own - ref)
    return f


def group_mean(codes, weights, values, n):
    num = np.bincount(codes, weights=weights * values, minlength=n); den = np.bincount(codes, weights=weights, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def run_window(label, entry, cfg, d_raw, B, rng):
    scorer = StuffScorer(entry, cfg)
    F = entry["future_frame"]; ars = entry["f_ars"]
    # Identidad con los scores publicados por 18.
    pub = pd.read_csv(BASE_DIR / "18_three_scores.csv", usecols=["window", "stuff_plus"])
    pub = pub[pub.window == label]["stuff_plus"].to_numpy()
    base_all = scorer.score(F, ars)
    diff = float(np.nanmax(np.abs(base_all - pub)))
    if diff > 1e-6:
        raise ValueError(f"{label}: el Stuff+ reconstruido no coincide con 18 (max diff {diff}).")
    alt = F["altitude_category"].astype(str).to_numpy()
    hist = d_raw[d_raw.year.isin(WINDOWS[label])]
    cells = t20.cells(hist, "Extreme Altitude")
    p0 = t20.fit(cells, "mult_global_var")

    coh = F[alt == "No Altitude"].reset_index(drop=True); ars_c = ars[alt == "No Altitude"]
    base = base_all[alt == "No Altitude"]
    types = sorted(coh["AutoPitchType"].astype(str).unique())
    tcode = pd.Index(types).get_indexer(coh["AutoPitchType"].astype(str))
    pit = pd.Index(pd.unique(np.concatenate([cells["pitcher_anon_id"].to_numpy(), coh["pitcher_anon_id"].to_numpy()])))
    cell_code, coh_code = pit.get_indexer(cells["pitcher_anon_id"]), pit.get_indexer(coh["pitcher_anon_id"])
    d0 = scorer.score(transform(coh, "mult_global_var", p0), ars_c) - base
    coh = t16.pd.DataFrame(coh)
    coh["delta"] = d0
    rep_type = np.full((B, len(types)), np.nan); rep_groups = np.full((B, 2), np.nan)
    ivb_t = coh.groupby("AutoPitchType", observed=True)["InducedVertBreak"].transform(lambda s: pd.qcut(s, 3, labels=["IVB bajo", "IVB medio", "IVB alto"], duplicates="drop").astype(str))
    hb_t = coh.groupby("AutoPitchType", observed=True)["HorzBreak"].transform(lambda s: pd.qcut(s.abs(), 3, labels=["|HB| bajo", "|HB| medio", "|HB| alto"], duplicates="drop").astype(str))
    coh["shape_bin"] = coh["AutoPitchType"].astype(str) + " | " + ivb_t + " | " + hb_t
    bins = pd.Index(sorted(coh["shape_bin"].unique())); bcode = bins.get_indexer(coh["shape_bin"])
    rep_bin = np.full((B, len(bins)), np.nan); rep_r = np.full((B, 2), np.nan)
    is_sd = coh["AutoPitchType"].astype(str).isin(SPEED_DIFF).to_numpy(); is_mg = coh["AutoPitchType"].astype(str).isin(MAGNUS).to_numpy()
    for b in range(B):
        mult = np.bincount(rng.integers(0, len(pit), len(pit)), minlength=len(pit)).astype(float)
        pb = t20.fit(cells, "mult_global_var", mult[cell_code])
        rep_r[b] = [pb["InducedVertBreak"], pb["HorzBreak"]]
        db = scorer.score(transform(coh, "mult_global_var", pb), ars_c) - base
        w = mult[coh_code]
        rep_type[b] = group_mean(tcode, w, db, len(types))
        rep_bin[b] = group_mean(bcode, w, db, len(bins))
        rep_groups[b] = [np.sum(w * db * is_sd) / np.sum(w * is_sd), np.sum(w * db * is_mg) / np.sum(w * is_mg)]
        if (b + 1) % 25 == 0:
            print(f"  {label}: bootstrap {b + 1}/{B}", flush=True)
    ci = lambda a: np.nanpercentile(a, [2.5, 97.5], axis=0)
    ones = np.ones(len(coh))
    pt_type = group_mean(tcode, ones, d0, len(types)); lo, hi = ci(rep_type)
    ncell = cells.groupby("AutoPitchType").size()

    # Chequeo inverso.
    rev = F[alt == "Extreme Altitude"].reset_index(drop=True); ars_r = ars[alt == "Extreme Altitude"]
    d_rev = base_all[alt == "Extreme Altitude"] - scorer.score(transform(rev, "mult_global_var", p0, inverse=True), ars_r)
    rev_type = pd.Series(d_rev).groupby(rev["AutoPitchType"].astype(str).to_numpy()).mean()

    # Contraste OBSERVADO: Stuff+ real del mismo pitcher x tipo en Extreme vs No (celdas >= 20 en ambos).
    ob = pd.DataFrame({"p": F["pitcher_anon_id"].to_numpy(), "t": F["AutoPitchType"].astype(str).to_numpy(), "alt": alt, "s": base_all})
    ob = ob[np.isin(alt, ["No Altitude", "Extreme Altitude"])]
    cc = ob.groupby(["p", "t", "alt"])["s"].agg(["size", "mean"]).unstack("alt").dropna()
    cc = cc[(cc[("size", "No Altitude")] >= 20) & (cc[("size", "Extreme Altitude")] >= 20)]
    cw = np.minimum(cc[("size", "No Altitude")], cc[("size", "Extreme Altitude")]).to_numpy(float)
    cd = (cc[("mean", "Extreme Altitude")] - cc[("mean", "No Altitude")]).to_numpy()
    ct = cc.index.get_level_values("t").to_numpy(); cp = pd.Index(cc.index.get_level_values("p").unique()); ccode = cp.get_indexer(cc.index.get_level_values("p"))
    obs = {}
    for t in types:
        m = ct == t
        if m.sum() < 5:
            continue
        pt = np.sum(cw[m] * cd[m]) / np.sum(cw[m])
        reps = []
        for _ in range(B):
            mm = np.bincount(rng.integers(0, len(cp), len(cp)), minlength=len(cp)).astype(float)[ccode]
            ww = cw * mm
            reps.append(np.sum(ww[m] * cd[m]) / np.sum(ww[m]) if np.sum(ww[m]) > 0 else np.nan)
        obs[t] = (pt, *np.nanpercentile(reps, [2.5, 97.5]), int(m.sum()))

    # Sensibilidad a la forma.
    sens = {}
    for form in ["aditiva_tipo", "mult_global_magnus"]:
        ps = t20.fit(cells, form)
        ds = scorer.score(transform(coh, form, ps), ars_c) - base
        sens[form] = pd.Series(ds).groupby(coh["AutoPitchType"].astype(str).to_numpy()).mean()

    by_type = pd.DataFrame({"window": label, "AutoPitchType": types, "n_pitches": np.bincount(tcode, minlength=len(types)),
                            "n_pitchers": [coh.loc[tcode == i, "pitcher_anon_id"].nunique() for i in range(len(types))],
                            "n_celdas_historial": [int(ncell.get(t, 0)) for t in types],
                            "delta_stuff": pt_type, "ci95_lo": lo, "ci95_hi": hi,
                            "delta_inverso": [rev_type.get(t, np.nan) for t in types],
                            "delta_observado_within": [obs.get(t, (np.nan,) * 4)[0] for t in types],
                            "obs_ci95_lo": [obs.get(t, (np.nan,) * 4)[1] for t in types],
                            "obs_ci95_hi": [obs.get(t, (np.nan,) * 4)[2] for t in types],
                            "obs_n_celdas": [obs.get(t, (np.nan,) * 4)[3] for t in types],
                            "delta_aditiva_tipo": [sens["aditiva_tipo"].get(t, np.nan) for t in types],
                            "delta_mult_magnus": [sens["mult_global_magnus"].get(t, np.nan) for t in types]})
    blo, bhi = ci(rep_bin)
    by_bin = pd.DataFrame({"window": label, "shape_bin": bins, "n_pitches": np.bincount(bcode, minlength=len(bins)),
                           "n_pitchers": [coh.loc[bcode == i, "pitcher_anon_id"].nunique() for i in range(len(bins))],
                           "delta_stuff": group_mean(bcode, ones, d0, len(bins)), "ci95_lo": blo, "ci95_hi": bhi})
    pf = coh.groupby("pitcher_anon_id").agg(n=("delta", "size"), delta_stuff=("delta", "mean"), stuff_nivel_mar=("delta", lambda s: np.nan)).reset_index()
    pf["stuff_nivel_mar"] = pd.Series(base).groupby(coh["pitcher_anon_id"].to_numpy()).mean().reindex(pf["pitcher_anon_id"]).to_numpy()
    pf = pf[pf.n >= MIN_PITCHER].assign(window=label)
    groups = {"speed_diff": (float(np.mean(d0[is_sd])), *ci(rep_groups[:, 0])),
              "magnus": (float(np.mean(d0[is_mg])), *ci(rep_groups[:, 1])),
              "diferencia": (float(np.mean(d0[is_sd]) - np.mean(d0[is_mg])), *ci(rep_groups[:, 0] - rep_groups[:, 1]))}
    rr = {"r_IVB": (p0["InducedVertBreak"], *ci(rep_r[:, 0])), "r_HB": (p0["HorzBreak"], *ci(rep_r[:, 1]))}
    coh_out = coh[["PitchUID", "pitcher_anon_id", "AutoPitchType", "shape_bin", "InducedVertBreak", "HorzBreak", "RelSpeed",
                   "SpinRate", "delta"]].assign(window=label, stuff_nivel_mar=base)
    return by_type, by_bin, pf, groups, rr, coh_out


def evidence(bt):
    rows = []
    for t, g in bt.groupby("AutoPitchType"):
        g = g.set_index("window")
        if len(g) < 2:
            continue
        s = np.sign(g["delta_stuff"])
        excl = (g["ci95_lo"] > 0) | (g["ci95_hi"] < 0)
        same = s.nunique() == 1
        rev_ok = (np.sign(g["delta_inverso"]) == s).all()
        cells_ok = (g["n_celdas_historial"] >= MIN_CELLS).all()
        # Confirmacion empirica: el Stuff+ OBSERVADO del mismo pitcher x tipo (Extreme - No) tiene el mismo
        # signo con IC que excluye 0 en ambas ventanas (regla endurecida el 26/09/2026 tras ver que en
        # slider/changeup/curva la perdida modelada no aparece en el neto observado).
        obs_ok = (((g["obs_ci95_hi"] < 0) & (s < 0)) | ((g["obs_ci95_lo"] > 0) & (s > 0))).all()
        if not cells_ok or not same:
            lvl = "Incierto"
        elif excl.all() and rev_ok and obs_ok:
            lvl = "Robusto"
        else:
            lvl = "Condicional"
        p = g.loc[PRIMARY]
        rows.append({"AutoPitchType": t, "nivel_evidencia": lvl, "delta_2026": p.delta_stuff, "ci95_lo": p.ci95_lo, "ci95_hi": p.ci95_hi,
                     "delta_2025": g.loc["2024 -> 2025", "delta_stuff"], "relevante_>=1pt": bool(abs(p.delta_stuff) >= RELEVANT),
                     "observado_2026": p.delta_observado_within, "observado_2025": g.loc["2024 -> 2025", "delta_observado_within"],
                     "confirmado_observado": bool(obs_ok),
                     "celdas_min": int(g["n_celdas_historial"].min())})
    return pd.DataFrame(rows)


def h4_tests(cfg, mods, rng, B):
    """(a) pendiente IVB ~ SpinRate en Four-Seam por entorno; (b) ablacion spin vs movimiento para whiff."""
    d = pd.read_parquet(BASE_DIR / "stuff_model_features_v1.parquet").dropna(subset=["is_whiff", "pitcher_anon_id"])
    d = t16.mirror(d)
    # (a) Within-pitcher: movimiento inducido por cada 1000 rpm, mismo pitcher x tipo x anio, entorno vs No.
    d["mov"] = np.sqrt(d["InducedVertBreak"] ** 2 + d["HorzBreak"] ** 2)
    rows = []
    for env in ["Medium Altitude", "Extreme Altitude"]:
        x = d[d.altitude_category.astype(str).isin(["No Altitude", env])].dropna(subset=["mov", "SpinRate"])
        g = x.groupby(["pitcher_anon_id", "AutoPitchType", "year", x.altitude_category.astype(str)], observed=True).agg(
            n=("mov", "size"), mov=("mov", "mean"), spin=("SpinRate", "mean")).unstack().dropna()
        g = g[(g[("n", "No Altitude")] >= 20) & (g[("n", env)] >= 20)]
        eff_n = 1000 * g[("mov", "No Altitude")] / g[("spin", "No Altitude")]; eff_e = 1000 * g[("mov", env)] / g[("spin", env)]
        spin_d = (g[("spin", env)] - g[("spin", "No Altitude")]).to_numpy()
        w = np.minimum(g[("n", "No Altitude")], g[("n", env)]).to_numpy(float)
        ratio = (eff_e / eff_n).to_numpy()
        pit = pd.Index(g.index.get_level_values(0).unique()); codes = pit.get_indexer(g.index.get_level_values(0))
        for name, vals in [("movimiento por 1000 rpm: razon entorno/No (within-pitcher)", ratio),
                           ("cambio de SpinRate (rpm, within-pitcher)", spin_d)]:
            pt = np.sum(w * vals) / np.sum(w)
            reps = []
            for _ in range(B):
                m = np.bincount(rng.integers(0, len(pit), len(pit)), minlength=len(pit)).astype(float)[codes]
                reps.append(np.sum(w * m * vals) / np.sum(w * m))
            rows.append({"prueba": name, "entorno": env, "valor": pt, "ci95_lo": np.percentile(reps, 2.5),
                         "ci95_hi": np.percentile(reps, 97.5), "n": len(g)})
    # Ablacion (ventana 2024-25 -> 2026, variante base, target whiff por pitch).
    for c in ["AutoPitchType", "PitcherThrows"]:
        d[c] = d[c].astype("category")
    tr, te = d[d.year.isin([2024, 2025])], d[d.year == 2026]
    fn = cfg["targets"]["whiff_per_pitch"]
    sets = {"fisica completa (Stuff+)": t16.PHYS_NUM + t16.PHYS_CAT,
            "spin sin movimiento": ["RelSpeed", "SpinRate", "SpinAxis", "RelHeight", "RelSide", "Extension"] + t16.PHYS_CAT,
            "movimiento sin spin": ["RelSpeed", "InducedVertBreak", "HorzBreak", "RelHeight", "RelSide", "Extension"] + t16.PHYS_CAT}
    for name, cols in sets.items():
        _m, (p,) = t16.fit_predict(mods["boosting"], fn, tr, cols, [te[cols]])
        e, y = fn(te); e = e.to_numpy()
        for env in ["No Altitude", "Medium Altitude", "Extreme Altitude", "todos"]:
            m = e & ((te.altitude_category.astype(str) == env).to_numpy() if env != "todos" else True)
            rows.append({"prueba": f"AUC whiff 2026: {name}", "entorno": env, "valor": roc_auc_score(y.to_numpy()[m], p[m]), "n": int(m.sum())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--boot", type=int, default=200); ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(); rng = np.random.default_rng(args.seed); pd.set_option("display.width", 230)
    mods = t16.load_modules(); cfg = t18.build_cfg(mods)
    bundle = joblib.load(BASE_DIR / "18_three_scores_bundle.pkl")
    choice = json.load(open(BASE_DIR / "20_transform_choice.json"))["forma_elegida"]
    if choice != "mult_global_var":
        raise ValueError(f"20 eligio {choice}; este script implementa mult_global_var como principal.")
    d_raw = t20.load_raw()
    bts, bbs, pfs, gs, rrs, cohs = [], [], [], {}, {}, []
    for label, entry in bundle.items():
        print(f"\n=== {label} ===", flush=True)
        bt, bb, pf, g, rr, co = run_window(label, entry, cfg, d_raw, args.boot, rng)
        bts.append(bt); bbs.append(bb); pfs.append(pf); gs[label] = g; rrs[label] = rr; cohs.append(co)
    bt, bb, pf = pd.concat(bts), pd.concat(bbs), pd.concat(pfs)
    ev = evidence(bt)
    bt.to_csv(BASE_DIR / "21_delta_by_pitch_type.csv", index=False); bb.to_csv(BASE_DIR / "21_delta_by_shape_bin.csv", index=False)
    pf.to_csv(BASE_DIR / "21_delta_by_pitcher.csv", index=False); ev.to_csv(BASE_DIR / "21_evidence_levels.csv", index=False)
    pd.concat(cohs).to_parquet(BASE_DIR / "21_delta_rows.parquet", index=False)
    print("\nDelta Stuff+ (Extreme - No) por tipo, transformacion mult_global_var (IC95% bootstrap por pitcher):")
    print(bt.round(3).to_string(index=False))
    print("\nNiveles de evidencia:"); print(ev.round(3).to_string(index=False))
    sig = bb[((bb.ci95_lo > 0) | (bb.ci95_hi < 0)) & (bb.n_pitchers >= 30) & (bb.delta_stuff.abs() >= RELEVANT)]
    print(f"\nShape bins con IC sin 0, >= 30 pitchers y |delta| >= 1 pt: {len(sig)} de {len(bb)} (sin correccion por comparaciones multiples)")
    print(sig.round(2).to_string(index=False))
    for label in gs:
        print(f"\n{label}: r_IVB {rrs[label]['r_IVB'][0]:.3f} [{rrs[label]['r_IVB'][1]:.3f}, {rrs[label]['r_IVB'][2]:.3f}], "
              f"r_HB {rrs[label]['r_HB'][0]:.3f}; H2 speed-diff {gs[label]['speed_diff'][0]:+.2f}, magnus {gs[label]['magnus'][0]:+.2f}, "
              f"diferencia {gs[label]['diferencia'][0]:+.2f} [{gs[label]['diferencia'][1]:+.2f}, {gs[label]['diferencia'][2]:+.2f}]")
    pp = pf[pf.window == PRIMARY]
    print(f"Pitchers (n >= {MIN_PITCHER} en No, 2026): {len(pp)} | delta medio {pp.delta_stuff.mean():+.2f} | SD entre pitchers {pp.delta_stuff.std():.2f}")

    # ---------------- D4: H1-H4 ----------------
    h4 = h4_tests(cfg, mods, rng, args.boot)
    h4.to_csv(BASE_DIR / "21_h4_tests.csv", index=False)
    print("\nH4:"); print(h4.round(4).to_string(index=False))
    grad = pd.read_csv(BASE_DIR / "20_altitude_gradient.csv"); c17 = pd.read_csv(BASE_DIR / "17_contact_by_altitude.csv").set_index("altitude_category")
    w17 = pd.read_csv(BASE_DIR / "17_contact_within_pitcher.csv").set_index("metrica")
    gt = grad[(grad.AutoPitchType == "todos") & (grad.variable == "magnus")].set_index("entorno")
    g2 = gs[PRIMARY]
    nd = {r.entorno: r for r in h4[h4.prueba.str.startswith("movimiento por")].itertuples()}
    sd = {r.entorno: r for r in h4[h4.prueba.str.startswith("cambio de SpinRate")].itertuples()}
    ab = h4[h4.prueba.str.startswith("AUC")].pivot(index="prueba", columns="entorno", values="valor")
    hy = pd.DataFrame([
        {"hipotesis": "H1: la baja densidad reduce el movimiento inducido por spin",
         "evidencia": f"Movimiento conservado (within-pitcher, r Magnus): Medium {gt.loc['Medium Altitude','r']:.3f}, Extreme {gt.loc['Extreme Altitude','r']:.3f} (No = 1); gradiente monotono en los 7 tipos; transformacion validada en temporada futura (20). Efecto en Stuff+ 2026 (modelado / observado mismo pitcher): " + "; ".join(f"{t} {bt[(bt.window == PRIMARY) & (bt.AutoPitchType == t)].delta_stuff.iloc[0]:+.2f} / {bt[(bt.window == PRIMARY) & (bt.AutoPitchType == t)].delta_observado_within.iloc[0]:+.2f}" for t in ["Four-Seam", "Sinker", "Changeup", "Slider", "Curveball"]),
         "veredicto": "Soportada en movimiento (todos los tipos) y en Stuff+ observado para rectas; en slider/changeup/curva la perdida modelada no aparece en el neto observado"},
        {"hipotesis": "H2: pitches por diferencial de velocidad pierden menos valor relativo",
         "evidencia": f"Delta Stuff+ 2026: changeup/splitter {g2['speed_diff'][0]:+.2f} vs Magnus {g2['magnus'][0]:+.2f}; diferencia {g2['diferencia'][0]:+.2f} [{g2['diferencia'][1]:+.2f}, {g2['diferencia'][2]:+.2f}]",
         "veredicto": ("Soportada en terminos relativos: pierden menos, pero no ganan" if g2["speed_diff"][0] < 0 else "Soportada") if g2["diferencia"][1] > 0 else ("Contradicha" if g2["diferencia"][2] < 0 else "No concluyente")},
        {"hipotesis": "H3: los groundball pitchers valen mas en altitud",
         "evidencia": f"Ventaja GB - FB en runs: No {c17.loc['No Altitude','gb_minus_fb_value']:+.3f}, Extreme {c17.loc['Extreme Altitude','gb_minus_fb_value']:+.3f}; HR/FB within-pitcher {w17.loc['hr_per_flyball','delta_extreme_minus_no']:+.3f} [{w17.loc['hr_per_flyball','ci95_lo']:+.3f}, {w17.loc['hr_per_flyball','ci95_hi']:+.3f}]; dano de contacto esperado menor en pitches de alta propension GB (18, guia 34.3)",
         "veredicto": "Soportada (descriptiva; no entra al Stuff+)"},
        {"hipotesis": "H4: el spin rate solo es insuficiente",
         "evidencia": f"Mismo pitcher: el movimiento por 1000 rpm conserva {nd['Medium Altitude'].valor:.3f} en Medium y {nd['Extreme Altitude'].valor:.3f} [{nd['Extreme Altitude'].ci95_lo:.3f}, {nd['Extreme Altitude'].ci95_hi:.3f}] en Extreme, con SpinRate practicamente igual ({sd['Extreme Altitude'].valor:+.1f} rpm): las mismas rpm producen menos movimiento. Predictivamente, spin sin movimiento y movimiento sin spin dan AUC de whiff casi iguales en Extreme ({ab.loc['AUC whiff 2026: spin sin movimiento','Extreme Altitude']:.4f} vs {ab.loc['AUC whiff 2026: movimiento sin spin','Extreme Altitude']:.4f}; completo {ab.loc['AUC whiff 2026: fisica completa (Stuff+)','Extreme Altitude']:.4f})",
         "veredicto": "Parcial: soportada en lo fisico (rpm -> movimiento se degrada en altitud); no en lo predictivo (spin y movimiento aportan senal similar al modelo)"}])
    hy.to_csv(BASE_DIR / "21_hypotheses.csv", index=False)
    print("\nH1-H4:"); print(hy.to_string(index=False))
    json.dump({k: {kk: [float(x) for x in vv] for kk, vv in v.items()} for k, v in rrs.items()}, open(BASE_DIR / "21_transform_r.json", "w"), indent=2)
    print("\nArchivos: 21_delta_by_pitch_type.csv, 21_delta_by_shape_bin.csv, 21_delta_by_pitcher.csv, 21_evidence_levels.csv, "
          "21_delta_rows.parquet, 21_h4_tests.csv, 21_hypotheses.csv, 21_transform_r.json")


if __name__ == "__main__":
    main()
