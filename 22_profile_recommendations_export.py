"""Bloque D (D5-D7): perfil fisico ideal en altitud, recomendaciones, proyeccion por pitcher y
salidas congeladas para el dashboard.

Invalida formalmente el perfil de 15 (construido sobre el score V1 de 09, contaminado por
ubicacion). Todo sale del Stuff+ limpio de 18 (fisica pura, neutral al entorno) y de la
transformacion validada en 20/21. No se usan Location+ ni Pitching+ para el perfil fisico.

D5 Perfil por pitch type (Extreme Altitude, proxy del entorno Harp Helu):
  - Nivel: pitches OBSERVADOS en Extreme (su fisica ya incluye el entorno). Lift del quintil
    superior de Stuff+ contra todos los del tipo = (media top - media) / SD, IC95% bootstrap
    por pitcher. Zona = P25-P75 del quintil superior. Se aprende con 2025 (scores fuera de
    muestra de la ventana 1) y se confirma en 2026 (ventana 2). Una caracteristica entra al
    perfil si |lift| >= 0.10, su IC excluye 0 y el signo se repite en 2026 con IC sin 0.
    Compartida vs especifica: se compara con el lift en No Altitude.
  - Resiliencia: correlacion de rango entre cada caracteristica y el delta de 21 (pitches de
    No llevados a Extreme): positivo = mas de esa caracteristica, menos perdida.
  - Estabilidad por tipo: correlacion año a año por pitcher x tipo de los scores out-of-fold de 19.
  - Validacion con resultados reales: en Extreme 2026, brecha de whiff y chase observados entre
    el quintil alto y el bajo de Stuff+ (IC bootstrap por pitcher).
D6 Recomendaciones por tipo segun nivel de evidencia (21), mecanismo fisico (20), riesgo de
  contacto (17/18), perfil y resiliencia.
Proyeccion: cada pitch se lleva a su equivalente fisico a nivel del mar (No) y en Harp Helu
  (Extreme) con los factores multiplicativos (Extreme de la ventana; Medium de 20, todas las
  temporadas) y se califica con el Stuff+ de su ventana. Agregados por pitcher x tipo y por pitcher.
D7 Congelamiento: 22_dashboard_manifest.json (archivos con SHA256, factores, ventana del modelo)
  y simulate_pitch(), verificada contra el scorer oficial.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE_DIR = Path(__file__).resolve().parent
LEARN, VALID = "2024 -> 2025", "2024-2025 -> 2026"
FEATURES = ["RelSpeed", "InducedVertBreak", "HorzBreak", "SpinRate", "SpinAxis", "RelHeight", "RelSide", "Extension",
            "velo_diff_vs_fb", "ivb_diff_vs_fb", "hb_diff_vs_fb", "pitch_usage_pct"]
MIN_LIFT, MIN_ROWS, MIN_PITCHERS, MIN_CELL = 0.10, 300, 15, 20


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, BASE_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


t21 = _load("altitude_stuff_delta", "21_altitude_stuff_delta.py")
t18, t16 = t21.t18, t21.t16


def lifts(frame, score, B, rng):
    d = frame.dropna(subset=FEATURES)
    X, y = d[FEATURES].to_numpy(float), d[score].to_numpy()
    top = y >= np.quantile(y, 0.8)
    pit = pd.Index(d["pitcher_anon_id"].unique()); codes = pit.get_indexer(d["pitcher_anon_id"])
    sd = X.std(axis=0); sd[sd == 0] = np.nan
    n_all = np.bincount(codes, minlength=len(pit)).astype(float); n_top = np.bincount(codes, weights=top.astype(float), minlength=len(pit))
    sx_all = np.stack([np.bincount(codes, weights=X[:, j], minlength=len(pit)) for j in range(X.shape[1])], 1)
    sx_top = np.stack([np.bincount(codes, weights=X[:, j] * top, minlength=len(pit)) for j in range(X.shape[1])], 1)
    lf = lambda m: ((m @ sx_top) / (m @ n_top) - (m @ sx_all) / (m @ n_all)) / sd
    pt = lf(np.ones(len(pit)))
    reps = np.stack([lf(np.bincount(rng.integers(0, len(pit), len(pit)), minlength=len(pit)).astype(float)) for _ in range(B)])
    zone = d.loc[top, FEATURES].quantile([.25, .5, .75])
    return {"point": pt, "lo": np.nanpercentile(reps, 2.5, axis=0), "hi": np.nanpercentile(reps, 97.5, axis=0),
            "zone": zone, "median_all": d[FEATURES].median(), "n": len(d), "n_pitchers": len(pit)}


def gap_boot(score, outcome, groups, B, rng):
    ok = ~np.isnan(outcome)
    score, outcome, groups = score[ok], outcome[ok], groups[ok]
    hi, lo = score >= np.quantile(score, 0.8), score <= np.quantile(score, 0.2)
    pit = pd.Index(pd.unique(groups)); codes = pit.get_indexer(groups)
    S = np.stack([np.bincount(codes, weights=hi.astype(float), minlength=len(pit)), np.bincount(codes, weights=outcome * hi, minlength=len(pit)),
                  np.bincount(codes, weights=lo.astype(float), minlength=len(pit)), np.bincount(codes, weights=outcome * lo, minlength=len(pit))], 1)
    g = lambda m: (m @ S[:, 1]) / (m @ S[:, 0]) - (m @ S[:, 3]) / (m @ S[:, 2])
    reps = [g(np.bincount(rng.integers(0, len(pit), len(pit)), minlength=len(pit)).astype(float)) for _ in range(B)]
    return g(np.ones(len(pit))), np.percentile(reps, 2.5), np.percentile(reps, 97.5)


def factors():
    r21 = json.load(open(BASE_DIR / "21_transform_r.json"))
    gr = pd.read_csv(BASE_DIR / "20_altitude_gradient.csv")
    med = gr[(gr.AutoPitchType == "todos") & (gr.entorno == "Medium Altitude")].set_index("variable")["r"]
    return {w: {"Extreme Altitude": {"InducedVertBreak": v["r_IVB"][0], "HorzBreak": v["r_HB"][0]},
                "Medium Altitude": {"InducedVertBreak": float(med["InducedVertBreak"]), "HorzBreak": float(med["HorzBreak"])}}
            for w, v in r21.items()}


def to_env(frame, from_env, to_env_, fac):
    """Lleva el movimiento de from_env a to_env_ (factores relativos a No Altitude)."""
    f = frame.copy()
    for var, dcol in [("InducedVertBreak", "ivb_diff_vs_fb"), ("HorzBreak", "hb_diff_vs_fb")]:
        r_from = 1.0 if from_env == "No Altitude" else fac[from_env][var]
        r_to = 1.0 if to_env_ == "No Altitude" else fac[to_env_][var]
        k = r_to / r_from
        f[var] = f[var] * k; f[dcol] = f[dcol] * k
    return f


def project(entry, cfg, fac):
    """Nivel = Stuff+ oficial a nivel del mar; efecto de altitud = variante fisica (titular, guia 38).
    stuff_harp_helu = stuff_nivel_mar + delta_fisica. Se conserva la proyeccion oficial como sensibilidad."""
    out = {}
    F = entry["future_frame"]; ars = entry["f_ars"]; alt = F["altitude_category"].astype(str).to_numpy()
    for variant, force in [("oficial", False), ("fisica", True)]:
        sc = t21.StuffScorer(entry, cfg, force_base=force)
        sea, harp = np.full(len(F), np.nan), np.full(len(F), np.nan)
        for env in ["No Altitude", "Medium Altitude", "Extreme Altitude"]:
            m = alt == env
            if m.any():
                sea[m] = sc.score(to_env(F[m], env, "No Altitude", fac), ars[m])
                harp[m] = sc.score(to_env(F[m], env, "Extreme Altitude", fac), ars[m])
        out[variant] = (sea, harp)
        if variant == "oficial":
            obs = sc.score(F, ars)
    d_fis = out["fisica"][1] - out["fisica"][0]
    return pd.DataFrame({"PitchUID": F["PitchUID"].to_numpy(), "pitcher_anon_id": F["pitcher_anon_id"].to_numpy(),
                         "AutoPitchType": F["AutoPitchType"].astype(str).to_numpy(), "altitude_category": alt,
                         "information_level": np.where(ars, "arsenal_historico", "sin_arsenal"),
                         "stuff_observado": obs, "stuff_nivel_mar": out["oficial"][0], "stuff_harp_helu": out["oficial"][0] + d_fis,
                         "delta_fisica": d_fis, "stuff_harp_helu_oficial": out["oficial"][1],
                         "delta_oficial": out["oficial"][1] - out["oficial"][0]})


def simulate_pitch(pitch, fastball=None, measured_in="No Altitude", window=VALID, _cache={}):
    """Stuff+ de un pitch hipotetico a nivel del mar y en Harp Helu (proxy Extreme).

    pitch: dict con AutoPitchType, PitcherThrows ('Right'/'Left'), RelSpeed (mph), SpinRate (rpm),
      SpinAxis (grados, convencion Trackman), RelHeight, RelSide, Extension (ft), InducedVertBreak,
      HorzBreak (in, con el signo crudo de Trackman). Opcional: pitch_usage_pct (0-1).
    fastball: dict opcional con RelSpeed, SpinRate, InducedVertBreak, HorzBreak, RelHeight, RelSide,
      Extension de la recta primaria (mismo entorno de medicion). Si se da, se usan las features de
      arsenal (variante 'arsenal'); si no, la variante base.
    measured_in: entorno donde se midio el pitch ('No Altitude', 'Medium Altitude', 'Extreme Altitude').
    """
    if "bundle" not in _cache:
        mods = t16.load_modules(); cfg = t18.build_cfg(mods)
        b = joblib.load(BASE_DIR / "18_three_scores_bundle.pkl")
        _cache.update(bundle=b, cfg=cfg, fac=factors())
    entry, cfg, fac = _cache["bundle"][window], _cache["cfg"], _cache["fac"][window]
    F = entry["future_frame"]
    row = {c: np.nan for c in F.columns}
    row.update(pitch)
    has_ars = fastball is not None and "pitch_usage_pct" in pitch
    if fastball is not None:
        row["velo_diff_vs_fb"] = pitch["RelSpeed"] - fastball["RelSpeed"]
        row["ivb_diff_vs_fb"] = pitch["InducedVertBreak"] - fastball["InducedVertBreak"]
        row["hb_diff_vs_fb"] = pitch["HorzBreak"] - fastball["HorzBreak"]
        row["spin_diff_vs_fb"] = pitch["SpinRate"] - fastball["SpinRate"]
        row["extension_diff_vs_fb"] = pitch["Extension"] - fastball["Extension"]
        row["release_distance_vs_fb"] = float(np.hypot(pitch["RelHeight"] - fastball["RelHeight"], pitch["RelSide"] - fastball["RelSide"]))
    df = pd.DataFrame([row])
    for col in F.columns:
        if isinstance(F[col].dtype, pd.CategoricalDtype):
            df[col] = pd.Categorical(df[col].astype(object), categories=F[col].cat.categories)
    df = t16.mirror(df)
    ars = np.array([has_ars])
    sc = t21.StuffScorer(entry, cfg)
    out = {}
    for env, key in [("No Altitude", "nivel_mar"), ("Extreme Altitude", "harp_helu")]:
        x = to_env(df, measured_in, env, fac)
        out[f"stuff_plus_{key}"] = float(sc.score(x, ars)[0])
        out[f"IVB_{key}"] = float(x["InducedVertBreak"].iloc[0])
        out[f"HB_{key}_convencion_diestro"] = float(x["HorzBreak"].iloc[0])
    out["delta_stuff_plus"] = out["stuff_plus_harp_helu"] - out["stuff_plus_nivel_mar"]
    out["variante"] = "arsenal" if has_ars else "base"
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--boot", type=int, default=300); ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(); rng = np.random.default_rng(args.seed); pd.set_option("display.width", 240)
    mods = t16.load_modules(); cfg = t18.build_cfg(mods)
    bundle = joblib.load(BASE_DIR / "18_three_scores_bundle.pkl")
    scores = pd.read_csv(BASE_DIR / "18_three_scores.csv", usecols=["window", "PitchUID", "stuff_plus"])
    frames = {}
    for w, e in bundle.items():
        f = e["future_frame"].copy()
        s = scores[scores.window == w]["stuff_plus"].to_numpy()
        if len(s) != len(f):
            raise ValueError("18_three_scores.csv y el bundle no estan alineados.")
        f["stuff_plus"] = s; f["type"] = f["AutoPitchType"].astype(str); f["alt"] = f["altitude_category"].astype(str)
        frames[w] = f
    raw = pd.read_parquet(BASE_DIR / "stuff_model_features_v1.parquet", columns=["PitchUID", "is_whiff", "outside_strike_zone", "swung_outside_strike_zone"])
    raw["chase"] = np.where(raw["outside_strike_zone"] == 1, raw["swung_outside_strike_zone"], np.nan)
    lab = raw.set_index("PitchUID")
    learn, valid = frames[LEARN], frames[VALID]
    types = [t for t, g in learn[learn.alt == "Extreme Altitude"].groupby("type") if len(g) >= MIN_ROWS and g.pitcher_anon_id.nunique() >= MIN_PITCHERS]
    drows = pd.read_parquet(BASE_DIR / "21_delta_rows.parquet"); drows = drows[(drows.window == VALID) & (drows.variante == "fisica")]
    oof = pd.read_parquet(BASE_DIR / "19_cv_oof.parquet", columns=["pitcher_anon_id", "year", "AutoPitchType", "stuff_plus"])

    # ---------------- D5 ----------------
    prof, val = [], []
    for t in sorted(types):
        sel = lambda fr, env: fr[(fr.type == t) & (fr.alt == env)]
        L, V, Nn = lifts(sel(learn, "Extreme Altitude"), "stuff_plus", args.boot, rng), lifts(sel(valid, "Extreme Altitude"), "stuff_plus", args.boot, rng), lifts(sel(learn, "No Altitude"), "stuff_plus", args.boot, rng)
        dt = drows[drows.AutoPitchType == t].merge(valid[["PitchUID"] + [c for c in FEATURES if c not in drows]], on="PitchUID", how="left")
        for j, f in enumerate(FEATURES):
            sig = (L["lo"][j] > 0 or L["hi"][j] < 0) and abs(L["point"][j]) >= MIN_LIFT
            rep = np.sign(V["point"][j]) == np.sign(L["point"][j]) and (V["lo"][j] > 0 or V["hi"][j] < 0)
            no_sig = Nn["lo"][j] > 0 or Nn["hi"][j] < 0
            ok = dt[[f, "delta"]].dropna()
            res = spearmanr(ok[f], ok["delta"])[0] if len(ok) > 50 else np.nan
            prof.append({"AutoPitchType": t, "feature": f, "lift_extreme_2025": L["point"][j], "lift_lo": L["lo"][j], "lift_hi": L["hi"][j],
                         "lift_extreme_2026": V["point"][j], "lift_2026_lo": V["lo"][j], "lift_2026_hi": V["hi"][j],
                         "lift_no_altitude_2025": Nn["point"][j], "en_perfil": bool(sig and rep),
                         "entorno": ("compartido con nivel del mar" if no_sig and np.sign(Nn["point"][j]) == np.sign(L["point"][j]) else
                                     "opuesto en nivel del mar" if no_sig else "especifico de altitud") if sig and rep else "",
                         "zona_p25": L["zone"].loc[.25, f], "zona_p50": L["zone"].loc[.5, f], "zona_p75": L["zone"].loc[.75, f],
                         "mediana_tipo": L["median_all"][f], "resiliencia_spearman": res,
                         "n_learn": L["n"], "n_pitchers_learn": L["n_pitchers"]})
        # Validacion con resultados reales en Extreme 2026.
        v = sel(valid, "Extreme Altitude")
        for oc in ["is_whiff", "chase"]:
            y = lab.loc[v["PitchUID"], oc].to_numpy(float)
            g, lo, hi = gap_boot(v["stuff_plus"].to_numpy(), y, v["pitcher_anon_id"].to_numpy(), args.boot, rng)
            val.append({"AutoPitchType": t, "resultado": oc, "brecha_quintil_alto_menos_bajo_pp": 100 * g, "ci95_lo": 100 * lo, "ci95_hi": 100 * hi, "n": len(v)})
    prof, val = pd.DataFrame(prof), pd.DataFrame(val)
    g = oof.groupby(["pitcher_anon_id", "AutoPitchType", "year"]).agg(n=("stuff_plus", "size"), s=("stuff_plus", "mean")).reset_index()
    g = g[g.n >= 50]
    stab = []
    for t in sorted(types):
        rs = []
        for y0, y1 in [(2024, 2025), (2025, 2026)]:
            a = g[(g.AutoPitchType == t) & (g.year == y0)].set_index("pitcher_anon_id")["s"]; b = g[(g.AutoPitchType == t) & (g.year == y1)].set_index("pitcher_anon_id")["s"]
            ix = a.index.intersection(b.index)
            rs.append((np.corrcoef(a[ix], b[ix])[0, 1], len(ix)))
        stab.append({"AutoPitchType": t, "anio_a_anio_24_25": rs[0][0], "n_24_25": rs[0][1], "anio_a_anio_25_26": rs[1][0], "n_25_26": rs[1][1]})
    stab = pd.DataFrame(stab)
    prof.to_csv(BASE_DIR / "22_pitch_type_profiles.csv", index=False); val.to_csv(BASE_DIR / "22_profile_outcome_validation.csv", index=False)
    stab.to_csv(BASE_DIR / "22_stuff_stability_by_type.csv", index=False)
    print("PERFIL (Extreme; caracteristicas que entran al perfil):")
    print(prof[prof.en_perfil][["AutoPitchType", "feature", "lift_extreme_2025", "lift_extreme_2026", "zona_p25", "zona_p75", "mediana_tipo", "entorno", "resiliencia_spearman"]].round(3).to_string(index=False))
    print("\nResiliencia (Spearman caracteristica vs delta; + = menos perdida), |rho| >= 0.10:")
    print(prof[prof.resiliencia_spearman.abs() >= 0.10][["AutoPitchType", "feature", "resiliencia_spearman"]].round(3).to_string(index=False))
    print("\nValidacion con resultados reales en Extreme 2026 (quintil alto - bajo de Stuff+, pp):")
    print(val.round(2).to_string(index=False))
    print("\nEstabilidad de Stuff+ por tipo (pitcher x tipo, >= 50 pitches):"); print(stab.round(3).to_string(index=False))

    # ---------------- Proyeccion ----------------
    fac = factors()
    proj = pd.concat([project(bundle[w], cfg, fac[w]).assign(window=w) for w in bundle])
    proj.to_parquet(BASE_DIR / "22_pitch_projection.parquet", index=False)
    proj = proj.dropna(subset=["stuff_nivel_mar", "stuff_harp_helu"])  # sin altitude_category valida no se proyecta
    pt = proj.groupby(["window", "pitcher_anon_id", "AutoPitchType"]).agg(
        n=("PitchUID", "size"), stuff_observado=("stuff_observado", "mean"), stuff_nivel_mar=("stuff_nivel_mar", "mean"),
        stuff_harp_helu=("stuff_harp_helu", "mean"), share_extreme=("altitude_category", lambda s: (s == "Extreme Altitude").mean())).reset_index()
    pt["delta_harp_vs_mar"] = pt["stuff_harp_helu"] - pt["stuff_nivel_mar"]
    pt = pt[pt.n >= MIN_CELL]
    pp = proj.groupby(["window", "pitcher_anon_id"]).agg(n=("PitchUID", "size"), stuff_nivel_mar=("stuff_nivel_mar", "mean"),
                                                          stuff_harp_helu=("stuff_harp_helu", "mean")).reset_index()
    pp["delta_harp_vs_mar"] = pp["stuff_harp_helu"] - pp["stuff_nivel_mar"]
    pp = pp[pp.n >= 100]
    pp["rank_harp_helu"] = pp.groupby("window")["stuff_harp_helu"].rank(ascending=False)
    pt.to_csv(BASE_DIR / "22_pitcher_arsenal_projection.csv", index=False); pp.to_csv(BASE_DIR / "22_pitcher_projection.csv", index=False)
    q = pp[pp.window == VALID]
    print(f"\nProyeccion por pitcher (2026, >= 100 pitches): {len(q)} pitchers; delta medio {q.delta_harp_vs_mar.mean():+.2f}, "
          f"rango {q.delta_harp_vs_mar.min():+.2f} a {q.delta_harp_vs_mar.max():+.2f}; corr(ranking mar, ranking Harp) = "
          f"{spearmanr(q.stuff_nivel_mar, q.stuff_harp_helu)[0]:.3f}")

    # ---------------- D6 ----------------
    ev = pd.read_csv(BASE_DIR / "21_evidence_levels.csv").set_index("AutoPitchType")
    bt_all = pd.read_csv(BASE_DIR / "21_delta_by_pitch_type.csv")
    bt = bt_all[(bt_all.window == VALID) & (bt_all.variante == "fisica")].set_index("AutoPitchType")
    bt25 = bt_all[(bt_all.window == LEARN) & (bt_all.variante == "fisica")].set_index("AutoPitchType")
    bt_of = bt_all[(bt_all.window == VALID) & (bt_all.variante == "oficial")].set_index("AutoPitchType")
    gr = pd.read_csv(BASE_DIR / "20_altitude_gradient.csv"); gr = gr[gr.entorno == "Extreme Altitude"]
    ce = pd.read_csv(BASE_DIR / "18_contact_env_coherence.csv"); ce = ce[ce.grupo == "pitch_type"].set_index("nivel")["mean"]
    rec = []
    for t in ev.index:
        e = ev.loc[t]; b = bt.loc[t]
        dI = gr[(gr.AutoPitchType == t) & (gr.variable == "InducedVertBreak")]["delta"].iloc[0]
        dH = gr[(gr.AutoPitchType == t) & (gr.variable == "HorzBreak")]["delta"].iloc[0]
        p_ = prof[(prof.AutoPitchType == t) & prof.en_perfil]
        perfil = "; ".join(f"{r.feature} {r.zona_p25:.1f} a {r.zona_p75:.1f} (mediana del tipo {r.mediana_tipo:.1f})" for r in p_.itertuples()) or "sin caracteristica que cumpla el criterio"
        rz = prof[(prof.AutoPitchType == t) & (prof.resiliencia_spearman.abs() >= 0.10)]
        resil = "; ".join(f"{'mas' if r.resiliencia_spearman > 0 else 'menos'} {r.feature} pierde menos (rho {r.resiliencia_spearman:+.2f})" for r in rz.itertuples()) or "sin rasgo claro"
        lvl = e.nivel_evidencia
        d25, d_of = bt25.loc[t, "delta_stuff"], bt_of.loc[t, "delta_stuff"]
        obs = b.delta_observado_within
        obs_txt = "sin contraste observado suficiente" if obs != obs else f"observado en el mismo pitcher {obs:+.1f}"
        verb = "pierde" if b.delta_stuff < 0 else "gana"
        base_txt = (f"{t}: en Harp Helu {verb} {abs(b.delta_stuff):.1f} puntos de Stuff+ [{b.ci95_lo:+.1f}, {b.ci95_hi:+.1f}] "
                    f"porque el aire reduce su movimiento (IVB {dI:+.1f} in, HB {dH:+.1f} in, mismo pitcher); {obs_txt}. ")
        if lvl == "Robusto":
            txt = base_txt + ("Modelo y observacion coinciden: descontarlo al evaluar este pitch para CDMX." if abs(b.delta_stuff) >= 1
                              else "Efecto consistente pero pequeño.")
        elif lvl == "Condicional":
            txt = base_txt + "La observacion va en la misma direccion pero con incertidumbre: usar como ajuste parcial, no como criterio de exclusion."
        elif e.celdas_min >= 30 and max(abs(b.delta_stuff), abs(d25)) < 0.7:
            txt = (f"{t}: efecto de altitud practicamente nulo en su Stuff+ (2026 {b.delta_stuff:+.1f}, 2025 {d25:+.1f}; {obs_txt}). "
                   "No descontarlo.")
        else:
            txt = f"{t}: no concluir (pocas celdas within-pitcher: {int(e.celdas_min)}). Estimacion puntual {b.delta_stuff:+.1f}."
        if abs(d_of - b.delta_stuff) >= 0.7:
            txt += (f" Sensibilidad: con el modelo que usa el arsenal ({d_of:+.1f}) tambien se penaliza que su separacion contra la recta "
                    "se encoja; eso no aparece en lo observado.")
        rec.append({"AutoPitchType": t, "nivel_evidencia": lvl, "delta_stuff_2026": b.delta_stuff, "ci95_lo": b.ci95_lo, "ci95_hi": b.ci95_hi,
                    "delta_observado_within": b.delta_observado_within, "delta_2025": d25, "delta_oficial_2026": d_of,
                    "delta_ivb_in": dI, "delta_hb_in": dH,
                    "dano_contacto_extra_runs_100": ce.get(t, np.nan), "perfil_extreme": perfil, "resiliencia": resil, "recomendacion": txt})
    rec = pd.DataFrame(rec); rec.to_csv(BASE_DIR / "22_recommendations.csv", index=False)
    print("\nRECOMENDACIONES:")
    for r in rec.itertuples():
        print(f"- [{r.nivel_evidencia}] {r.recomendacion}\n    perfil: {r.perfil_extreme}\n    resiliencia: {r.resiliencia}")

    # ---------------- D7 ----------------
    ex = bundle[VALID]["future_frame"]
    cand = ex[(~bundle[VALID]["f_ars"]) & ex["InducedVertBreak"].notna() & ex["HorzBreak"].notna()
              & ex["altitude_category"].astype(str).eq("Medium Altitude").to_numpy()]
    ok = cand.iloc[0]
    raw_row = pd.read_parquet(BASE_DIR / "stuff_model_features_v1.parquet").set_index("PitchUID").loc[ok["PitchUID"]]
    sim = simulate_pitch({c: raw_row[c] for c in ["AutoPitchType", "PitcherThrows", "RelSpeed", "SpinRate", "SpinAxis", "RelHeight",
                                                  "RelSide", "Extension", "InducedVertBreak", "HorzBreak"]}, measured_in=str(ok["altitude_category"]))
    ref = proj[(proj.window == VALID) & (proj.PitchUID == ok["PitchUID"])].iloc[0]
    if (abs(sim["stuff_plus_nivel_mar"] - ref.stuff_nivel_mar) > 1e-6 or abs(sim["stuff_plus_harp_helu"] - ref.stuff_harp_helu_oficial) > 1e-6
            or abs(sim["delta_stuff_plus"]) < 1e-9):
        raise ValueError(f"simulate_pitch no reproduce la proyeccion: {sim} vs {ref.to_dict()}")
    print(f"\nsimulate_pitch verificado contra la proyeccion oficial (pitch {ok['PitchUID']}): {sim}")
    files = ["18_three_scores_bundle.pkl", "18_three_scores.csv", "18_weights.csv", "20_transform_params.csv", "20_altitude_gradient.csv",
             "21_delta_by_pitch_type.csv", "21_delta_by_shape_bin.csv", "21_evidence_levels.csv", "21_hypotheses.csv", "21_transform_r.json",
             "22_pitch_type_profiles.csv", "22_recommendations.csv", "22_pitcher_projection.csv", "22_pitcher_arsenal_projection.csv",
             "22_pitch_projection.parquet", "17_contact_by_altitude.csv", "19_exit_table.csv"]
    man = {"congelado": "2026-09-26", "ventana_modelo_dashboard": VALID, "escala": "Stuff+ = 100 + 10 z (referencia: historial 2024-2025)",
           "factores_movimiento_vs_No_Altitude": fac[VALID], "entorno_proxy_harp_helu": "Extreme Altitude",
           "simulador": "22_profile_recommendations_export.simulate_pitch",
           "archivos": {f: hashlib.sha256(open(BASE_DIR / f, "rb").read()).hexdigest() for f in files if (BASE_DIR / f).exists()}}
    json.dump(man, open(BASE_DIR / "22_dashboard_manifest.json", "w"), indent=2, ensure_ascii=False)
    print("Manifiesto: 22_dashboard_manifest.json")


if __name__ == "__main__":
    main()
