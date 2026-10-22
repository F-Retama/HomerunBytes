"""Efecto TOTAL modelado de la altitud sobre Stuff+ (bloque separado de 10).

Tres resultados que se mantienen separados en la narrativa:
  1. delta_direct (10): cambiar solo altitude_category. ~0.
  2. delta observacional (10): contraste real within-pitcher/type/year. Ruidoso.
  3. delta_total modelado (este script): estimacion principal.

delta_total = Score(X^Extreme, Extreme) - Score(X^No, No): "si este shape se
transportara a Extreme Altitude (proxy del entorno Harp Helu) y su fisica
cambiara como historicamente cambia ese tipo de pitch, cuanto cambiaria su
valoracion". No es causalidad pura. Ubicacion, count, arsenal historico y
handedness se mantienen fijos (en la realidad la ubicacion de llegada tambien
cambiaria; no se modela).

Transformacion de entorno (decisiones de diseno)
  - Solo se desplazan las variables que la atmosfera puede cambiar EN VUELO:
    InducedVertBreak, HorzBreak, VertApprAngle, HorzApprAngle. Los diferenciales
    de arsenal ivb_diff_vs_fb y hb_diff_vs_fb se desplazan por
    delta_tipo - delta_recta_primaria del pitcher; el resto no cambia.
  - Variables de release (velocidad, spin, altura/lado de release, extension,
    angulos de release) las controla el pitcher: NO se desplazan. Se estiman
    sus deltas within-pitcher como PLACEBO: si no son ~0, indican calibracion o
    composicion de parques y no efecto del aire.
  - El vector de deltas se estima JUNTO por tipo de pitch, con las mismas
    celdas pitcher x tipo x anio (No vs Extreme, n>=20 en ambos, ponderadas por
    el menor n), solo con el HISTORIAL de cada ventana (fold-safe), y se valida
    contra los deltas observados en la temporada futura. Es una traslacion por
    tipo: no captura efectos proporcionales por pitch (p.ej. mas spin, mas
    perdida). No se hace shrinkage entre tipos: diluiria diferencias reales; la
    incertidumbre la lleva el bootstrap.
  - HorzBreak, HorzApprAngle y RelSide tienen signo absoluto (zurdos ~ -diestros).
    La transformacion se estima y se aplica en convencion de DIESTRO: se reflejan
    esas variables en zurdos antes de agregar y el desplazamiento se aplica con el
    signo de lateralidad de cada pitch (y de la recta en hb_diff_vs_fb). Pitches
    con lateralidad no Right/Left se excluyen. (Una version previa mezclaba signos;
    corregido el 25/09/2026.)
  - Cohorte principal: pitches observados en No Altitude de la temporada
    evaluada; la copia Extreme = X + delta. Chequeo inverso: pitches observados
    en Extreme, transformados hacia No con -delta.

Incertidumbre: bootstrap agrupado por pitcher. Cada replicate remuestrea
pitchers (con multiplicidad) y aplica el mismo remuestreo a las celdas que
estiman la transformacion y a la cohorte evaluada; la unidad independiente no
es el pitch. Regla de titular: solo se reporta "gana/pierde" si el IC95% no
cruza 0; en otro caso "incierto".
"""

import argparse
import importlib.util
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
SHIFT_VARS = ["InducedVertBreak", "HorzBreak", "VertApprAngle", "HorzApprAngle"]
PLACEBO_VARS = ["RelSpeed", "SpinRate", "RelHeight", "RelSide", "Extension",
                "VertRelAngle", "HorzRelAngle"]
ALL_VARS = SHIFT_VARS + PLACEBO_VARS
WINDOW_HISTORY = {"2024 -> 2025": [2024], "2024-2025 -> 2026": [2024, 2025]}
PRIMARY = "2024-2025 -> 2026"
MIN_CELL = 20
HORZ_VARS = ["HorzBreak", "HorzApprAngle", "RelSide"]


def handed_sign(series):
    return series.astype(str).map({"Right": 1.0, "Left": -1.0})
MIN_PITCHER_PITCHES = 100
MIN_RELEVANT_POINTS = 1.0
MIN_BIN_PITCHERS = 30


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, BASE_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_cells(df):
    d = df[df["altitude_category"].astype(str).isin(["No Altitude", "Extreme Altitude"])].copy()
    sign = handed_sign(d["PitcherThrows"])
    d = d[sign.notna()].copy()
    for col in HORZ_VARS:
        d[col] = d[col] * sign[sign.notna()]
    d["AutoPitchType"] = d["AutoPitchType"].astype(str)
    d["altitude_category"] = d["altitude_category"].astype(str)
    g = d.groupby(["pitcher_anon_id", "AutoPitchType", "year", "altitude_category"])
    agg = g[ALL_VARS].mean()
    agg["n"] = g.size()
    agg = agg.reset_index()
    wide = agg.pivot(index=["pitcher_anon_id", "AutoPitchType", "year"],
                     columns="altitude_category", values=ALL_VARS + ["n"]).dropna()
    keep = (wide[("n", "No Altitude")] >= MIN_CELL) & (wide[("n", "Extreme Altitude")] >= MIN_CELL)
    wide = wide[keep]
    dX = np.column_stack([wide[(v, "Extreme Altitude")] - wide[(v, "No Altitude")] for v in ALL_VARS])
    w0 = np.minimum(wide[("n", "No Altitude")], wide[("n", "Extreme Altitude")]).to_numpy(float)
    keys = wide.index.to_frame(index=False)
    return keys, dX, w0


def estimate_transform(type_idx, pitch_idx, w0, dX, mult, n_types):
    w = w0 * mult[pitch_idx]
    den = np.bincount(type_idx, weights=w, minlength=n_types)
    num = np.stack([np.bincount(type_idx, weights=w * dX[:, j], minlength=n_types)
                    for j in range(dX.shape[1])], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den[:, None]


class Cohort:
    def __init__(self, frame, types, pitchers):
        self.frame = frame.reset_index(drop=True)
        self.type_idx = pd.Index(types).get_indexer(self.frame["AutoPitchType"].astype(str))
        self.fb_idx = pd.Index(types).get_indexer(self.frame["primary_fastball_type"].astype(str))
        self.pitch_idx = pd.Index(pitchers).get_indexer(self.frame["pitcher_anon_id"])
        self.sign = handed_sign(self.frame["PitcherThrows"]).to_numpy()


def shifted_frame(coh, D, sign):
    d = D[:, :4] * sign
    own, fb = d[coh.type_idx].copy(), d[coh.fb_idx].copy()
    for arr in (own, fb):
        arr[:, [1, 3]] *= coh.sign[:, None]
    valid = ~(np.isnan(own).any(axis=1) | np.isnan(fb).any(axis=1) | np.isnan(coh.sign))
    own0, fb0 = np.nan_to_num(own), np.nan_to_num(fb)
    f = coh.frame.copy()
    for j, col in enumerate(SHIFT_VARS):
        f[col] = f[col] + own0[:, j]
    f["ivb_diff_vs_fb"] = f["ivb_diff_vs_fb"] + own0[:, 0] - fb0[:, 0]
    f["hb_diff_vs_fb"] = f["hb_diff_vs_fb"] + own0[:, 1] - fb0[:, 1]
    return f, valid


def deltas(t10, entry, coh, base, D, sign):
    f, valid = shifted_frame(coh, D, sign)
    if sign > 0:
        sc = t10.score_at_level(entry, f, "Extreme Altitude")
        sgn = 1.0
    else:
        sc = t10.score_at_level(entry, f, "No Altitude")
        sgn = -1.0
    out = {"stuff": sgn * (sc["stuff"] - base["stuff"]),
           "no_chase": sgn * (sc["stuff_no_chase"] - base["stuff_no_chase"])}
    for c in t10.COMPONENTS:
        out[c] = sgn * (sc["probs"][c] - base["probs"][c])
    return out, valid, f


def group_mean(idx, weights, values, n):
    num = np.bincount(idx, weights=weights * values, minlength=n)
    den = np.bincount(idx, weights=weights, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def decision(lo, hi):
    if lo > 0:
        return "gana Stuff+"
    if hi < 0:
        return "pierde Stuff+"
    return "incierto (IC cruza 0)"


def run_window(t10, entry, raw, label, n_boot, seed):
    hist = raw[raw["year"].isin(WINDOW_HISTORY[label]) & raw["altitude_category"].notna()]
    keys, dX, w0 = build_cells(hist)
    frame = entry["future_frame"]
    alt = frame["altitude_category"].astype(str)
    types = sorted(set(keys["AutoPitchType"]) | set(frame["AutoPitchType"].astype(str)))
    pitchers = np.unique(np.concatenate([keys["pitcher_anon_id"].to_numpy(), frame["pitcher_anon_id"].to_numpy()]))
    T = len(types)
    cell_type = pd.Index(types).get_indexer(keys["AutoPitchType"])
    cell_pitch = pd.Index(pitchers).get_indexer(keys["pitcher_anon_id"])
    ones = np.ones(len(pitchers))
    D0 = estimate_transform(cell_type, cell_pitch, w0, dX, ones, T)
    n_cells = np.bincount(cell_type, minlength=T)

    coh = Cohort(frame[alt == "No Altitude"], types, pitchers)
    base = t10.score_at_level(entry, coh.frame, "No Altitude")
    coh.frame = t10.add_shape_bins(coh.frame.copy())
    bins = pd.Index(sorted(coh.frame["shape_bin"].unique()))
    bin_idx = bins.get_indexer(coh.frame["shape_bin"])

    comps = ["stuff", "no_chase"] + t10.COMPONENTS
    K = len(comps)
    d0, valid0, f0 = deltas(t10, entry, coh, base, D0, +1)
    v0 = valid0.astype(float)
    point_type = np.stack([group_mean(coh.type_idx, v0, d0[k], T) for k in comps], axis=1)
    point_bin = group_mean(bin_idx, v0, d0["stuff"], len(bins))
    sc_phys = t10.score_at_level(entry, f0, "No Altitude")
    phys = sc_phys["stuff"] - base["stuff"]
    label_only = (t10.score_at_level(entry, f0, "Extreme Altitude")["stuff"] - sc_phys["stuff"])

    rng = np.random.default_rng(seed)
    rep_type = np.full((n_boot, T, K), np.nan)
    rep_bin = np.full((n_boot, len(bins)), np.nan)
    rep_D = np.full((n_boot, T, len(ALL_VARS)), np.nan)
    for b in range(n_boot):
        mult = np.bincount(rng.integers(0, len(pitchers), len(pitchers)), minlength=len(pitchers)).astype(float)
        Db = estimate_transform(cell_type, cell_pitch, w0, dX, mult, T)
        rep_D[b] = Db
        db, vb, _ = deltas(t10, entry, coh, base, Db, +1)
        w = mult[coh.pitch_idx] * vb
        for k, name in enumerate(comps):
            rep_type[b, :, k] = group_mean(coh.type_idx, w, np.where(vb, db[name], 0.0), T)
        rep_bin[b] = group_mean(bin_idx, w, np.where(vb, db["stuff"], 0.0), len(bins))
        if (b + 1) % 25 == 0:
            print(f"  bootstrap {b + 1}/{n_boot}", flush=True)

    ci = lambda a: (np.nanpercentile(a, 2.5, axis=0), np.nanpercentile(a, 97.5, axis=0))
    rows = []
    for t in range(T):
        sel = valid0 & (coh.type_idx == t)
        if sel.sum() == 0:
            continue
        lo, hi = ci(rep_type[:, t, 0])
        row = {"window": label, "AutoPitchType": types[t], "n_rows": int(sel.sum()),
               "n_pitchers": int(coh.frame.loc[sel, "pitcher_anon_id"].nunique()),
               "n_cells_history": int(n_cells[t]),
               "mean_delta_total": point_type[t, 0], "median_delta_total": float(np.median(d0["stuff"][sel])),
               "ci95_lo": lo, "ci95_hi": hi, "sd_boot": float(np.nanstd(rep_type[:, t, 0])),
               "prop_boot_pos": float(np.nanmean(rep_type[:, t, 0] > 0)),
               "decision": decision(lo, hi),
               "mean_delta_physical": float(phys[sel].mean()), "mean_delta_label": float(label_only[sel].mean())}
        for k, name in enumerate(comps):
            clo, chi = ci(rep_type[:, t, k])
            col = "no_chase" if name == "no_chase" else ("stuff" if name == "stuff" else t10.COMPONENT_LABELS[name])
            if name != "stuff":
                row[f"d_{col}"] = point_type[t, k]
                row[f"d_{col}_lo"], row[f"d_{col}_hi"] = clo, chi
        row["relevante_>=1pt"] = bool(abs(point_type[t, 0]) >= MIN_RELEVANT_POINTS)
        # Contribucion exacta en puntos Stuff+ (el score es lineal en los z): permite ver
        # que componente empuja el delta y probar sensibilidad a los pesos iguales.
        consts = {n: 10 / (entry["component_stats"][n][1] * len(t10.COMPONENTS) * entry["pp5"][1])
                  for n in t10.COMPONENTS}
        for n in t10.COMPONENTS:
            k = comps.index(n)
            lab = t10.COMPONENT_LABELS[n]
            row[f"contrib_{lab}"] = point_type[t, k] * consts[n]
            row[f"contrib_{lab}_lo"], row[f"contrib_{lab}_hi"] = (
                np.nanpercentile(rep_type[:, t, k] * consts[n], 2.5), np.nanpercentile(rep_type[:, t, k] * consts[n], 97.5))
        gb = comps.index("groundball")
        rep_nogb = rep_type[:, t, 0] - rep_type[:, t, gb] * consts["groundball"]
        nlo, nhi = np.nanpercentile(rep_nogb, 2.5), np.nanpercentile(rep_nogb, 97.5)
        row["delta_no_groundball"] = point_type[t, 0] - point_type[t, gb] * consts["groundball"]
        row["delta_no_groundball_lo"], row["delta_no_groundball_hi"] = nlo, nhi
        row["decision_no_groundball"] = decision(nlo, nhi)
        row["robusto_a_quitar_groundball"] = row["decision"] == row["decision_no_groundball"]
        rows.append(row)
    by_type = pd.DataFrame(rows)

    blo, bhi = ci(rep_bin)
    bin_rows = []
    for i, name in enumerate(bins):
        sel = valid0 & (bin_idx == i)
        if sel.sum() == 0:
            continue
        bin_rows.append({"window": label, "shape_bin": name, "n_rows": int(sel.sum()),
                         "n_pitchers": int(coh.frame.loc[sel, "pitcher_anon_id"].nunique()),
                         "mean_delta_total": point_bin[i], "ci95_lo": blo[i], "ci95_hi": bhi[i],
                         "prop_boot_pos": float(np.nanmean(rep_bin[:, i] > 0)),
                         "decision": decision(blo[i], bhi[i])})
    by_bin = pd.DataFrame(bin_rows)

    pf = coh.frame.assign(delta_total=d0["stuff"], valid=valid0)
    pf = pf[pf["valid"]].groupby("pitcher_anon_id")["delta_total"].agg(["count", "mean", "median", "std"]).reset_index()
    by_pitcher = pf[pf["count"] >= MIN_PITCHER_PITCHES].assign(window=label)

    dlo, dhi = ci(rep_D)
    trows = []
    for t in range(T):
        for j, v in enumerate(ALL_VARS):
            trows.append({"window": label, "AutoPitchType": types[t], "variable": v,
                          "role": "shift" if v in SHIFT_VARS else "placebo",
                          "delta_extreme_minus_no": D0[t, j], "ci95_lo": dlo[t, j], "ci95_hi": dhi[t, j],
                          "n_cells_history": int(n_cells[t])})
    transform = pd.DataFrame(trows)

    fk, fdX, fw0 = build_cells(frame)
    ftype = pd.Index(types).get_indexer(fk["AutoPitchType"])
    fpitch = pd.Index(pitchers).get_indexer(fk["pitcher_anon_id"])
    Df = estimate_transform(ftype, fpitch, fw0, fdX, ones, T)
    vrows = []
    for t in range(T):
        for j, v in enumerate(SHIFT_VARS):
            vrows.append({"window": label, "AutoPitchType": types[t], "variable": v,
                          "history_estimate": D0[t, j], "future_observed": Df[t, j],
                          "n_cells_future": int(np.bincount(ftype, minlength=T)[t])})
    validation = pd.DataFrame(vrows)

    rev = Cohort(frame[alt == "Extreme Altitude"], types, pitchers)
    rbase = t10.score_at_level(entry, rev.frame, "Extreme Altitude")
    dr, vr, _ = deltas(t10, entry, rev, rbase, D0, -1)
    rev_type = group_mean(rev.type_idx, vr.astype(float), dr["stuff"], T)
    reverse = pd.DataFrame({"window": label, "AutoPitchType": types,
                            "forward_mean": point_type[:, 0], "reverse_mean": rev_type})
    reps = {"types": types, "comps": comps, "rep_type": rep_type, "point_type": point_type,
            "component_stats": entry["component_stats"], "pp5": entry["pp5"],
            "n_cells_history": n_cells, "n_rows": np.array([int((valid0 & (coh.type_idx == t)).sum()) for t in range(T)])}
    return by_type, by_bin, by_pitcher, transform, validation, reverse, reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    pd.set_option("display.width", 220)
    t10 = _load("altitude_counterfactual", "10_altitude_counterfactual.py")
    bundle_path = BASE_DIR / "09_pitching_plus_bundle.pkl"
    if not bundle_path.exists():
        raise FileNotFoundError("Falta 09_pitching_plus_bundle.pkl: correr 09_pitching_plus.py primero.")
    bundle = joblib.load(bundle_path)
    if "primary_fastball_type" not in next(iter(bundle.values()))["future_frame"]:
        raise ValueError("El bundle es anterior a este script: volver a correr 09_pitching_plus.py.")
    raw = pd.read_parquet(BASE_DIR / "stuff_model_features_v1.parquet",
                          columns=["pitcher_anon_id", "AutoPitchType", "year", "altitude_category",
                                   "is_whiff", "PitcherThrows"] + ALL_VARS)
    raw = raw.dropna(subset=["is_whiff", "pitcher_anon_id"])

    reps_all = {}
    out = {k: [] for k in ["type", "bin", "pitcher", "transform", "validation", "reverse"]}
    for label, entry in bundle.items():
        print(f"\n{'=' * 80}\nDELTA TOTAL {label} | bootstrap B={args.bootstrap} por pitcher\n{'=' * 80}", flush=True)
        res = run_window(t10, entry, raw, label, args.bootstrap, args.seed)
        for key, r in zip(out, res):
            out[key].append(r)
        bt, bb, bp, tr, va, rv = res[:6]
        reps_all[label] = res[6]
        f = lambda x: f"{x:.2f}"
        cols = ["AutoPitchType", "n_rows", "n_pitchers", "n_cells_history", "mean_delta_total",
                "median_delta_total", "ci95_lo", "ci95_hi", "prop_boot_pos", "decision", "relevante_>=1pt"]
        print("\nDelta total por AutoPitchType (puntos Stuff+; IC95% bootstrap por pitcher):")
        print(bt[cols].to_string(index=False, float_format=f))
        cc = ["AutoPitchType"] + [f"contrib_{v}" for v in t10.COMPONENT_LABELS.values()] + \
             ["delta_no_groundball", "delta_no_groundball_lo", "delta_no_groundball_hi",
              "decision_no_groundball", "robusto_a_quitar_groundball"]
        print("\nContribucion de cada componente al delta total (puntos Stuff+, pesos iguales) y sensibilidad a quitar groundball:")
        print(bt[cc].to_string(index=False, float_format=f))
        print("\nDescomposicion (media): fisica desplazada vs etiqueta de altitud; sin chase; componentes (prob, pp):")
        dc = ["AutoPitchType", "mean_delta_physical", "mean_delta_label", "d_no_chase"] + \
             [c for c in bt.columns if c.startswith("d_p_") and not c.endswith(("_lo", "_hi"))]
        show = bt[dc].copy()
        for c in show.columns:
            if c.startswith("d_p_"):
                show[c] = show[c] * 100
        print(show.to_string(index=False, float_format=f))
        print("\nTransformacion estimada en historial (Extreme - No), variables desplazadas:")
        print(tr[tr["role"] == "shift"].pivot(index="AutoPitchType", columns="variable",
              values="delta_extreme_minus_no").to_string(float_format=f))
        print("Placebo (release, no desplazadas; deberian ser ~0):")
        print(tr[tr["role"] == "placebo"].pivot(index="AutoPitchType", columns="variable",
              values="delta_extreme_minus_no").to_string(float_format=lambda x: f"{x:.3f}"))
        print("\nValidacion: historial vs observado en la temporada futura:")
        print(va.pivot(index="AutoPitchType", columns="variable",
              values=["history_estimate", "future_observed"]).to_string(float_format=f))
        print("\nChequeo inverso (Extreme observado -> No) vs directo:")
        print(rv.to_string(index=False, float_format=f))
        sig = bb[(bb["decision"] != "incierto (IC cruza 0)") & (bb["n_pitchers"] >= MIN_BIN_PITCHERS)
                 & (bb["mean_delta_total"].abs() >= MIN_RELEVANT_POINTS)]
        print(f"\nShape bins con IC que no cruza 0, >= {MIN_BIN_PITCHERS} pitchers y |delta| >= {MIN_RELEVANT_POINTS} pt: "
              f"{len(sig)} de {len(bb)} (sin correccion por comparaciones multiples)")
        if len(sig):
            print(sig.sort_values("mean_delta_total").to_string(index=False, float_format=f))
        print(f"Pitchers n>={MIN_PITCHER_PITCHES}: {len(bp)} | delta medio {bp['mean'].mean():.2f} | SD entre pitchers {bp['mean'].std():.2f}")

    for key, name in [("type", "by_pitch_type"), ("bin", "by_shape_bin"), ("pitcher", "by_pitcher"),
                      ("transform", "transform"), ("validation", "transform_validation"),
                      ("reverse", "reverse_check")]:
        pd.concat(out[key], ignore_index=True).to_csv(BASE_DIR / f"11_altitude_total_{name}.csv", index=False)
    joblib.dump(reps_all, BASE_DIR / "11_altitude_total_reps.pkl")
    print("\nArchivos: 11_altitude_total_{by_pitch_type,by_shape_bin,by_pitcher,transform,transform_validation,reverse_check}.csv")


if __name__ == "__main__":
    main()
