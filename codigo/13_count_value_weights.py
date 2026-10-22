"""V2 pasos 2-5: valor por conteo (cadena de Markov) y pesos del compuesto.

Objetivo acotado: contestar si, al reemplazar pesos iguales por pesos ligados
a valor de carrera observable, cambia la conclusion ambiental por pitch type.
NO es un run expectancy de 24 estados (el dataset no tiene estado base-out) ni
usa RunsScored (en bolas en juego promedia 0.25 en sencillo y 0.02 en out: no
valora outs ni corredores en base). V1 (pesos iguales) se mantiene intacto.

Valor por conteo
  - 12 estados (Balls 0-3 x Strikes 0-2). Balls/Strikes son el conteo PREVIO al
    pitch (verificado: 99.95% de los strikeouts ocurren con 2 strikes y 99.98%
    de las bases por bolas con 3 bolas).
  - Transiciones estimadas empiricamente por conteo desde PitchCall, solo con
    el HISTORIAL de cada ventana (fold-safe): bola, strike, foul (con 2 strikes
    no cambia el conteo), bola en juego, HBP. No requiere reconstruir
    plate appearances.
  - Los eventos terminales reciben un pago de referencia externa. Excepcion a
    la regla "sin pesos externos" de la guia (seccion 22), autorizada el
    24/09/2026 SOLO para eventos terminales, cuando el dataset no contiene el
    estado base-out para estimarlos internamente; NO se usan para ponderar
    directamente las cinco probabilidades. Fuente: tabla de run values de The
    Book / FanGraphs 2006 (https://insidethebook.com/ee/index.php/site/comments/run_values_of_events/):
    1B .47, 2B .77, 3B 1.09, HR 1.39, BB .32, HBP .35, K -.30, out en roletazo
    -.24, out en elevado -.28. Supuestos propios (no estan en esa tabla): error
    = .47 (como sencillo); FieldersChoice y Sacrificio = out segun hit_type; out
    sin hit_type = -.26 (promedio simple); bunt = roletazo. Son constantes de
    otro entorno de carreras (MLB 2006), aproximaciones, no valores de LMB.
  - Valor realizado por pitch, desde la perspectiva del pitcher:
    y = -(V(estado siguiente o pago terminal) - V(conteo actual)). Por
    construccion E[y | conteo] ~ 0 (se verifica fuera de muestra).

Pesos
  - Regresion de y sobre los cinco z de 09 (en el altitude_category real de
    cada pitch), no negativa y con ridge leve, centrada en la cohorte. Los
    pesos se normalizan a suma 1 para ser comparables con V1 (1/5 cada uno).
  - Cada ventana aprende con SUS propias predicciones fuera de muestra (los
    modelos de 09 nunca vieron esa temporada) y sus propios pesos de conteo.
    Estabilidad: se comparan ventanas y, ademas, los pesos de 2025 se evaluan
    hacia adelante en 2026.
  - Bootstrap agrupado por pitcher (via estadisticos suficientes por pitcher).
  - RMSE contra el baseline nulo (predecir y=0 = xRunValue promedio por conteo,
    que es 0 por construccion): el chequeo de calidad que pide el criterio 1.
"""

import argparse
import importlib.util
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import nnls

BASE_DIR = Path(__file__).resolve().parent
COMPONENTS = ["whiff_per_pitch", "whiff_given_swing", "groundball", "called_strike", "chase"]
LEVELS = ["No Altitude", "Medium Altitude", "Extreme Altitude"]
WINDOW_HISTORY = {"2024 -> 2025": [2024], "2024-2025 -> 2026": [2024, 2025]}
PAY = {"BB": 0.32, "HBP": 0.35, "K": -0.30, "Single": 0.47, "Double": 0.77, "Triple": 1.09,
       "HomeRun": 1.39, "Error": 0.47, "GroundOut": -0.24, "AirOut": -0.28, "OutUnknown": -0.26}
STABLE_MAX_DIFF = 0.10


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, BASE_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def classify(df):
    pc = df["PitchCall"].astype(str)
    b = df["Balls"].astype(int).to_numpy()
    s = df["Strikes"].astype(int).to_numpy()
    cls = np.select(
        [pc.isin(["BallCalled", "BallinDirt"]), pc.isin(["StrikeCalled", "StrikeSwinging"]),
         pc.isin(["FoulBall", "FoulBallNotFieldable", "FoulBallFieldable"]), pc.eq("InPlay"),
         pc.eq("HitByPitch")], ["ball", "strike", "foul", "bip", "hbp"], default="other")
    pr, ht = df["play_result"].astype(str), df["hit_type"].astype(str)
    out_like = pr.isin(["Out", "FieldersChoice", "Sacrifice"])
    bip_pay = np.select(
        [pr.eq("Single"), pr.eq("Double"), pr.eq("Triple"), pr.eq("HomeRun"), pr.eq("Error"),
         out_like & ht.isin(["GroundBall", "Bunt"]), out_like & ht.isin(["FlyBall", "LineDrive", "Popup"]), out_like],
        [PAY["Single"], PAY["Double"], PAY["Triple"], PAY["HomeRun"], PAY["Error"],
         PAY["GroundOut"], PAY["AirOut"], PAY["OutUnknown"]], default=np.nan)
    cls = np.where((cls == "bip") & np.isnan(bip_pay), "other", cls)
    pay = np.select([cls == "bip", cls == "hbp", (cls == "ball") & (b == 3), (cls == "strike") & (s == 2)],
                    [bip_pay, PAY["HBP"], PAY["BB"], PAY["K"]], default=np.nan)
    nb = np.where(cls == "ball", b + 1, b)
    ns = np.where(cls == "strike", s + 1, np.where(cls == "foul", np.minimum(s + 1, 2), s))
    terminal = ~np.isnan(pay)
    return pd.DataFrame({"PitchUID": df["PitchUID"].to_numpy(), "b": b, "s": s, "cls": cls, "pay": pay,
                         "nb": np.where(terminal, -1, nb), "ns": np.where(terminal, -1, ns)})


def fit_count_value(t):
    t = t[t["cls"] != "other"]
    P, r = np.zeros((12, 12)), np.zeros(12)
    for b in range(4):
        for s in range(3):
            m = t[(t["b"] == b) & (t["s"] == s)]
            i, n = b * 3 + s, len(m)
            f = m["cls"].value_counts() / n
            if b < 3:
                P[i, (b + 1) * 3 + s] += f.get("ball", 0)
            else:
                r[i] += f.get("ball", 0) * PAY["BB"]
            if s < 2:
                P[i, b * 3 + s + 1] += f.get("strike", 0)
            else:
                r[i] += f.get("strike", 0) * PAY["K"]
            P[i, b * 3 + min(s + 1, 2)] += f.get("foul", 0)
            r[i] += f.get("bip", 0) * m.loc[m["cls"] == "bip", "pay"].mean() if f.get("bip", 0) > 0 else 0
            r[i] += f.get("hbp", 0) * PAY["HBP"]
    return np.linalg.solve(np.eye(12) - P, r)


def realized_value(t, V):
    cur = V[t["b"].to_numpy() * 3 + t["s"].to_numpy()]
    nxt_state = np.where(t["nb"].to_numpy() >= 0, V[np.maximum(t["nb"].to_numpy(), 0) * 3 + np.maximum(t["ns"].to_numpy(), 0)], np.nan)
    nxt = np.where(np.isnan(t["pay"].to_numpy()), nxt_state, t["pay"].to_numpy())
    y = -(nxt - cur)
    return np.where(t["cls"].to_numpy() == "other", np.nan, y)


def actual_z(t10, entry):
    frame = entry["future_frame"]
    alt = frame["altitude_category"].astype(str).to_numpy()
    per = {lvl: t10.score_at_level(entry, frame, lvl) for lvl in LEVELS}
    Z = np.column_stack([
        (np.select([alt == l for l in LEVELS], [per[l]["probs"][n] for l in LEVELS]) - entry["component_stats"][n][0])
        / entry["component_stats"][n][1] for n in COMPONENTS])
    return frame, Z


def nn_ridge(G, c, rel_alpha=0.01):
    A = G + rel_alpha * np.mean(np.diag(G)) * np.eye(len(c))
    R = np.linalg.cholesky(A).T
    return nnls(R, np.linalg.solve(R.T, c))[0]


def pitcher_stats(codes, Z, y, n_p):
    G = np.zeros((n_p, 5, 5)); c = np.zeros((n_p, 5)); n = np.bincount(codes, minlength=n_p).astype(float)
    np.add.at(G, codes, Z[:, :, None] * Z[:, None, :])
    np.add.at(c, codes, Z * y[:, None])
    return G, c, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    pd.set_option("display.width", 200)
    t10 = _load("altitude_counterfactual", "10_altitude_counterfactual.py")
    bundle = joblib.load(BASE_DIR / "09_pitching_plus_bundle.pkl")
    raw = pd.read_parquet(BASE_DIR / "stuff_model_features_v1.parquet",
                          columns=["PitchUID", "year", "Balls", "Strikes", "PitchCall", "play_result", "hit_type"])
    tall = classify(raw)
    tall["year"] = raw["year"].to_numpy()
    print(f"Pitches clasificados: {len(tall):,} | excluidos ('other'/sin pago): {(tall['cls'] == 'other').sum():,}")

    data, table_rows = {}, []
    for label, entry in bundle.items():
        hist = tall[tall["year"].isin(WINDOW_HISTORY[label])]
        V = fit_count_value(hist)
        frame, Z = actual_z(t10, entry)
        cohort = tall.set_index("PitchUID").loc[frame["PitchUID"]].reset_index()
        y = realized_value(cohort, V)
        ok = ~np.isnan(y)
        for b in range(4):
            for s in range(3):
                m = ok & (cohort["b"].to_numpy() == b) & (cohort["s"].to_numpy() == s)
                table_rows.append({"window": label, "balls": b, "strikes": s,
                                   "V_offense": V[b * 3 + s], "n_cohort": int(m.sum()),
                                   "mean_y_cohort": float(y[m].mean()) if m.sum() else np.nan})
        print(f"\n{label}: V(0-0)={V[0]:+.3f} V(3-0)={V[9]:+.3f} V(0-2)={V[2]:+.3f} V(3-2)={V[11]:+.3f} "
              f"(perspectiva ofensiva, runs) | media y cohorte={np.nanmean(y):+.5f}, filas con y: {int(ok.sum()):,}/{len(y):,}")
        pit = pd.Index(frame.loc[ok, "pitcher_anon_id"].unique())
        codes = pit.get_indexer(frame.loc[ok, "pitcher_anon_id"])
        data[label] = {"Z": Z[ok], "y": y[ok], "codes": codes, "n_p": len(pit)}
        print("Correlacion entre componentes (z):")
        print(pd.DataFrame(Z[ok], columns=COMPONENTS).corr().round(2).to_string())
    pd.DataFrame(table_rows).to_csv(BASE_DIR / "13_count_value_table.csv", index=False)

    rng = np.random.default_rng(args.seed)
    fits = {}
    for label, d in data.items():
        Z, y = d["Z"], d["y"]
        zbar, ybar = Z.mean(axis=0), y.mean()
        Zc, yc = Z - zbar, y - ybar
        G, c, n = pitcher_stats(d["codes"], Zc, yc, d["n_p"])
        w = nn_ridge(G.sum(0), c.sum(0))
        e = np.full(5, 0.2)
        a1 = (e @ c.sum(0)) / (e @ G.sum(0) @ e)
        draws = np.zeros((args.bootstrap, 5)); draws_raw = np.zeros((args.bootstrap, 5))
        for b in range(args.bootstrap):
            mult = np.bincount(rng.integers(0, d["n_p"], d["n_p"]), minlength=d["n_p"]).astype(float)
            wb = nn_ridge(np.tensordot(mult, G, 1), mult @ c)
            draws_raw[b] = wb
            draws[b] = wb / wb.sum() if wb.sum() > 0 else np.nan
        fits[label] = {"w": w, "w_norm": w / w.sum(), "a1": a1, "zbar": zbar, "draws": draws,
                       "draws_raw": draws_raw, "G": G, "c": c, "n": n}

    rows = []
    for label, f in fits.items():
        lo, hi = np.nanpercentile(f["draws"], 2.5, axis=0), np.nanpercentile(f["draws"], 97.5, axis=0)
        for k, name in enumerate(COMPONENTS):
            rows.append({"window": label, "component": name, "weight_v1": 0.2, "weight_v2": f["w_norm"][k],
                         "ci95_lo": lo[k], "ci95_hi": hi[k], "sd_boot": float(np.nanstd(f["draws"][:, k]))})
    wdf = pd.DataFrame(rows)
    print("\nPESOS V2 (normalizados a suma 1; V1 = 0.20 cada uno):")
    print(wdf.pivot(index="component", columns="window", values="weight_v2").round(3).to_string())
    a, b = list(fits)
    la, lb = wdf[wdf.window == a].set_index("component"), wdf[wdf.window == b].set_index("component")
    overlap = (la["ci95_lo"] <= lb["ci95_hi"]) & (lb["ci95_lo"] <= la["ci95_hi"])
    maxdiff = float((la["weight_v2"] - lb["weight_v2"]).abs().max())
    stable = bool(overlap.all() and maxdiff <= STABLE_MAX_DIFF)
    print(f"\nEstabilidad entre ventanas: IC95% solapan en todos={bool(overlap.all())}, max |diferencia|={maxdiff:.3f} "
          f"(umbral interno {STABLE_MAX_DIFF}) -> {'ESTABLE' if stable else 'NO ESTABLE: reportar V2 como sensibilidad'}")
    print("IC95% por ventana:")
    print(wdf[["window", "component", "weight_v2", "ci95_lo", "ci95_hi"]].round(3).to_string(index=False))
    wdf["stable_across_windows"] = stable
    wdf.to_csv(BASE_DIR / "13_component_weights.csv", index=False)

    print(f"\n{'=' * 78}\nRMSE FUERA DE MUESTRA del valor realizado por pitch (y, runs desde la perspectiva del pitcher)\n{'=' * 78}")
    perf = []
    for train_l, test_l in [(a, b), (b, a)]:
        f, d = fits[train_l], data[test_l]
        Zt = d["Z"] - f["zbar"]
        preds = {"Null (xRunValue prom. por conteo = 0)": np.zeros(len(d["y"])),
                 "V1 equal-weight": f["a1"] * Zt @ np.full(5, 0.2), "V2 learned": Zt @ f["w"]}
        n_p = d["n_p"]
        sse = {k: np.bincount(d["codes"], weights=(d["y"] - p) ** 2, minlength=n_p) for k, p in preds.items()}
        cnt = np.bincount(d["codes"], minlength=n_p).astype(float)
        base = sse["Null (xRunValue prom. por conteo = 0)"]
        for k in preds:
            mse = sse[k].sum() / cnt.sum()
            ci = np.zeros(2)
            if k != "Null (xRunValue prom. por conteo = 0)":
                gains = []
                for _ in range(1000):
                    m = np.bincount(rng.integers(0, n_p, n_p), minlength=n_p)
                    gains.append(1 - (m @ sse[k]) / (m @ base))
                ci = np.percentile(gains, [2.5, 97.5])
            perf.append({"weights_learned_on": train_l, "evaluated_on": test_l, "model": k,
                         "rmse": float(np.sqrt(mse)), "mse_reduction_vs_null": 1 - sse[k].sum() / base.sum(),
                         "ci95_lo": ci[0], "ci95_hi": ci[1], "forward_in_time": train_l == a})
        v1, v2 = sse["V1 equal-weight"], sse["V2 learned"]
        gains = []
        for _ in range(1000):
            m = np.bincount(rng.integers(0, n_p, n_p), minlength=n_p)
            gains.append(1 - (m @ v2) / (m @ v1))
        perf.append({"weights_learned_on": train_l, "evaluated_on": test_l, "model": "V2 vs V1 (reduccion de MSE)",
                     "rmse": np.nan, "mse_reduction_vs_null": 1 - v2.sum() / v1.sum(),
                     "ci95_lo": np.percentile(gains, 2.5), "ci95_hi": np.percentile(gains, 97.5),
                     "forward_in_time": train_l == a})
    pdf = pd.DataFrame(perf)
    print(pdf.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    pdf.to_csv(BASE_DIR / "13_v2_out_of_sample.csv", index=False)

    v2 = {}
    for label, f in fits.items():
        Z = data[label]["Z"]
        v2[label] = {"weights_norm": f["w_norm"], "weights_draws_norm": f["draws"], "components": COMPONENTS,
                     "sd_raw_v1": float((Z @ np.full(5, 0.2)).std()), "sd_raw_v2": float((Z @ f["w_norm"]).std()),
                     "Z": Z.astype(np.float32),
                     "stable_across_windows": stable}
    joblib.dump(v2, BASE_DIR / "13_v2_bundle.pkl")
    print("\nArchivos: 13_count_value_table.csv, 13_component_weights.csv, 13_v2_out_of_sample.csv, 13_v2_bundle.pkl")


if __name__ == "__main__":
    main()
