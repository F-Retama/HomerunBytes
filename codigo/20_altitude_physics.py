"""Bloque D (D1-D2): forma de la transformacion ambiental del movimiento y gradiente por altitud.

Solo se transforman las variables que la atmosfera cambia en vuelo Y que usa el Stuff+ limpio
(18): InducedVertBreak y HorzBreak (y sus diferenciales contra la recta, derivados en 21).
Velocidad, spin y release los controla el pitcher. Horizontales en convencion de diestro.

Celdas within-pitcher: pitcher x tipo x temporada con >= 20 pitches en ambos entornos; medias
por celda; ponderacion por el menor n. Una forma se APRENDE con el historial de cada ventana
(2024 -> 2025; 2024-25 -> 2026) y se EVALUA prediciendo las medias en Extreme de las celdas
de la temporada futura a partir de sus medias en No Altitude.

D1, formas comparadas (Extreme predicho a partir de No):
  sin_cambio         E = N
  aditiva_tipo       E = N + delta_t (por variable)                         2 x 7 parametros
  mult_tipo          E = r_t * N (por variable; minimos cuadrados por el origen) 2 x 7
  mult_global_var    E = r_v * N (un factor por variable, comun a todos los tipos) 2
  mult_global_magnus E = r * N (un solo factor para IVB y HB: el movimiento inducido por
                     spin escala con la densidad del aire)                   1
Regla fijada antes de ver resultados: gana la de menor RMSE ponderado promedio en las dos
ventanas; si otra con menos parametros queda a menos de 2% de ese RMSE, se elige la mas simple.

D2, gradiente: el mismo esquema para Medium - No y Extreme - No (y r por entorno), con IC95%
bootstrap por pitcher. Si Medium cae entre No y Extreme, la lectura ambiental no depende de
una comparacion binaria entre grupos de parques.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
VARS = ["InducedVertBreak", "HorzBreak"]
MIN_CELL = 20
WINDOWS = {"2024 -> 2025": ([2024], 2025), "2024-2025 -> 2026": ([2024, 2025], 2026)}
TOL = 0.02
FORMS = ["sin_cambio", "aditiva_tipo", "mult_tipo", "mult_global_var", "mult_global_magnus"]
N_PARAMS = {"sin_cambio": 0, "aditiva_tipo": 14, "mult_tipo": 14, "mult_global_var": 2, "mult_global_magnus": 1}


def load_raw():
    cols = ["pitcher_anon_id", "AutoPitchType", "year", "altitude_category", "PitcherThrows", "is_whiff"] + VARS
    d = pd.read_parquet(BASE_DIR / "stuff_model_features_v1.parquet", columns=cols).dropna(subset=["is_whiff", "pitcher_anon_id"])
    sign = d["PitcherThrows"].astype(str).map({"Right": 1.0, "Left": -1.0})
    d = d[sign.notna()].copy()
    d["HorzBreak"] = d["HorzBreak"] * sign[sign.notna()]
    d["AutoPitchType"] = d["AutoPitchType"].astype(str)
    d["altitude_category"] = d["altitude_category"].astype(str)
    return d


def cells(d, env):
    """Celdas pitcher x tipo x anio con medias en No y en env."""
    x = d[d["altitude_category"].isin(["No Altitude", env])]
    g = x.groupby(["pitcher_anon_id", "AutoPitchType", "year", "altitude_category"])
    agg = g[VARS].mean(); agg["n"] = g.size()
    w = agg.reset_index().pivot(index=["pitcher_anon_id", "AutoPitchType", "year"], columns="altitude_category",
                                values=VARS + ["n"]).dropna()
    w = w[(w[("n", "No Altitude")] >= MIN_CELL) & (w[("n", env)] >= MIN_CELL)]
    out = w.index.to_frame(index=False)
    for v in VARS:
        out[f"{v}_N"] = w[(v, "No Altitude")].to_numpy(); out[f"{v}_E"] = w[(v, env)].to_numpy()
    out["w"] = np.minimum(w[("n", "No Altitude")], w[("n", env)]).to_numpy(float)
    return out


def fit(c, form, mult=None):
    """Parametros de una forma. mult: multiplicidad bootstrap por fila de celda."""
    w = c["w"].to_numpy() * (1.0 if mult is None else mult)
    types = sorted(c["AutoPitchType"].unique())
    p = {}
    if form == "aditiva_tipo":
        for t in types:
            m = (c["AutoPitchType"] == t).to_numpy()
            for v in VARS:
                p[(t, v)] = np.sum(w[m] * (c[f"{v}_E"] - c[f"{v}_N"]).to_numpy()[m]) / np.sum(w[m])
    elif form == "mult_tipo":
        for t in types:
            m = (c["AutoPitchType"] == t).to_numpy()
            for v in VARS:
                N, E = c[f"{v}_N"].to_numpy()[m], c[f"{v}_E"].to_numpy()[m]
                p[(t, v)] = np.sum(w[m] * N * E) / np.sum(w[m] * N * N)
    elif form == "mult_global_var":
        for v in VARS:
            N, E = c[f"{v}_N"].to_numpy(), c[f"{v}_E"].to_numpy()
            p[v] = np.sum(w * N * E) / np.sum(w * N * N)
    elif form == "mult_global_magnus":
        N = np.concatenate([c[f"{v}_N"].to_numpy() for v in VARS]); E = np.concatenate([c[f"{v}_E"].to_numpy() for v in VARS])
        ww = np.concatenate([w, w])
        p["r"] = np.sum(ww * N * E) / np.sum(ww * N * N)
    return p


def predict(c, form, p):
    out = {}
    for v in VARS:
        N = c[f"{v}_N"].to_numpy(); t = c["AutoPitchType"].to_numpy()
        if form == "sin_cambio":
            out[v] = N
        elif form == "aditiva_tipo":
            out[v] = N + np.array([p.get((ti, v), 0.0) for ti in t])
        elif form == "mult_tipo":
            out[v] = N * np.array([p.get((ti, v), 1.0) for ti in t])
        elif form == "mult_global_var":
            out[v] = N * p[v]
        else:
            out[v] = N * p["r"]
    return out


def wrmse(y, yhat, w):
    return float(np.sqrt(np.sum(w * (y - yhat) ** 2) / np.sum(w)))


def boot(c, stat, B, rng):
    pit = pd.Index(c["pitcher_anon_id"].unique()); codes = pit.get_indexer(c["pitcher_anon_id"])
    reps = []
    for _ in range(B):
        m = np.bincount(rng.integers(0, len(pit), len(pit)), minlength=len(pit)).astype(float)[codes]
        reps.append(stat(m))
    return np.nanpercentile(reps, [2.5, 97.5], axis=0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--boot", type=int, default=500); ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(); rng = np.random.default_rng(args.seed); pd.set_option("display.width", 220)
    d = load_raw()

    # ---------------- D1 ----------------
    rows, params = [], {}
    for label, (hist_years, fut) in WINDOWS.items():
        ch = cells(d[d.year.isin(hist_years)], "Extreme Altitude"); cf = cells(d[d.year == fut], "Extreme Altitude")
        for form in FORMS:
            p = fit(ch, form); params[(label, form)] = p
            pr = predict(cf, form, p)
            for v in VARS:
                rows.append({"window": label, "form": form, "variable": v, "AutoPitchType": "todos", "n_celdas_test": len(cf),
                             "rmse": wrmse(cf[f"{v}_E"].to_numpy(), pr[v], cf["w"].to_numpy())})
                for t in sorted(cf["AutoPitchType"].unique()):
                    m = (cf["AutoPitchType"] == t).to_numpy()
                    rows.append({"window": label, "form": form, "variable": v, "AutoPitchType": t, "n_celdas_test": int(m.sum()),
                                 "rmse": wrmse(cf[f"{v}_E"].to_numpy()[m], pr[v][m], cf["w"].to_numpy()[m])})
    ev = pd.DataFrame(rows)
    ev.to_csv(BASE_DIR / "20_transform_form_eval.csv", index=False)
    overall = ev[ev.AutoPitchType == "todos"].groupby(["form", "variable"])["rmse"].mean().unstack()
    overall["promedio"] = overall.mean(axis=1)
    overall["n_params"] = [N_PARAMS[f] for f in overall.index]
    best = overall["promedio"].idxmin()
    cands = overall[(overall["promedio"] <= overall.loc[best, "promedio"] * (1 + TOL))].sort_values("n_params")
    chosen = cands.index[0]
    print("D1: RMSE ponderado (pulgadas) prediciendo medias en Extreme de celdas futuras, promedio de dos ventanas:")
    print(overall.round(4).to_string())
    print(f"Menor RMSE: {best}; elegida por la regla (<= {TOL:.0%} y menos parametros): {chosen}")
    print("\nPor tipo (ventana 2024-2025 -> 2026):")
    print(ev[(ev.window == "2024-2025 -> 2026") & (ev.AutoPitchType != "todos")].pivot_table(
        index=["AutoPitchType", "variable"], columns="form", values="rmse").round(3).to_string())

    # Parametros finales (historial de cada ventana) con IC bootstrap por pitcher.
    prow = {}
    for label, (hist_years, _f) in WINDOWS.items():
        ch = cells(d[d.year.isin(hist_years)], "Extreme Altitude")
        for form in ["aditiva_tipo", "mult_tipo", "mult_global_var", "mult_global_magnus"]:
            p = fit(ch, form); keys = list(p)
            lo, hi = boot(ch, lambda m: np.array([fit(ch, form, m)[k] for k in keys]), args.boot, rng)
            for k, a, b in zip(keys, lo, hi):
                t, v = (k if isinstance(k, tuple) else ("todos", k))
                prow.setdefault("rows", []).append({"window": label, "form": form, "AutoPitchType": t, "variable": v,
                                                    "param": p[k], "ci95_lo": a, "ci95_hi": b, "n_celdas": len(ch)})
    pp = pd.DataFrame(prow["rows"]); pp.to_csv(BASE_DIR / "20_transform_params.csv", index=False)
    print("\nParametros (IC95% bootstrap por pitcher):")
    print(pp[pp.form.isin(["mult_global_magnus", "mult_global_var"])].round(4).to_string(index=False))

    # ---------------- D2: gradiente ----------------
    grow = []
    for env in ["Medium Altitude", "Extreme Altitude"]:
        c = cells(d, env)
        for t in ["todos"] + sorted(c["AutoPitchType"].unique()):
            cc = c if t == "todos" else c[c["AutoPitchType"] == t]
            if len(cc) < 5:
                continue
            for v in VARS + ["magnus"]:
                def stat(m, cc=cc, v=v):
                    w = cc["w"].to_numpy() * m
                    if v == "magnus":
                        N = np.concatenate([cc[f"{x}_N"].to_numpy() for x in VARS]); E = np.concatenate([cc[f"{x}_E"].to_numpy() for x in VARS])
                        ww = np.concatenate([w, w]); return np.array([np.nan, np.sum(ww * N * E) / np.sum(ww * N * N)])
                    N, E = cc[f"{v}_N"].to_numpy(), cc[f"{v}_E"].to_numpy()
                    return np.array([np.sum(w * (E - N)) / np.sum(w), np.sum(w * N * E) / np.sum(w * N * N)])
                pt = stat(np.ones(len(cc)))
                lo, hi = boot(cc, stat, args.boot, rng)
                grow.append({"entorno": env, "AutoPitchType": t, "variable": v, "n_celdas": len(cc),
                             "delta": pt[0], "delta_lo": lo[0], "delta_hi": hi[0], "r": pt[1], "r_lo": lo[1], "r_hi": hi[1]})
    gr = pd.DataFrame(grow); gr.to_csv(BASE_DIR / "20_altitude_gradient.csv", index=False)
    wide = gr.pivot_table(index=["AutoPitchType", "variable"], columns="entorno", values=["delta", "r"])
    wide[("fraccion_medium", "")] = wide[("delta", "Medium Altitude")] / wide[("delta", "Extreme Altitude")]
    print("\nD2: gradiente within-pitcher (todas las temporadas; delta en pulgadas, r = factor multiplicativo):")
    print(wide.round(3).to_string())

    json.dump({"forma_elegida": chosen, "tolerancia": TOL,
               "rmse_promedio": overall["promedio"].round(5).to_dict()}, open(BASE_DIR / "20_transform_choice.json", "w"), indent=2)
    print(f"\nArchivos: 20_transform_form_eval.csv, 20_transform_params.csv, 20_altitude_gradient.csv, 20_transform_choice.json")


if __name__ == "__main__":
    main()
