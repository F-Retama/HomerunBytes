"""V2 paso 6: matriz V1 vs V2 del efecto total de altitud y niveles de evidencia.

Recombina las replicas bootstrap de 11 (delta de cada componente por pitch type,
en probabilidad) con distintos vectores de pesos, sin re-simular:
  delta_puntos = 10 * sum_k w_k * (dp_k / std_k) / SD(compuesto en la cohorte)
Esquemas: V1 (pesos iguales, baseline intacto); V2 propio (pesos aprendidos en
esa ventana, con sus draws bootstrap emparejados con las replicas de altitud);
V2 cruzado (pesos de la otra ventana); V2-stuff (pesos V2 renormalizados solo
entre los componentes fisicos: whiff, whiff|swing, groundball). Todos se escalan
con la SD del compuesto sobre la misma cohorte, para que los puntos sean
comparables entre esquemas (por eso V1 difiere levemente de 11, que uso la SD
de train).

Niveles de evidencia (criterio propio, explicito; los umbrales son internos):
  Robusto: en AMBAS ventanas V1, V2 propio y V2 cruzado tienen IC95% que excluye
    0 con el mismo signo, el chequeo inverso de 11 coincide, y >= 30 celdas de
    historial.
  Condicional: hay senal consistente entre ventanas en al menos un esquema (IC
    excluye 0, mismo signo en ambas) pero el signo cambia con el esquema de
    pesos, o algun IC cruza 0.
  Incierto: < 30 celdas de historial en alguna ventana, o el signo no es
    estable entre ventanas en ningun esquema.
Relevancia practica >= 1 punto (0.1 SD) es un umbral interno de analisis, no
del brief.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
COMPONENTS = ["whiff_per_pitch", "whiff_given_swing", "groundball", "called_strike", "chase"]
STUFF_IDX = [0, 1, 2]
MIN_CELLS = 30
RELEVANT = 1.0


def points(reps, t, w, sd):
    idx = [reps["comps"].index(n) for n in COMPONENTS]
    std = np.array([reps["component_stats"][n][1] for n in COMPONENTS])
    dp = reps["rep_type"][:, t, :][:, idx] / std
    dp0 = reps["point_type"][t, idx] / std
    if w.ndim == 1:
        return 10 * float(dp0 @ w) / sd, 10 * (dp @ w) / sd
    w0 = w.mean(axis=0)
    return 10 * float(dp0 @ w0) / sd, 10 * np.einsum("bk,bk->b", dp, w) / sd


def stuff_only(w):
    v = np.zeros_like(w, dtype=float)
    v[..., STUFF_IDX] = w[..., STUFF_IDX]
    s = v.sum(axis=-1, keepdims=True)
    return np.where(s > 0, v / np.where(s > 0, s, 1), np.nan)


def summarize(pt, rep):
    lo, hi = np.nanpercentile(rep, 2.5), np.nanpercentile(rep, 97.5)
    return pt, lo, hi, (1 if lo > 0 else -1 if hi < 0 else 0)


def main():
    pd.set_option("display.width", 220)
    reps_all = joblib.load(BASE_DIR / "11_altitude_total_reps.pkl")
    v2 = joblib.load(BASE_DIR / "13_v2_bundle.pkl")
    rev = pd.read_csv(BASE_DIR / "11_altitude_total_reverse_check.csv")
    windows = list(reps_all)
    rows = []
    for wi, window in enumerate(windows):
        reps, own = reps_all[window], v2[window]
        other = v2[windows[1 - wi]]
        Z = own["Z"].astype(float)
        eq = np.full(5, 0.2)
        schemes = {
            "V1 (pesos iguales)": (eq, Z),
            "V2 propio": (own["weights_draws_norm"], own["weights_norm"]),
            "V2 cruzado": (other["weights_draws_norm"], other["weights_norm"]),
            "V2-stuff": (stuff_only(own["weights_draws_norm"]), stuff_only(own["weights_norm"])),
        }
        for t, name in enumerate(reps["types"]):
            if reps["n_rows"][t] == 0:
                continue
            row = {"window": window, "AutoPitchType": name, "n_cells_history": int(reps["n_cells_history"][t]),
                   "n_rows": int(reps["n_rows"][t])}
            for label, (wd, wpt) in schemes.items():
                sd = float((Z @ (eq if label.startswith("V1") else np.nan_to_num(wpt))).std())
                w_use = eq if label.startswith("V1") else wd
                pt, rep = points(reps, t, w_use, sd)
                p, lo, hi, sg = summarize(pt, rep)
                key = label.split(" ")[0] + ("_" + label.split(" ")[1] if " " in label and not label.startswith("V1") else "")
                row.update({f"{key}_delta": p, f"{key}_lo": lo, f"{key}_hi": hi, f"{key}_sign": sg})
            wpt = own["weights_norm"]
            sd2 = float((Z @ wpt).std())
            idx = [reps["comps"].index(n) for n in COMPONENTS]
            std = np.array([reps["component_stats"][n][1] for n in COMPONENTS])
            for k, n in enumerate(COMPONENTS):
                row[f"V2_contrib_{n}"] = 10 * wpt[k] * reps["point_type"][t, idx[k]] / std[k] / sd2
            rr = rev[(rev["window"] == window) & (rev["AutoPitchType"] == name)]
            row["reverse_same_sign"] = bool(len(rr) and np.sign(rr["reverse_mean"].iloc[0]) == np.sign(rr["forward_mean"].iloc[0]))
            rows.append(row)
    m = pd.DataFrame(rows)
    m.to_csv(BASE_DIR / "14_v1_v2_matrix_by_window.csv", index=False)

    ev = []
    for name, g in m.groupby("AutoPitchType"):
        g = g.set_index("window").loc[windows]
        def consistent(key):
            s = g[f"{key}_sign"].to_numpy()
            return bool((s != 0).all() and (s == s[0]).all())
        v1c, v2c, xc = consistent("V1"), consistent("V2_propio"), consistent("V2_cruzado")
        same_sign = v1c and v2c and xc and (g["V1_sign"].iloc[0] == g["V2_propio_sign"].iloc[0]
                                            == g["V2_cruzado_sign"].iloc[0])
        few = (g["n_cells_history"] < MIN_CELLS).any()
        if few:
            level = "Incierto"
        elif same_sign and g["reverse_same_sign"].all():
            level = "Robusto"
        elif v1c or v2c or xc:
            level = "Condicional"
        else:
            level = "Incierto"
        prim = g.loc[windows[-1]]
        ev.append({"AutoPitchType": name, "V1_2024-25->2026": prim["V1_delta"], "V2_propio_2024-25->2026": prim["V2_propio_delta"],
                   "V2_cruzado_2024-25->2026": prim["V2_cruzado_delta"], "V2_stuff_2024-25->2026": prim["V2-stuff_delta"],
                   "V1_2024->2025": g.loc[windows[0], "V1_delta"], "V2_propio_2024->2025": g.loc[windows[0], "V2_propio_delta"],
                   "n_cells_min": int(g["n_cells_history"].min()), "nivel_evidencia": level,
                   "relevante_>=1pt_V2": bool(abs(prim["V2_propio_delta"]) >= RELEVANT)})
    evd = pd.DataFrame(ev).sort_values("V2_propio_2024-25->2026")
    evd.to_csv(BASE_DIR / "14_evidence_levels.csv", index=False)

    f = lambda x: f"{x:.2f}"
    print("MATRIZ V1 vs V2 (puntos Stuff+, IC95% bootstrap; escala por SD del compuesto en la cohorte)")
    for window in windows:
        g = m[m["window"] == window]
        print(f"\n{window}")
        cols = ["AutoPitchType", "n_cells_history"]
        for k in ["V1", "V2_propio", "V2_cruzado", "V2-stuff"]:
            cols += [f"{k}_delta", f"{k}_lo", f"{k}_hi"]
        print(g[cols].to_string(index=False, float_format=f))
    print("\nContribucion de cada componente al delta V2 propio (puntos):")
    cc = ["window", "AutoPitchType"] + [f"V2_contrib_{n}" for n in COMPONENTS]
    print(m[cc].to_string(index=False, float_format=f))
    print("\nNIVELES DE EVIDENCIA (ventana primaria 2024-25->2026 y 2024->2025):")
    print(evd.to_string(index=False, float_format=f))
    print("\nArchivos: 14_v1_v2_matrix_by_window.csv, 14_evidence_levels.csv")


if __name__ == "__main__":
    main()
