"""V2 paso 1: sensibilidad del delta total de altitud al peso relativo de groundball.

No modifica el score V1 (pesos iguales). Usa las replicas bootstrap por pitcher
guardadas por 11_altitude_total_effect.py (11_altitude_total_reps.pkl), asi que
no re-simula nada.

Como el score es lineal en los z de los cinco componentes, el delta total por
pitch type es exactamente
    delta(w) = suma_{k != groundball} c_k + w * c_groundball,
con c_k la contribucion de cada componente a peso 1 (puntos Stuff+). w es el
peso de groundball relativo a 1 = el peso de cada uno de los otros cuatro
componentes. w = 1 es V1. El "punto de quiebre" es el w donde delta(w) = 0.
Se reporta su IC95% bootstrap solo cuando c_groundball es estable en signo en
todas las replicas; si no, el quiebre no esta bien definido y se marca.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
GB = "groundball"
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
COMPONENTS = ["whiff_per_pitch", "whiff_given_swing", "groundball", "called_strike", "chase"]


def contributions(reps, arr):
    """arr: (..., K) deltas de probabilidad -> contribucion en puntos Stuff+ a peso 1."""
    comps = reps["comps"]
    pp_std = reps["pp5"][1]
    out = {}
    for n in COMPONENTS:
        k = comps.index(n)
        std = reps["component_stats"][n][1]
        out[n] = arr[..., k] * 10 / (std * len(COMPONENTS) * pp_std)
    return out


def main():
    bundle = joblib.load(BASE_DIR / "11_altitude_total_reps.pkl")
    curve_rows, be_rows = [], []
    pd.set_option("display.width", 200)
    for window, reps in bundle.items():
        types = reps["types"]
        c_rep = contributions(reps, reps["rep_type"])
        c_pt = contributions(reps, reps["point_type"])
        for t, name in enumerate(types):
            if reps["n_rows"][t] == 0:
                continue
            others_rep = sum(c_rep[n][:, t] for n in COMPONENTS if n != GB)
            others_pt = sum(c_pt[n][t] for n in COMPONENTS if n != GB)
            gb_rep, gb_pt = c_rep[GB][:, t], c_pt[GB][t]
            for w in WEIGHTS:
                d = others_rep + w * gb_rep
                curve_rows.append({"window": window, "AutoPitchType": name, "gb_weight": w,
                                   "delta": others_pt + w * gb_pt,
                                   "ci95_lo": np.nanpercentile(d, 2.5), "ci95_hi": np.nanpercentile(d, 97.5)})
            stable = (gb_rep > 0).all() or (gb_rep < 0).all()
            be_pt = -others_pt / gb_pt if gb_pt != 0 else np.nan
            if stable:
                be = -others_rep / gb_rep
                lo, hi = np.nanpercentile(be, 2.5), np.nanpercentile(be, 97.5)
            else:
                lo = hi = np.nan
            if gb_pt > 0:
                side = "positivo si w > quiebre" if others_pt < 0 else "positivo con cualquier w >= 0"
            else:
                side = "negativo si w > quiebre" if others_pt > 0 else "negativo con cualquier w >= 0"
            be_rows.append({"window": window, "AutoPitchType": name, "contrib_otros_w1": others_pt,
                            "contrib_groundball_w1": gb_pt, "gb_weight_breakeven": be_pt,
                            "breakeven_ci95_lo": lo, "breakeven_ci95_hi": hi,
                            "gb_sign_stable_in_bootstrap": bool(stable),
                            "n_cells_history": int(reps["n_cells_history"][t]), "lectura": side})
    curve = pd.DataFrame(curve_rows)
    be = pd.DataFrame(be_rows)
    curve.to_csv(BASE_DIR / "12_gb_weight_curve.csv", index=False)
    be.to_csv(BASE_DIR / "12_gb_weight_breakeven.csv", index=False)
    f = lambda x: f"{x:.2f}"
    for window in bundle:
        print(f"\n{'=' * 80}\n{window}: delta total (puntos Stuff+) vs peso relativo de groundball (1 = V1)\n{'=' * 80}")
        pv = curve[curve["window"] == window].pivot(index="AutoPitchType", columns="gb_weight", values="delta")
        print(pv.to_string(float_format=f))
        print("\nPunto de quiebre (w donde delta = 0) e IC95% bootstrap:")
        print(be[be["window"] == window].drop(columns="window").to_string(index=False, float_format=f))
    print("\nArchivos: 12_gb_weight_curve.csv, 12_gb_weight_breakeven.csv")


if __name__ == "__main__":
    main()
