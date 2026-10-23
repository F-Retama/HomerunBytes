"""Paquete ligero del dashboard para la versión hosteada (el jurado no corre código).

El bundle de 18 pesa ~400 MB porque guarda todos los lanzamientos (132,243 de la ventana 2026 con 114 columnas, más la
ventana 2024 -> 2025 que el dashboard no usa). Este script arma dashboard/lite/engine_lite.pkl con:
  - los modelos de la ventana 2024-2025 -> 2026 (solo los componentes que el dashboard evalúa);
  - una muestra de hasta --k pitches por pitcher y tipo (pitchers seleccionables en Simulator y Game), solo con las
    columnas que el modelo usa: sin resultados (whiff, contacto, carreras), sin PitchUID ni IDs de juego, bateador o catcher;
  - una muestra de --league-n pitches por tipo y mano (dispersión de la liga para prospectos y 'Pitch manual');
  - promedios exactos con TODOS los pitches (punto de partida del Simulator) y la referencia de la liga ya calculada;
  - 25 pitches con sus scores oficiales para verificar que el motor ligero reproduce 18.
Después compara el motor ligero contra el completo y guarda las diferencias en 25_dashboard_lite_check.csv.

Ejecutar desde codigo/ (necesita 18_three_scores_bundle.pkl y dashboard/data):
    python 25_dashboard_lite.py            # k=50, league-n=500
"""

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
from dashboard.lib import data, engine  # noqa: E402

OUT = BASE_DIR / "dashboard" / "lite" / "engine_lite.pkl"
META = ["pitcher_anon_id", "AutoPitchType", "PitcherThrows", "altitude_category"]


def selectable(year=2026):
    """Pitchers y tipos que el Simulator (n >= 50, tipos con n >= 20) y Game (subconjunto) dejan elegir."""
    P, A = data.load("pitchers"), data.load("arsenal")
    pool = P[(P.year == year) & (P.n >= 50) & P.stuff_harp.notna()]
    ars = A[(A.year == year) & A.pitcher_anon_id.isin(pool.pitcher_anon_id) & (A.n >= 20)]
    return list(zip(ars.pitcher_anon_id, ars.AutoPitchType))


def slim_families(E):
    fams = {}
    for f, d in E.fams.items():
        w = np.asarray(d["weights"], float)
        need = {c for c, wi in zip(E.comps, w) if abs(wi) > 1e-9} | set(engine.SHOW)
        if f == "pitching":
            need.add("contact_value")
        models = {}
        for c in E.comps:
            if c not in need:
                continue
            models[c] = {v: ({"p_bip": m.p_bip, "reg": m.reg} if type(m).__name__ == "ContactValueModel" else m) for v, m in d["models"][c].items()}
        fams[f] = {"variants": d["variants"], "weights": w, "stats": d["stats"], "models": models}
    return fams


def build(k, league_n):
    t0 = time.time()
    E = engine.Engine(lite=False)
    cols = list(dict.fromkeys(META + [c for d in E.fams.values() for v in d["variants"].values() for c in v]))
    F = E.F
    alt_ok = F.altitude_category.astype(str).isin(engine.ENVS).to_numpy()

    # muestra por pitcher y tipo + resumen exacto con todos sus pitches
    parts, ars_parts, summary, pop_exact = [], [], {}, {}
    keys = selectable()
    for pid, ptype in keys:
        rows, ars = E.pitcher_rows(pid, ptype)
        if not len(rows):
            continue
        summary[(pid, ptype)] = E.pitcher_summary(pid, ptype)
        for hyb in (True, False):
            pop_exact[((pid, ptype), hyb)] = E.population_table(rows, ars, hybrid=hyb)
        if len(rows) > k:
            idx = np.sort(np.random.default_rng(0).choice(len(rows), k, replace=False))
            rows, ars = rows.iloc[idx], ars[idx]
        parts.append(rows[cols]); ars_parts.append(ars)
    prow = pd.concat(parts, ignore_index=True)
    pars = np.concatenate(ars_parts)

    # dispersión de la liga por tipo y mano + promedio exacto del modo 'Pitch manual'
    combos = F.loc[alt_ok, ["AutoPitchType", "PitcherThrows"]].astype(str).drop_duplicates().itertuples(index=False)
    lparts, manual_base = [], {}
    for ptype, hand in combos:
        rows = E.league_sample(ptype, hand, n=league_n)[0]
        lparts.append(rows[cols])
        manual_base[(ptype, hand)] = E.manual_rows(ptype, hand)[1]
    lrow = pd.concat(lparts, ignore_index=True)

    ref = E.reference_base()

    sc = pd.read_csv(BASE_DIR / "18_three_scores.csv", usecols=["window", "stuff_plus", "location_plus", "pitching_plus"])
    sc = sc[sc.window == engine.WINDOW].reset_index(drop=True)
    idx = np.random.default_rng(0).choice(len(F), 25, replace=False)
    ver = {"rows": F.iloc[idx][cols].reset_index(drop=True), "ars": E.entry["f_ars"][idx],
           **{f: sc[f"{f}_plus"].to_numpy()[idx] for f in ["stuff", "location", "pitching"]}}

    L = {"window": engine.WINDOW, "k": k, "league_n": league_n, "families": slim_families(E), "components": E.comps,
         "signs": dict(E.signs), "fac": E.fac, "cats": E.cats, "ranges": E.ranges,
         "pitcher_rows": prow, "pitcher_ars": pars, "pitcher_summary": summary, "pop_exact": pop_exact,
         "league_rows": lrow, "manual_base": manual_base, "reference_base": ref, "verify": ver}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(L, OUT, compress=3)
    distinct = pd.concat([prow, lrow, ver["rows"]]).drop_duplicates()
    info = {"k": k, "league_n": league_n, "pitcher_types": len(summary), "pitcher_rows": len(prow), "league_rows": len(lrow),
            "distinct_rows_aprox": len(distinct), "full_rows": len(F), "columns": len(cols), "full_columns": F.shape[1],
            "size_mb": round(OUT.stat().st_size / 1e6, 1), "seconds": round(time.time() - t0)}
    print(json.dumps(info, indent=1))
    return E, info


def check(E_full, n_pitchers=40, seed=1):
    """Diferencias del motor ligero contra el completo en lo que ve el usuario."""
    E_lite = engine.Engine(lite=True)
    E_full.__dict__.pop("_samples", None)  # build() dejó en caché las muestras chicas: comparar contra las originales (2000)
    out = []
    v = E_lite.verify()
    out += [{"prueba": f"verify_{f}", "max_abs": d, "mean_abs": np.nan, "n": 25} for f, d in v.items()]

    keys = list(E_lite.L["pitcher_summary"])
    rng = np.random.default_rng(seed)
    pick = [keys[i] for i in rng.choice(len(keys), min(n_pitchers, len(keys)), replace=False)]
    d_pop, d_shift, d_game, d_hyb = [], [], [], []
    for pid, ptype in pick:
        rf, af = E_full.pitcher_rows(pid, ptype)
        rl, al = E_lite.pitcher_rows(pid, ptype)
        a = E_full.population_table(rf, af).set_index("env").stuff_plus
        b = E_lite.population_table(rl, al).set_index("env").stuff_plus
        d_pop.append((a - b).abs().max())
        sh = {"RelSpeed": 1.0, "InducedVertBreak": 1.0}
        a = E_full.population_table(rf, af, sh).set_index("env").stuff_plus
        b = E_lite.population_table(rl, al, sh).set_index("env").stuff_plus
        d_shift.append((a - b).abs().max())
        a = E_full.population_table(rf, af, sh, hybrid=False).set_index("env").stuff_plus
        b = E_lite.population_table(rl, al, sh, hybrid=False).set_index("env").stuff_plus
        d_hyb.append((a - b).abs().max())
    for pid, ptype in pick[:10]:
        ctx = {"BatterSide": "Right", "Balls": 0, "Strikes": 0, "Outs": 0}
        g1 = E_full.game_grid(pid, ptype, ctx, "Extreme Altitude", nx=7, ny=7)["pitching_plus"]
        g2 = E_lite.game_grid(pid, ptype, ctx, "Extreme Altitude", nx=7, ny=7)["pitching_plus"]
        d_game.append(float(np.abs(g1 - g2).mean()))
    out.append({"prueba": "simulator_stuff_actual (Stuff+, por entorno)", "max_abs": max(d_pop), "mean_abs": float(np.mean(d_pop)), "n": len(pick)})
    out.append({"prueba": "simulator_escenario (+1 mph, +1 in IVB, Stuff+)", "max_abs": max(d_shift), "mean_abs": float(np.mean(d_shift)), "n": len(pick)})
    out.append({"prueba": "simulator_escenario_sensibilidad (variante oficial)", "max_abs": max(d_hyb), "mean_abs": float(np.mean(d_hyb)), "n": len(pick)})
    out.append({"prueba": "game_pitching_plus (media de la rejilla)", "max_abs": max(d_game), "mean_abs": float(np.mean(d_game)), "n": len(d_game)})

    d_pro = []
    for ptype, hand in [("Four-Seam", "Right"), ("Slider", "Right"), ("Changeup", "Left"), ("Curveball", "Right"), ("Sinker", "Left")]:
        vals = E_full.league_sample(ptype, hand)[2]
        a = E_full.prospect(ptype, hand, vals).set_index("env").stuff_plus
        b = E_lite.prospect(ptype, hand, vals).set_index("env").stuff_plus
        d_pro.append((a - b).abs().max())
    out.append({"prueba": "scouting_prospecto (Stuff+, por entorno)", "max_abs": max(d_pro), "mean_abs": float(np.mean(d_pro)), "n": len(d_pro)})
    res = pd.DataFrame(out)
    res.to_csv(BASE_DIR / "25_dashboard_lite_check.csv", index=False)
    print(res.to_string(index=False))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=50, help="máximo de pitches guardados por pitcher y tipo")
    ap.add_argument("--league-n", type=int, default=500, help="pitches de la liga por tipo y mano")
    ap.add_argument("--sin-check", action="store_true")
    args = ap.parse_args()
    E, _ = build(args.k, args.league_n)
    if not args.sin_check:
        check(E)
