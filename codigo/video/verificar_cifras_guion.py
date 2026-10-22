"""Cifras del guion del video leidas con la misma logica de cada vista del dashboard (motor + tablas de 23/24).

Uso, desde la raiz del repo y con las tablas del dashboard ya generadas (python correr_prototipo.py --sin-abrir):
    python video/verificar_cifras_guion.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dashboard.lib import data, engine  # noqa: E402

Y = 2026
D = ROOT / "dashboard" / "data"
P, A = pd.read_parquet(D / "pitchers.parquet"), pd.read_parquet(D / "arsenal.parquet")
FP, I, T = pd.read_parquet(D / "fingerprint.parquet"), pd.read_parquet(D / "inbox.parquet"), pd.read_parquet(D / "teams.parquet")
P, A, FP, I = P[P.year == Y], A[A.year == Y], FP[FP.year == Y], I[I.year == Y]

print("== Roster: Equipo Extreme-1 ==")
ids = T[(T.year == Y) & (T.team_label == "Equipo Extreme-1")].pitcher_anon_id
R = P[P.pitcher_anon_id.isin(ids)]
Rm = R[R.stuff_mar.notna()]
print(f"n={len(R)}  KPI nivel del mar {np.average(Rm.stuff_mar, weights=Rm.n):.2f}  temporada tal cual {np.average(R.stuff_plus, weights=R.n):.2f}")
print("alertas:", R.alert.value_counts().to_dict())
Ir = I[I.pitcher_anon_id.isin(R.pitcher_anon_id)].assign(o=lambda d: d.alert.map({"red": 0, "yellow": 1, "green": 2})).sort_values(["o", "delta"])
for r in Ir.head(4).itertuples():
    print(f"  inbox: {r.pitcher_anon_id} {r.type} {r.AutoPitchType} | {r.headline} | {r.evidence_text}")
print("liga 2026 alertas:", P[P.stuff_harp.notna() & (P.n >= 100)].alert.value_counts().to_dict())

PID, PT = "pitcher_00964", "Four-Seam"
print("\n== Player 00964 ==")
a = A[(A.pitcher_anon_id == PID) & (A.AutoPitchType == PT)].iloc[0]
f = FP[(FP.pitcher_anon_id == PID) & (FP.AutoPitchType == PT)].set_index("env")
print(f"FF stuff_mar {a.stuff_mar:.2f} stuff_harp {a.stuff_harp:.2f} delta {a.delta:.2f} exceso {a.exceso:.2f} obs {a.delta_observado:.2f} accion {a.action} evid {a.evidence}")
print(f"IVB mar {f.loc['No Altitude','ivb']:.2f} -> harp {f.loc['Extreme Altitude','ivb']:.2f}; HB {f.loc['No Altitude','hb']:.2f} -> {f.loc['Extreme Altitude','hb']:.2f}")
print(f"whiff {100*a.whiff_mar:.2f} -> {100*a.whiff_harp:.2f} ({100*(a.whiff_harp-a.whiff_mar):+.2f} pp); chase {100*(a.chase_harp-a.chase_mar):+.2f} pp")
pp = P[P.pitcher_anon_id == PID].iloc[0]
print(f"pitcher: stuff_mar {pp.stuff_mar:.2f} harp {pp.stuff_harp:.2f} loc+ {pp.location_plus:.1f} pitch+ {pp.pitching_plus:.1f} fit {pp.fit} alert {pp.alert}")
for r in A[A.pitcher_anon_id == PID].sort_values("usage", ascending=False).itertuples():
    print(f"  {r.AutoPitchType:10s} uso {r.usage:.2f} mar {r.stuff_mar:.1f} harp {r.stuff_harp:.1f} delta {r.delta:+.2f} accion {r.action}")
EV = pd.read_parquet(D / "evidence_levels.parquet")
print("Δ liga por tipo:", {r.AutoPitchType: (round(r.delta_2026, 2), round(r.observado_2026, 2)) for r in EV.itertuples()})

print("\n== Simulator ==")
E = engine.Engine()
FIELDS = {"RelSpeed": 0.1, "InducedVertBreak": 0.1, "HorzBreak": 0.1, "SpinRate": 10.0, "SpinAxis": 1.0, "Extension": 0.05, "RelHeight": 0.05}


def base_of(sea):
    b = {k: float(np.nanmean(sea[k])) for k in ["RelSpeed", "InducedVertBreak", "HorzBreak", "SpinRate", "Extension", "RelHeight", "RelSide"]}
    b["SpinAxis"] = float(np.degrees(np.arctan2(np.nanmean(np.sin(np.radians(sea.SpinAxis))), np.nanmean(np.cos(np.radians(sea.SpinAxis))))) % 360)
    return {k: (round(v / FIELDS[k]) * FIELDS[k] if k in FIELDS else v) for k, v in b.items()}


def show(tab, lab):
    t = tab.set_index("env")
    print(f"{lab}: " + " | ".join(f"{engine.ENV_LABEL[e]} {t.loc[e,'stuff_plus']:.2f} (IVB {t.loc[e,'IVB']:.1f}, whiff {100*t.loc[e,'p_whiff_per_pitch']:.1f}%)" for e in engine.ENVS))


rows, ars = E.pitcher_rows(PID, PT)
b = base_of(E.sea_level(rows))
print(f"00964 FF: {len(rows)} pitches, base {dict((k, round(v, 2)) for k, v in b.items())}")
show(E.population_table(rows, ars), "actual")
show(E.population_table(rows, ars, {"RelSpeed": 1.0}), "+1 mph")
show(E.population_table(rows, ars, {"InducedVertBreak": 2.0}), "+2 in IVB")
so = E.population_table(rows, ars, {}, hybrid=False).set_index("env").stuff_plus
print(f"sensibilidad con arsenal {so['Extreme Altitude'] - so['No Altitude']:+.2f}")

for hand in ["Right"]:
    m = ((E.F.AutoPitchType.astype(str) == PT) & (E.F.PitcherThrows.astype(str) == hand) & E.F.altitude_category.astype(str).isin(engine.ENVS)).to_numpy()
    mr = E.F[m].sample(min(int(m.sum()), 3000), random_state=0).reset_index(drop=True)
    mb = base_of(E.sea_level(mr))
    za = np.zeros(len(mr), bool)
    print(f"Manual FF {hand}: base {dict((k, round(v, 2)) for k, v in mb.items())}")
    show(E.population_table(mr, za), "  promedio liga")
    sh = {"RelSpeed": 95.0 - mb["RelSpeed"], "InducedVertBreak": 18.0 - mb["InducedVertBreak"]}
    show(E.population_table(mr, za, sh), "  95 mph, IVB 18")
    print("  fuera de rango:", E.out_of_range({**mb, "RelSpeed": 95.0, "InducedVertBreak": 18.0, "AutoPitchType": PT, "PitcherThrows": hand}))

print("\n== Mapa de movimiento (whiff real) ==")
MM = data.movement_map(Y)
for t in ["Four-Seam", "Slider", "Changeup"]:
    mt = MM[MM.AutoPitchType == t]
    hi, lo = mt[mt.grupo == "Más efectivos"], mt[mt.grupo == "Menos efectivos"]
    print(f"{t}: n={len(mt)} ({len(hi)}/{len(lo)}) whiff {100*hi.obs_whiff.mean():.1f} vs {100*lo.obs_whiff.mean():.1f}; "
          f"stuff_harp {hi.stuff_harp.mean():.1f} vs {lo.stuff_harp.mean():.1f}; velo {hi.velo.mean():.1f} vs {lo.velo.mean():.1f}; "
          f"IVB harp {hi.ivb_harp.mean():.1f} vs {lo.ivb_harp.mean():.1f}; IVB mar {hi.ivb_mar.mean():.1f} vs {lo.ivb_mar.mean():.1f}; "
          f"HB harp {hi.hb_harp.mean():.1f} vs {lo.hb_harp.mean():.1f}")
a964 = MM[(MM.pitcher_anon_id == PID) & (MM.AutoPitchType == PT)]
print("00964 FF en el mapa:", a964[["obs_whiff", "grupo", "ivb_harp", "hb_harp"]].to_dict("records"))

print("\n== Game 00776 Four-Seam, 0-2 vs derecho ==")
ctx = {"BatterSide": "Right", "Balls": 0, "Strikes": 2, "Outs": 0}
for env in ["Extreme Altitude", "No Altitude"]:
    g = E.game_grid("pitcher_00776", PT, ctx, env)
    i = np.unravel_index(g["pitching_plus"].argmax(), g["pitching_plus"].shape)
    zone = (np.abs(g["xs"])[None, :] <= 0.83) & (g["ys"][:, None] >= 1.5) & (g["ys"][:, None] <= 3.5)
    print(f"{engine.ENV_LABEL[env]}: Pitching+ max {g['pitching_plus'][i]:.1f} en x={g['xs'][i[1]]:.2f} z={g['ys'][i[0]]:.2f}; "
          f"whiff {100*g['whiff_per_pitch'][i]:.1f}%; costo ahi {g['damage'][i]:.2f}; costo zona {g['damage'][zone].mean():.2f} runs/100")
