"""Arsenal en Harp Helu: ¿que repertorio produce el Stuff+ mas alto en altitud? (entregable 1 del reto).

Lee la capa de datos de 23 (dashboard/data/arsenal.parquet y pitchers.parquet), donde el Stuff+ en
Harp Helu = Stuff+ oficial a nivel del mar + delta fisico (guia 38). Pitchers con >= 100 pitches en
la temporada. Todo es descriptivo: quien lanza que repertorio no es aleatorio.

Salidas (CSV en la raiz y parquet en dashboard/data):
  24_arsenal_composicion   Stuff+ del arsenal (ponderado por uso) por composicion = tipos con uso
                           >= 10%; solo composiciones con >= 10 pitchers. IC95% bootstrap por pitcher.
  24_arsenal_rectas        Cuartiles de % de rectas (Four-Seam + Sinker) y pendiente por +10 pp de
                           rectas sobre el Stuff+ a nivel del mar, en Harp Helu y sobre el delta. Cada
                           cuartil trae tambien la tasa de roletazos real (gb_rate, ver 24_arsenal_contraste).
  24_arsenal_mezcla        Descomposicion Stuff+ Harp = mezcla + calidad. Mezcla = Stuff+ que tendria
                           el pitcher si cada pitch suyo fuera el promedio de liga de su tipo;
                           calidad = el resto. R2 de la mezcla a nivel del mar y en Harp Helu.
  24_arsenal_tipos         Stuff+ medio de liga por tipo (nivel del mar, Harp Helu), ponderado por pitches.
  24_arsenal_top           Los 10 arsenales individuales con mayor Stuff+ en Harp Helu (>= 300 pitches).
  24_arsenal_contraste     Contraste con resultados reales (Revision_Pre_Guion A7): el Stuff+ pondera sobre todo
                           swing-and-miss (chase y whiff) y casi nada el roletazo. Pendiente del valor realizado por
                           100 pitches lanzados en Extreme sobre el % de rectas (pitchers con >= 100 pitches en Extreme,
                           ponderada por esos pitches, IC bootstrap por pitcher) y tasa de roletazos de los pitchers con
                           sinker >= 30% contra el resto. Tasa de roletazos = roletazos / bolas en juego sin toques
                           (guia 22), promedio simple por pitcher, todos los parques; la calcula 23 (gb_rate) y los
                           fouls no cuentan.

La pendiente de "dependencia de la recta" en Stuff+ es casi aritmetica (las rectas pierden mas y el resto casi nada,
por el uso de cada tipo): describe la lectura de swing-and-miss del modelo, no una recomendacion de roster.
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA = BASE_DIR / "dashboard" / "data"
MIN_P, MIN_TOP, MIN_USAGE, MIN_GROUP, B, SEED = 100, 300, 0.10, 10, 2000, 42
MIN_EXT, SINKER_HEAVY = 100, 0.30
CODE = {"Four-Seam": "FF", "Sinker": "SI", "Cutter": "FC", "Slider": "SL", "Curveball": "CU", "Changeup": "CH", "Splitter": "FS"}
ORDER = ["FF", "SI", "FC", "SL", "CU", "CH", "FS"]
FB = ["Four-Seam", "Sinker"]


def wavg(g, col):
    return float(np.average(g[col], weights=g["n"]))


def boot_ci(fn, d, rng):
    idx = np.arange(len(d))
    v = [fn(d.iloc[rng.choice(idx, len(idx))]) for _ in range(B)]
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def slope10(d, y):
    return float(np.polyfit(d["pct_rectas"], d[y], 1)[0] * 10)


def main():
    A, P = pd.read_parquet(DATA / "arsenal.parquet"), pd.read_parquet(DATA / "pitchers.parquet")
    rng = np.random.default_rng(SEED)
    comp, rect, mezcla, tipos, top, contra = [], [], [], [], [], []
    for year in sorted(P.year.unique()):
        p = P[(P.year == year) & (P.n >= MIN_P)].set_index("pitcher_anon_id")
        a = A[(A.year == year) & A.pitcher_anon_id.isin(p.index) & A.stuff_harp.notna() & A.stuff_mar.notna()].copy()
        if len(p) < 50:
            continue
        lt = pd.DataFrame({t: {"n_pitches": int(g.n.sum()), "stuff_mar": wavg(g, "stuff_mar"), "stuff_harp": wavg(g, "stuff_harp")}
                           for t, g in a.groupby("AutoPitchType")}).T
        lt["delta"] = lt.stuff_harp - lt.stuff_mar
        tipos.append(lt.rename_axis("AutoPitchType").reset_index().assign(year=year))
        a["liga_mar"], a["liga_harp"] = a.AutoPitchType.map(lt.stuff_mar), a.AutoPitchType.map(lt.stuff_harp)
        rows = []
        for pid, g in a.groupby("pitcher_anon_id"):
            sig = [CODE[t] for t in g[g.usage >= MIN_USAGE].AutoPitchType]
            rows.append({"pitcher_anon_id": pid, "n": int(g.n.sum()), "stuff_mar": wavg(g, "stuff_mar"), "stuff_harp": wavg(g, "stuff_harp"),
                         "mezcla_mar": wavg(g, "liga_mar"), "mezcla_harp": wavg(g, "liga_harp"),
                         "pct_rectas": 100 * g.n[g.AutoPitchType.isin(FB)].sum() / g.n.sum(),
                         "composicion": "+".join(c for c in ORDER if c in sig),
                         "usos": " · ".join(f"{CODE[r.AutoPitchType]} {r.usage:.0%}" for r in g.sort_values("usage", ascending=False).itertuples() if r.usage >= MIN_USAGE)})
        d = pd.DataFrame(rows)
        d["delta"] = d.stuff_harp - d.stuff_mar
        d["calidad_harp"] = d.stuff_harp - d.mezcla_harp
        d["hand"], d["role"] = d.pitcher_anon_id.map(p.hand), d.pitcher_anon_id.map(p.role)
        for c in ["gb_rate", "value_per_100_extreme", "n_extreme"]:
            d[c] = d.pitcher_anon_id.map(p[c]) if c in p else np.nan
        si = a[a.AutoPitchType == "Sinker"].set_index("pitcher_anon_id")["usage"]
        d["uso_sinker"] = d.pitcher_anon_id.map(si).fillna(0.0)

        # composicion
        for sig, g in d.groupby("composicion"):
            if len(g) < MIN_GROUP or not sig:
                continue
            lo, hi = boot_ci(lambda x: x.stuff_harp.mean(), g, rng)
            comp.append({"year": year, "composicion": sig, "n_pitchers": len(g), "stuff_mar": g.stuff_mar.mean(), "stuff_harp": g.stuff_harp.mean(),
                         "ci95_lo": lo, "ci95_hi": hi, "delta": g.delta.mean(), "pct_rectas": g.pct_rectas.mean()})

        # % de rectas
        d["cuartil"] = pd.qcut(d.pct_rectas, 4, labels=["Q1 (menos rectas)", "Q2", "Q3", "Q4 (mas rectas)"])
        for q, g in d.groupby("cuartil", observed=True):
            rect.append({"year": year, "grupo": q, "n_pitchers": len(g), "pct_rectas": g.pct_rectas.mean(), "stuff_mar": g.stuff_mar.mean(),
                         "stuff_harp": g.stuff_harp.mean(), "delta": g.delta.mean(), "ci95_lo": np.nan, "ci95_hi": np.nan,
                         "gb_rate": g.gb_rate.mean()})
        for y, lab in [("stuff_mar", "Pendiente por +10 pp de rectas: Stuff+ nivel del mar"), ("stuff_harp", "Pendiente por +10 pp de rectas: Stuff+ Harp Helu"),
                       ("delta", "Pendiente por +10 pp de rectas: delta de altitud")]:
            lo, hi = boot_ci(lambda x, y=y: slope10(x, y), d, rng)
            rect.append({"year": year, "grupo": lab, "n_pitchers": len(d), "pct_rectas": np.nan, "stuff_mar": np.nan, "stuff_harp": np.nan,
                         "delta": slope10(d, y), "ci95_lo": lo, "ci95_hi": hi})

        # contraste con resultados reales en Extreme
        e = d[d.n_extreme >= MIN_EXT].dropna(subset=["value_per_100_extreme"])
        if len(e) >= 30:
            sl = lambda x: float(np.polyfit(x.pct_rectas, x.value_per_100_extreme, 1, w=np.sqrt(x.n_extreme))[0] * 10)
            lo, hi = boot_ci(sl, e, rng)
            contra.append({"year": year, "medida": "Valor real por 100 pitches en Extreme: pendiente por +10 pp de rectas", "valor": sl(e),
                           "ci95_lo": lo, "ci95_hi": hi, "n_pitchers": len(e)})
        for lab, g in [("Tasa de roletazos: sinker >= 30% del arsenal", d[d.uso_sinker >= SINKER_HEAVY]),
                       ("Tasa de roletazos: resto", d[d.uso_sinker < SINKER_HEAVY])]:
            g = g.dropna(subset=["gb_rate"])
            if len(g) >= 10:
                lo, hi = boot_ci(lambda x: x.gb_rate.mean(), g, rng)
                contra.append({"year": year, "medida": lab, "valor": g.gb_rate.mean(), "ci95_lo": lo, "ci95_hi": hi, "n_pitchers": len(g)})

        # mezcla vs calidad
        r2 = lambda x, y: float(np.corrcoef(x, y)[0, 1] ** 2)
        mezcla.append({"year": year, "n_pitchers": len(d), "sd_stuff_harp": d.stuff_harp.std(), "sd_mezcla_harp": d.mezcla_harp.std(),
                       "sd_calidad_harp": d.calidad_harp.std(), "r2_mezcla_nivel_mar": r2(d.mezcla_mar, d.stuff_mar),
                       "r2_mezcla_harp": r2(d.mezcla_harp, d.stuff_harp), "corr_mezcla_calidad": float(np.corrcoef(d.mezcla_harp, d.calidad_harp)[0, 1])})

        # top individuales
        t = d[d.n >= MIN_TOP].sort_values("stuff_harp", ascending=False).head(10)
        top.append(t[["pitcher_anon_id", "hand", "role", "n", "composicion", "usos", "pct_rectas", "stuff_mar", "stuff_harp", "delta", "mezcla_harp", "calidad_harp"]].assign(year=year))

    out = {"24_arsenal_composicion": pd.DataFrame(comp).sort_values(["year", "stuff_harp"], ascending=[True, False]),
           "24_arsenal_rectas": pd.DataFrame(rect), "24_arsenal_mezcla": pd.DataFrame(mezcla),
           "24_arsenal_tipos": pd.concat(tipos), "24_arsenal_top": pd.concat(top), "24_arsenal_contraste": pd.DataFrame(contra)}
    pd.set_option("display.width", 250)
    for name, df in out.items():
        df.to_csv(BASE_DIR / f"{name}.csv", index=False)
        df.to_parquet(DATA / f"{name[3:]}.parquet", index=False)
        print(f"\n== {name} ({len(df)})"); print(df.round(2).to_string(index=False))
    c = out["24_arsenal_composicion"].pivot(index="composicion", columns="year", values="stuff_harp").dropna()
    if c.shape[1] >= 2:
        print(f"\nEstabilidad del ranking de composiciones entre temporadas (Spearman, {len(c)} composiciones): {c.corr(method='spearman').iloc[0, 1]:.2f}")


if __name__ == "__main__":
    main()
