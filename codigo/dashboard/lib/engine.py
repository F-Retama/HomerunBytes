"""Motor del Simulator: califica pitches hipotéticos con los modelos congelados de 18.

Trabaja directamente en convención de diestro (HB, RelSide y SpinAxis ya espejados), igual que el
frame del bundle: no se vuelve a espejar. El movimiento de entrada se interpreta como equivalente a
nivel del mar (No Altitude) y se lleva a cada entorno con los factores validados (20/21/22).
Verificado contra los scores oficiales de 18 en verify() (Stuff+, Location+ y Pitching+).

Dos modos de carga:
- completo: el bundle de 18 (18_three_scores_bundle.pkl, ~400 MB con todos los lanzamientos). Es el que se usa al
  correr el pipeline localmente.
- ligero: dashboard/lite/engine_lite.pkl (lo genera 25_dashboard_lite.py). Trae solo los modelos de la ventana
  2026, una muestra de lanzamientos por pitcher y tipo (solo las columnas que usa el modelo, sin resultados ni IDs
  de juego o bateador), promedios exactos y las tablas de referencia ya calculadas. Es el que corre en la versión
  hosteada para el jurado. Se usa automáticamente si no existe el bundle completo.
"""

import importlib.util
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FULL_BUNDLE = ROOT / "18_three_scores_bundle.pkl"
LITE_BUNDLE = ROOT / "dashboard" / "lite" / "engine_lite.pkl"
WINDOW = "2024-2025 -> 2026"
SCALE = "scale_sin_contact_value"
ENVS = ["No Altitude", "Medium Altitude", "Extreme Altitude"]
ENV_LABEL = {"No Altitude": "Nivel del mar", "Medium Altitude": "Medium", "Extreme Altitude": "Harp Helú"}
SHOW = ["whiff_per_pitch", "chase", "called_strike"]


def to_env_frame(frame, from_env, to_env_, fac):
    """Lleva el movimiento de from_env a to_env_ (factores relativos a No Altitude). Copia de to_env en 22."""
    f = frame.copy()
    for var, dcol in [("InducedVertBreak", "ivb_diff_vs_fb"), ("HorzBreak", "hb_diff_vs_fb")]:
        r_from = 1.0 if from_env == "No Altitude" else fac[from_env][var]
        r_to = 1.0 if to_env_ == "No Altitude" else fac[to_env_][var]
        k = r_to / r_from
        f[var] = f[var] * k; f[dcol] = f[dcol] * k
    return f


def family_predict(models, frame, has_ars, variants):
    """Predicción por fila con la variante de su nivel de información. Copia de family_predict en 16."""
    p = np.full(len(frame), np.nan)
    if "arsenal" in variants and has_ars.any():
        p[has_ars] = models["arsenal"].predict_proba(frame.loc[has_ars, variants["arsenal"]])[:, 1]
    rest = ~has_ars if "arsenal" in variants else np.ones(len(frame), bool)
    if rest.any():
        p[rest] = models["base"].predict_proba(frame.loc[rest, variants["base"]])[:, 1]
    return p


class ContactValueLite:
    """Misma rama de contacto que ContactValueModel (16), sin depender del módulo del pipeline."""

    def __init__(self, p_bip, reg):
        self.p_bip, self.reg = p_bip, reg

    def predict_proba(self, X):
        x = self.p_bip.predict_proba(X)[:, 1] * self.reg.predict(X)
        return np.column_stack([-x, x])


def circ_mean(deg):
    return float(np.degrees(np.arctan2(np.nanmean(np.sin(np.radians(deg))), np.nanmean(np.cos(np.radians(deg))))) % 360)


BASE_FIELDS = ["RelSpeed", "InducedVertBreak", "HorzBreak", "SpinRate", "Extension", "RelHeight", "RelSide"]


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class Engine:
    def __init__(self, lite=None):
        self.lite = (not FULL_BUNDLE.exists()) if lite is None else lite
        if self.lite:
            L = joblib.load(LITE_BUNDLE)
            self.L = L
            self.fams = L["families"]
            for d in self.fams.values():
                for c, vm in d["models"].items():
                    for v, m in list(vm.items()):
                        if isinstance(m, dict) and "p_bip" in m:
                            vm[v] = ContactValueLite(m["p_bip"], m["reg"])
            self.signs, self.comps, self.fac = L["signs"], L["components"], L["fac"]
            self.cats, self.ranges = L["cats"], L["ranges"]
            return
        self.t22 = _load("profile_recommendations_export", "22_profile_recommendations_export.py")
        self.t16, self.t18 = self.t22.t16, self.t22.t18
        self.cfg = self.t18.build_cfg(self.t16.load_modules())
        self.bundle = joblib.load(FULL_BUNDLE)
        self.entry = self.bundle[WINDOW]
        self.fams = self.entry["families"]
        self.signs = self.cfg["signs"]
        self.F = self.entry["future_frame"]
        self.fac = self.t22.factors()[WINDOW]
        self.comps = self.cfg["components"]
        self.cats = {c: self.F[c].cat.categories for c in self.F.columns if isinstance(self.F[c].dtype, pd.CategoricalDtype)}
        num = ["RelSpeed", "SpinRate", "SpinAxis", "RelHeight", "RelSide", "Extension", "InducedVertBreak", "HorzBreak"]
        self.ranges = self.F.groupby(self.F["AutoPitchType"].astype(str))[num].quantile([0.01, 0.99]).unstack()

    # ---------------- filas ----------------
    def frame(self, pitch, fb=None, usage=None, ctx=None, n=1):
        """pitch/fb: dict en convención de diestro y movimiento a nivel del mar. ctx: ubicación/count/contexto."""
        ctx = ctx or {}
        row = {"RelSpeed": pitch["RelSpeed"], "SpinRate": pitch["SpinRate"], "SpinAxis": pitch["SpinAxis"], "RelHeight": pitch["RelHeight"],
               "RelSide": pitch["RelSide"], "Extension": pitch["Extension"], "InducedVertBreak": pitch["InducedVertBreak"],
               "HorzBreak": pitch["HorzBreak"], "AutoPitchType": pitch["AutoPitchType"], "PitcherThrows": pitch["PitcherThrows"],
               "PlateLocHeight": ctx.get("PlateLocHeight", 2.5), "PlateLocSide": ctx.get("PlateLocSide", 0.0),
               "Balls": ctx.get("Balls", 0), "Strikes": ctx.get("Strikes", 0), "BatterSide": ctx.get("BatterSide", "Right"),
               "Outs": ctx.get("Outs", 0), "Inning": ctx.get("Inning", 1), "prev_pitch_type": ctx.get("prev_pitch_type", "Primer_pitch"),
               "altitude_category": "No Altitude"}
        has_ars = fb is not None and usage is not None
        if has_ars:
            row.update({"velo_diff_vs_fb": pitch["RelSpeed"] - fb["RelSpeed"], "ivb_diff_vs_fb": pitch["InducedVertBreak"] - fb["InducedVertBreak"],
                        "hb_diff_vs_fb": pitch["HorzBreak"] - fb["HorzBreak"], "spin_diff_vs_fb": pitch["SpinRate"] - fb["SpinRate"],
                        "pitch_usage_pct": usage, "extension_diff_vs_fb": pitch["Extension"] - fb["Extension"],
                        "release_distance_vs_fb": float(np.hypot(pitch["RelHeight"] - fb["RelHeight"], pitch["RelSide"] - fb["RelSide"]))})
        else:
            row.update({k: np.nan for k in ["velo_diff_vs_fb", "ivb_diff_vs_fb", "hb_diff_vs_fb", "spin_diff_vs_fb", "pitch_usage_pct",
                                            "extension_diff_vs_fb", "release_distance_vs_fb"]})
        df = pd.DataFrame([row] * n)
        return self._cat(df), np.full(n, has_ars)

    def _cat(self, df):
        for c, cats in self.cats.items():
            if c in df:
                df[c] = pd.Categorical(df[c].astype(object), categories=cats)
        return df

    def to_env(self, df, env):
        out = to_env_frame(df, "No Altitude", env, self.fac)
        out["altitude_category"] = pd.Categorical([env] * len(out), categories=self.cats["altitude_category"])
        return out

    # ---------------- scores ----------------
    def score(self, family, df, has_ars):
        fam = self.fams[family]
        w = np.asarray(fam["weights"], float)
        need = {c for c, wi in zip(self.comps, w) if abs(wi) > 1e-9} | set(SHOW)
        probs = {c: family_predict(fam["models"][c], df, has_ars, fam["variants"]) for c in self.comps if c in need}
        z = np.zeros(len(df))
        for c, wi in zip(self.comps, w):
            if abs(wi) > 1e-9:
                mu, sd = fam["stats"][c]
                z += wi * self.signs.get(c, 1.0) * (probs[c] - mu) / sd
        m, s = fam["stats"][SCALE]
        return 100 + 10 * (z - m) / s, probs

    def altitude_table(self, pitch, fb=None, usage=None, ctx=None):
        """Stuff+ (y Pitching+ con ctx) del mismo pitch en los tres entornos, con el esquema hibrido de la guia 38:
        nivel = variante oficial (con arsenal si hay fb) a nivel del mar; efecto del entorno = variante fisica."""
        base, ars = self.frame(pitch, fb, usage, ctx)
        phys = np.zeros(len(base), bool)
        fams = ["stuff"] + (["pitching"] if ctx is not None else [])
        sea = self.to_env(base, "No Altitude")
        off0 = {f: self.score(f, sea, ars) for f in fams}
        phy0 = {f: self.score(f, sea, phys) for f in fams}
        rows = []
        for env in ENVS:
            x = self.to_env(base, env)
            phy = {f: self.score(f, x, phys) for f in fams}
            s = off0["stuff"][0][0] + phy["stuff"][0][0] - phy0["stuff"][0][0]
            r = {"env": env, "label": ENV_LABEL[env], "stuff_plus": float(s), "IVB": float(x["InducedVertBreak"].iloc[0]),
                 "HB": float(x["HorzBreak"].iloc[0]),
                 **{f"p_{c}": float(off0["stuff"][1][c][0] + phy["stuff"][1][c][0] - phy0["stuff"][1][c][0]) for c in SHOW}}
            if ctx is not None:
                r["pitching_plus"] = float(off0["pitching"][0][0] + phy["pitching"][0][0] - phy0["pitching"][0][0])
            rows.append(r)
        return pd.DataFrame(rows)

    # ---------------- what-if sobre los pitches reales de un pitcher ----------------
    SHIFTABLE = {"RelSpeed": ["velo_diff_vs_fb"], "InducedVertBreak": ["ivb_diff_vs_fb"], "HorzBreak": ["hb_diff_vs_fb"],
                 "SpinRate": ["spin_diff_vs_fb"], "SpinAxis": [], "Extension": ["extension_diff_vs_fb"], "RelHeight": [], "RelSide": []}

    def pitcher_rows(self, pid, ptype):
        """Pitches reales del pitcher y tipo. En modo ligero es la muestra guardada (hasta lite["k"] por pitcher y tipo)."""
        if self.lite:
            S = self.L["pitcher_rows"]
            m = ((S["pitcher_anon_id"] == pid) & (S["AutoPitchType"].astype(str) == ptype)).to_numpy()
            rows = S[m].reset_index(drop=True)
            rows.attrs["lite_key"] = (pid, ptype)  # population_table usa la tabla exacta guardada para este pitcher
            return rows, self.L["pitcher_ars"][m]
        m = ((self.F["pitcher_anon_id"] == pid) & (self.F["AutoPitchType"].astype(str) == ptype)
             & self.F["altitude_category"].astype(str).isin(ENVS)).to_numpy()
        return self.F[m].reset_index(drop=True), self.entry["f_ars"][m]

    def pitcher_summary(self, pid, ptype):
        """Punto de partida del Simulator con TODOS los pitches del pitcher (exacto también en modo ligero):
        medias a nivel del mar (eje circular), total de pitches, mano y si usa la variante con arsenal."""
        if self.lite:
            return dict(self.L["pitcher_summary"][(pid, ptype)])
        rows, ars = self.pitcher_rows(pid, ptype)
        sea = self.sea_level(rows)
        base = {f: float(np.nanmean(sea[f])) if sea[f].notna().any() else np.nan for f in self.NUM_FEATS}
        base["SpinAxis"] = circ_mean(sea.SpinAxis)
        return {"base": base, "n": len(rows), "hand": rows.PitcherThrows.astype(str).mode().iloc[0], "ars_share": float(ars.mean())}

    def manual_rows(self, ptype, hand, n=3000):
        """Pitches de la liga (tipo y mano) para el modo 'Pitch manual' y su promedio a nivel del mar.
        Modo ligero: la muestra de league_sample; el promedio es el de la muestra completa original (exacto)."""
        if self.lite:
            rows = self.league_sample(ptype, hand)[0]
            return rows, dict(self.L["manual_base"][(ptype, hand)])
        m = ((self.F.AutoPitchType.astype(str) == ptype) & (self.F.PitcherThrows.astype(str) == hand)
             & self.F.altitude_category.astype(str).isin(ENVS)).to_numpy()
        rows = self.F[m].sample(min(int(m.sum()), n), random_state=0).reset_index(drop=True)
        sea = self.sea_level(rows)
        base = {f: float(np.nanmean(sea[f])) for f in BASE_FIELDS}
        base["SpinAxis"] = circ_mean(sea.SpinAxis)
        return rows, base

    def sea_level(self, rows):
        """Cada pitch llevado a su equivalente a nivel del mar desde el entorno donde se lanzó."""
        out = rows.copy()
        alt = rows["altitude_category"].astype(str).to_numpy()
        for env in ["Medium Altitude", "Extreme Altitude"]:
            m = alt == env
            if m.any():
                out.loc[m] = to_env_frame(rows[m], env, "No Altitude", self.fac)
        return out

    def population_table(self, rows, ars, shifts=None, hybrid=True):
        """Stuff+ medio de los pitches reales del pitcher en cada entorno, con desplazamientos what-if
        (sumados a cada pitch a nivel del mar). hybrid=True (titular, guía 38): nivel = Stuff+ oficial a nivel
        del mar y efecto de altitud = variante física; con shifts vacíos reproduce la proyección de 22.
        hybrid=False: todo con la variante dada por ars (sensibilidad).
        Con shifts, el EFECTO DEL CAMBIO se mide con la variante oficial en cada entorno y se suma a la proyeccion
        hibrida del pitch actual: a nivel del mar el escenario es exactamente el Stuff+ oficial. Mezclar la
        variante fisica en el cambio (su sensibilidad al IVB es otra) cancelaba ganancias en altitud.

        Modo ligero con pitches de un pitcher (rows.attrs['lite_key']): la tabla sin cambios es la exacta, calculada con
        todos sus pitches; el EFECTO del what-if se estima con la muestra guardada y se suma a esa tabla."""
        key = rows.attrs.get("lite_key") if self.lite else None
        if key is not None:
            out = self.L["pop_exact"][(key, bool(hybrid))].copy()
            if not any(abs(d) > 1e-12 for d in (shifts or {}).values()):
                return out
            s0 = self._population_core(rows, ars, None, hybrid).set_index("env")
            s1 = self._population_core(rows, ars, shifts, hybrid).set_index("env").reindex(s0.index)
            out = out.set_index("env")
            for col in ["stuff_plus", "IVB", "HB"] + [f"p_{c}" for c in SHOW]:
                out[col] = out[col] + s1[col] - s0[col]
            return out.reset_index()
        return self._population_core(rows, ars, shifts, hybrid)

    def _population_core(self, rows, ars, shifts=None, hybrid=True):
        base = self.sea_level(rows)
        if not hybrid:
            return self._score_envs(self._shift(base, shifts), ars)
        out = self._hybrid(base, ars)
        if not any(abs(d) > 1e-12 for d in (shifts or {}).values()):
            return out
        off0 = self._score_envs(base, ars).set_index("env")
        off1 = self._score_envs(self._shift(base, shifts), ars).set_index("env").reindex(off0.index)
        out = out.set_index("env")
        for col in ["stuff_plus"] + [f"p_{c}" for c in SHOW]:
            out[col] = out[col] + off1[col] - off0[col]
        for col in ["IVB", "HB"]:
            out[col] = off1[col]
        return out.reset_index()

    def _hybrid(self, sea, ars):
        off = self._score_envs(sea, ars).set_index("env")
        phy = self._score_envs(sea, np.zeros(len(sea), bool)).set_index("env")
        out = phy.copy()
        for col in ["stuff_plus"] + [f"p_{c}" for c in SHOW]:
            out[col] = off.loc["No Altitude", col] + phy[col] - phy.loc["No Altitude", col]
        return out.reset_index()

    def _shift(self, sea, shifts):
        """Desplaza la fisica de cada pitch y sus diferencias contra la recta. La recta de referencia es HISTORICA
        (temporadas anteriores, fold-safe), asi que un cambio en la recta misma tambien mueve su diferencia."""
        sea = sea.copy()
        for f, d in (shifts or {}).items():
            if abs(d) < 1e-12:
                continue
            sea[f] = (sea[f] + d) % 360 if f == "SpinAxis" else sea[f] + d
            for dc in self.SHIFTABLE[f]:
                sea[dc] = sea[dc] + d
        return sea

    def _score_envs(self, sea, ars):
        res = []
        for env in ENVS:
            x = self.to_env(sea, env)
            sc, p = self.score("stuff", x, ars)
            res.append({"env": env, "label": ENV_LABEL[env], "stuff_plus": float(np.mean(sc)), "IVB": float(x["InducedVertBreak"].mean()),
                        "HB": float(x["HorzBreak"].mean()), **{f"p_{c}": float(np.mean(p[c])) for c in SHOW}})
        return pd.DataFrame(res)

    # ---------------- prospectos ----------------
    def league_sample(self, ptype, hand, n=2000):
        """Pitches reales de la liga (tipo y mano) llevados a nivel del mar; se cachean."""
        self.__dict__.setdefault("_samples", {})
        key = (ptype, hand)
        if key not in self._samples:
            if self.lite:
                S = self.L["league_rows"]
                m = ((S.AutoPitchType.astype(str) == ptype) & (S.PitcherThrows.astype(str) == hand)).to_numpy()
                rows = S[m].reset_index(drop=True)
            else:
                m = ((self.F.AutoPitchType.astype(str) == ptype) & (self.F.PitcherThrows.astype(str) == hand)
                     & self.F.altitude_category.astype(str).isin(ENVS)).to_numpy()
                rows = self.F[m].sample(min(int(m.sum()), n), random_state=0).reset_index(drop=True)
            sea = self.sea_level(rows)
            means = {f: float(np.nanmean(sea[f])) for f in self.SHIFTABLE if f != "SpinAxis"}
            means["SpinAxis"] = float(np.degrees(np.arctan2(np.nanmean(np.sin(np.radians(sea.SpinAxis))),
                                                            np.nanmean(np.cos(np.radians(sea.SpinAxis))))) % 360)
            self._samples[key] = (rows, sea, means)
        return self._samples[key]

    def prospect(self, ptype, hand, values, fb=None, usage=None, hybrid=True):
        """Un pitch de un prospecto: dispersión real de la liga (tipo y mano) recentrada en sus métricas a nivel
        del mar. Con fb y usage usa la variante con contexto de arsenal (diferencias contra su recta); si no,
        la variante base (la validada para pitchers nuevos, guía 35). hybrid como en population_table."""
        _rows, sea, means = self.league_sample(ptype, hand)
        shifts = {f: values[f] - means[f] for f in self.SHIFTABLE if f in values and values[f] == values[f]}
        if "SpinAxis" in shifts:
            shifts["SpinAxis"] = (shifts["SpinAxis"] + 180) % 360 - 180
        x = self._shift(sea, {f: d for f, d in shifts.items()})
        if fb is not None and usage is not None:
            x["velo_diff_vs_fb"] = x["RelSpeed"] - fb["RelSpeed"]; x["ivb_diff_vs_fb"] = x["InducedVertBreak"] - fb["InducedVertBreak"]
            x["hb_diff_vs_fb"] = x["HorzBreak"] - fb["HorzBreak"]; x["spin_diff_vs_fb"] = x["SpinRate"] - fb["SpinRate"]
            x["extension_diff_vs_fb"] = x["Extension"] - fb["Extension"]; x["pitch_usage_pct"] = usage
            x["release_distance_vs_fb"] = np.hypot(x["RelHeight"] - fb["RelHeight"], x["RelSide"] - fb["RelSide"])
            ars = np.ones(len(x), bool)
        else:
            ars = np.zeros(len(x), bool)
        return self._hybrid(x, ars) if hybrid else self._score_envs(x, ars)

    def reference_base(self):
        """Δ de altitud de la liga (2026) con la variante SIN arsenal, por pitcher x tipo y por pitcher.
        Referencia justa para prospectos evaluados con esa variante (la variante cambia mucho el Δ de
        slider y changeup: con arsenal se penaliza que su separación contra la recta también se encoja)."""
        if self.lite:
            return self.L["reference_base"]
        if "_ref_base" not in self.__dict__:
            m = self.F.altitude_category.astype(str).isin(ENVS).to_numpy()
            rows = self.F[m].reset_index(drop=True)
            sea = self.sea_level(rows)
            ars = np.zeros(len(rows), bool)
            s_no, _ = self.score("stuff", self.to_env(sea, "No Altitude"), ars)
            s_ex, _ = self.score("stuff", self.to_env(sea, "Extreme Altitude"), ars)
            d = pd.DataFrame({"pitcher_anon_id": rows.pitcher_anon_id.to_numpy(), "AutoPitchType": rows.AutoPitchType.astype(str).to_numpy(),
                              "mar": s_no, "harp": s_ex})
            d["delta"] = d.harp - d.mar
            by_type = d.groupby(["pitcher_anon_id", "AutoPitchType"]).agg(n=("delta", "size"), delta=("delta", "mean"), harp=("harp", "mean")).reset_index()
            by_p = d.groupby("pitcher_anon_id").agg(n=("delta", "size"), delta=("delta", "mean"), harp=("harp", "mean")).reset_index()
            self._ref_base = (by_type[by_type.n >= 50], by_p[by_p.n >= 100])
        return self._ref_base

    # ---------------- Game: rejilla táctica ----------------
    NUM_FEATS = ["RelSpeed", "SpinRate", "RelHeight", "RelSide", "Extension", "InducedVertBreak", "HorzBreak", "velo_diff_vs_fb",
                 "ivb_diff_vs_fb", "hb_diff_vs_fb", "spin_diff_vs_fb", "pitch_usage_pct", "release_distance_vs_fb", "extension_diff_vs_fb"]

    def pitch_means(self, pid, ptype):
        """Pitch representativo del pitcher (medias a nivel del mar, eje circular) y si usa la variante con arsenal."""
        s = self.pitcher_summary(pid, ptype)
        return {**s["base"], "PitcherThrows": s["hand"]}, bool(s["ars_share"] >= 0.5), s["n"]

    MAX_GRID_ROWS = 150  # muestra fija de pitches reales por celda (velocidad); el promedio poblacional casi no cambia

    def _context_frame(self, sea, ctx, xs, ys):
        """Repite cada pitch (a nivel del mar) en cada ubicacion (xs[k], ys[k]) con el contexto de count y bateador."""
        n, L = len(sea), len(xs)
        big = sea.loc[sea.index.repeat(L)].reset_index(drop=True)
        big["PlateLocSide"], big["PlateLocHeight"] = np.tile(xs, n), np.tile(ys, n)
        for k, dflt in [("Balls", 0), ("Strikes", 0), ("Outs", 0), ("Inning", 1)]:
            big[k] = ctx.get(k, dflt)
        for k, dflt in [("BatterSide", "Right"), ("prev_pitch_type", "Primer_pitch")]:
            big[k] = pd.Categorical([ctx.get(k, dflt)] * len(big), categories=self.cats[k])
        return big

    def population_pitching(self, sea, ars, ctx, xs, ys, env, shifts=None):
        """Pitching+, probabilidades y valor del contacto PROMEDIO sobre los pitches reales (no un pitch promedio,
        guia 37.5: el modelo no es lineal) en cada ubicacion, con el esquema hibrido de la guia 38: nivel = variante
        oficial a nivel del mar; efecto del entorno = variante fisica (sin arsenal)."""
        n, L = len(sea), len(xs)
        big = self._context_frame(sea, ctx, xs, ys)
        a_off, a_phys = np.repeat(ars, L), np.zeros(n * L, bool)
        fam = self.fams["pitching"]

        def evaluate(x, a):
            pp_, pr_ = self.score("pitching", x, a)
            cv_ = family_predict(fam["models"]["contact_value"], x, a, fam["variants"])
            return {"pitching_plus": pp_, "contact_value": cv_, **{c: pr_[c] for c in SHOW}}
        sea_x = self.to_env(big, "No Altitude")
        out = evaluate(sea_x, a_off)
        if env != "No Altitude":
            p0, p1 = evaluate(sea_x, a_phys), evaluate(self.to_env(big, env), a_phys)
            out = {k: out[k] + p1[k] - p0[k] for k in out}
        if shifts and any(abs(d) > 1e-12 for d in shifts.values()):
            # efecto del cambio what-if con la variante oficial en el mismo entorno (igual que population_table)
            big1 = self._shift(big, shifts)
            e0, e1 = evaluate(self.to_env(big, env), a_off), evaluate(self.to_env(big1, env), a_off)
            out = {k: out[k] + e1[k] - e0[k] for k in out}
        return {k: v.reshape(n, L).mean(axis=0) for k, v in out.items()}

    def grid_rows(self, pid, ptype):
        """Pitches reales del pitcher y tipo llevados a nivel del mar (muestra fija si son muchos)."""
        rows, ars = self.pitcher_rows(pid, ptype)
        if len(rows) > self.MAX_GRID_ROWS:
            idx = np.sort(np.random.default_rng(0).choice(len(rows), self.MAX_GRID_ROWS, replace=False))
            rows, ars = rows.iloc[idx].reset_index(drop=True), ars[idx]
        return self.sea_level(rows), ars

    def game_grid(self, pid, ptype, ctx, env, nx=13, ny=15):
        """Pitching+, probabilidades y daño esperado en contacto (runs/100, + = más daño) en una rejilla de ubicaciones,
        promediados sobre los pitches reales del pitcher (poblacion, como el What-if y Player)."""
        sea, ars = self.grid_rows(pid, ptype)
        n_total = int(self.pitcher_summary(pid, ptype)["n"])
        xs, ys = np.linspace(-1.5, 1.5, nx), np.linspace(0.8, 4.0, ny)
        gx, gy = np.meshgrid(xs, ys)
        r = self.population_pitching(sea, ars, ctx, gx.ravel(), gy.ravel(), env)
        shp = gy.shape
        return {"xs": xs, "ys": ys, "pitching_plus": r["pitching_plus"].reshape(shp), "damage": (-100 * r["contact_value"]).reshape(shp),
                **{c: r[c].reshape(shp) for c in SHOW}, "n": n_total, "n_used": len(sea), "has_ars": bool(ars.mean() >= 0.5),
                "hand": sea.PitcherThrows.astype(str).mode().iloc[0]}

    def location_grid(self, pitch, ctx, nx=17, ny=19):
        """Location+ sobre una rejilla de ubicaciones (vista del catcher, pies)."""
        xs, ys = np.linspace(-1.6, 1.6, nx), np.linspace(0.6, 4.2, ny)
        gx, gy = np.meshgrid(xs, ys)
        df, ars = self.frame(pitch, ctx=ctx, n=gx.size)
        df["PlateLocSide"], df["PlateLocHeight"] = gx.ravel(), gy.ravel()
        s, _ = self.score("location", df, ars)
        return xs, ys, s.reshape(gy.shape)

    def out_of_range(self, pitch):
        t = pitch["AutoPitchType"]
        if t not in self.ranges.index:
            return []
        r = self.ranges.loc[t]
        msgs = []
        for f, lab in [("RelSpeed", "velocidad"), ("SpinRate", "spin"), ("InducedVertBreak", "IVB"), ("HorzBreak", "HB"),
                       ("Extension", "extensión"), ("RelHeight", "altura de release")]:
            lo, hi = r[(f, 0.01)], r[(f, 0.99)]
            if not lo <= pitch[f] <= hi:
                msgs.append(f"{lab} {pitch[f]:.1f} fuera del rango observado para {t} ({lo:.1f}–{hi:.1f})")
        return msgs

    # ---------------- verificación ----------------
    def verify(self, k=25, seed=0):
        """Reproduce los scores oficiales de 18 para pitches reales (fila completa del bundle).
        Modo ligero: los mismos k pitches y sus scores oficiales vienen guardados en el paquete."""
        if self.lite:
            V = self.L["verify"]
            return {fam: float(np.max(np.abs(self.score(fam, V["rows"], V["ars"])[0] - V[fam]))) for fam in ["stuff", "location", "pitching"]}
        sc = pd.read_csv(ROOT / "18_three_scores.csv", usecols=["window", "stuff_plus", "location_plus", "pitching_plus"])
        sc = sc[sc.window == WINDOW].reset_index(drop=True)
        idx = np.random.default_rng(seed).choice(len(self.F), k, replace=False)
        out = {}
        for fam in ["stuff", "location", "pitching"]:
            s, _ = self.score(fam, self.F.iloc[idx], self.entry["f_ars"][idx])
            out[fam] = float(np.max(np.abs(s - sc[f"{fam}_plus"].to_numpy()[idx])))
        return out


_ENGINE = None


def get():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = Engine()
    return _ENGINE


if __name__ == "__main__":
    e = get()
    print("max |diff| vs scores oficiales de 18:", e.verify())
