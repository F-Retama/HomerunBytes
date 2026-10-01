"""Motor del Simulator: califica pitches hipotéticos con los modelos congelados de 18.

Trabaja directamente en convención de diestro (HB, RelSide y SpinAxis ya espejados), igual que el
frame del bundle: no se vuelve a espejar. El movimiento de entrada se interpreta como equivalente a
nivel del mar (No Altitude) y se lleva a cada entorno con los factores validados (20/21/22).
Verificado contra los scores oficiales de 18 en verify() (Stuff+, Location+ y Pitching+).
"""

import importlib.util
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WINDOW = "2024-2025 -> 2026"
SCALE = "scale_sin_contact_value"
ENVS = ["No Altitude", "Medium Altitude", "Extreme Altitude"]
ENV_LABEL = {"No Altitude": "Nivel del mar", "Medium Altitude": "Medium", "Extreme Altitude": "Harp Helú"}
SHOW = ["whiff_per_pitch", "chase", "called_strike"]


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class Engine:
    def __init__(self):
        self.t22 = _load("profile_recommendations_export", "22_profile_recommendations_export.py")
        self.t16, self.t18 = self.t22.t16, self.t22.t18
        self.cfg = self.t18.build_cfg(self.t16.load_modules())
        self.bundle = joblib.load(ROOT / "18_three_scores_bundle.pkl")
        self.entry = self.bundle[WINDOW]
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
        out = self.t22.to_env(df, "No Altitude", env, self.fac)
        out["altitude_category"] = pd.Categorical([env] * len(out), categories=self.cats["altitude_category"])
        return out

    # ---------------- scores ----------------
    def score(self, family, df, has_ars):
        fam = self.entry["families"][family]
        w = np.asarray(fam["weights"], float)
        need = {c for c, wi in zip(self.comps, w) if abs(wi) > 1e-9} | set(SHOW)
        probs = {c: self.t16.family_predict(fam["models"][c], df, has_ars, fam["variants"]) for c in self.comps if c in need}
        z = np.zeros(len(df))
        for c, wi in zip(self.comps, w):
            if abs(wi) > 1e-9:
                mu, sd = fam["stats"][c]
                z += wi * self.cfg["signs"].get(c, 1.0) * (probs[c] - mu) / sd
        m, s = fam["stats"][SCALE]
        return 100 + 10 * (z - m) / s, probs

    def altitude_table(self, pitch, fb=None, usage=None, ctx=None):
        """Stuff+ (y Pitching+ con ctx) del mismo pitch en los tres entornos."""
        base, ars = self.frame(pitch, fb, usage, ctx)
        rows = []
        for env in ENVS:
            x = self.to_env(base, env)
            s, p = self.score("stuff", x, ars)
            r = {"env": env, "label": ENV_LABEL[env], "stuff_plus": float(s[0]), "IVB": float(x["InducedVertBreak"].iloc[0]),
                 "HB": float(x["HorzBreak"].iloc[0]), **{f"p_{c}": float(p[c][0]) for c in SHOW}}
            if ctx is not None:
                pp, _ = self.score("pitching", x, ars)
                r["pitching_plus"] = float(pp[0])
            rows.append(r)
        return pd.DataFrame(rows)

    # ---------------- what-if sobre los pitches reales de un pitcher ----------------
    SHIFTABLE = {"RelSpeed": ["velo_diff_vs_fb"], "InducedVertBreak": ["ivb_diff_vs_fb"], "HorzBreak": ["hb_diff_vs_fb"],
                 "SpinRate": ["spin_diff_vs_fb"], "SpinAxis": [], "Extension": ["extension_diff_vs_fb"], "RelHeight": [], "RelSide": []}

    def pitcher_rows(self, pid, ptype):
        m = ((self.F["pitcher_anon_id"] == pid) & (self.F["AutoPitchType"].astype(str) == ptype)
             & self.F["altitude_category"].astype(str).isin(ENVS)).to_numpy()
        return self.F[m].reset_index(drop=True), self.entry["f_ars"][m]

    def sea_level(self, rows):
        """Cada pitch llevado a su equivalente a nivel del mar desde el entorno donde se lanzó."""
        out = rows.copy()
        alt = rows["altitude_category"].astype(str).to_numpy()
        for env in ["Medium Altitude", "Extreme Altitude"]:
            m = alt == env
            if m.any():
                out.loc[m] = self.t22.to_env(rows[m], env, "No Altitude", self.fac)
        return out

    def population_table(self, rows, ars, shifts=None, hybrid=True):
        """Stuff+ medio de los pitches reales del pitcher en cada entorno, con desplazamientos what-if
        (sumados a cada pitch a nivel del mar). hybrid=True (titular, guía 38): nivel = Stuff+ oficial a nivel
        del mar y efecto de altitud = variante física; con shifts vacíos reproduce la proyección de 22.
        hybrid=False: todo con la variante dada por ars (sensibilidad)."""
        sea = self._shift(self.sea_level(rows), shifts)
        return self._hybrid(sea, ars) if hybrid else self._score_envs(sea, ars)

    def _hybrid(self, sea, ars):
        off = self._score_envs(sea, ars).set_index("env")
        phy = self._score_envs(sea, np.zeros(len(sea), bool)).set_index("env")
        out = phy.copy()
        for col in ["stuff_plus"] + [f"p_{c}" for c in SHOW]:
            out[col] = off.loc["No Altitude", col] + phy[col] - phy.loc["No Altitude", col]
        return out.reset_index()

    def _shift(self, sea, shifts):
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
        rows, ars = self.pitcher_rows(pid, ptype)
        sea = self.sea_level(rows)
        m = {c: float(np.nanmean(sea[c])) if sea[c].notna().any() else np.nan for c in self.NUM_FEATS}
        m["SpinAxis"] = float(np.degrees(np.arctan2(np.nanmean(np.sin(np.radians(sea.SpinAxis))), np.nanmean(np.cos(np.radians(sea.SpinAxis))))) % 360)
        m["PitcherThrows"] = rows.PitcherThrows.astype(str).mode().iloc[0]
        return m, bool(ars.mean() >= 0.5), len(rows)

    def game_grid(self, pid, ptype, ctx, env, nx=13, ny=15):
        """Pitching+, probabilidades y daño esperado en contacto (runs/100, + = más daño) en una rejilla de ubicaciones."""
        m, has_ars, n = self.pitch_means(pid, ptype)
        xs, ys = np.linspace(-1.5, 1.5, nx), np.linspace(0.8, 4.0, ny)
        gx, gy = np.meshgrid(xs, ys)
        N = gx.size
        df = pd.DataFrame({c: np.full(N, m[c]) for c in self.NUM_FEATS + ["SpinAxis"]})
        df["AutoPitchType"], df["PitcherThrows"] = ptype, m["PitcherThrows"]
        df["BatterSide"], df["prev_pitch_type"] = ctx.get("BatterSide", "Right"), ctx.get("prev_pitch_type", "Primer_pitch")
        for k, dflt in [("Balls", 0), ("Strikes", 0), ("Outs", 0), ("Inning", 1)]:
            df[k] = ctx.get(k, dflt)
        df["PlateLocSide"], df["PlateLocHeight"] = gx.ravel(), gy.ravel()
        df["altitude_category"] = "No Altitude"
        if not has_ars:
            for c in ["velo_diff_vs_fb", "ivb_diff_vs_fb", "hb_diff_vs_fb", "spin_diff_vs_fb", "pitch_usage_pct", "release_distance_vs_fb", "extension_diff_vs_fb"]:
                df[c] = np.nan
        x = self.to_env(self._cat(df), env)
        ars = np.full(N, has_ars)
        pp, probs = self.score("pitching", x, ars)
        fam = self.entry["families"]["pitching"]
        cv = self.t16.family_predict(fam["models"]["contact_value"], x, ars, fam["variants"])
        shp = gy.shape
        return {"xs": xs, "ys": ys, "pitching_plus": pp.reshape(shp), "damage": (-100 * cv).reshape(shp),
                **{c: probs[c].reshape(shp) for c in SHOW}, "n": n, "has_ars": has_ars, "hand": m["PitcherThrows"]}

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
        """Reproduce los scores oficiales de 18 para pitches reales (fila completa del bundle)."""
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
