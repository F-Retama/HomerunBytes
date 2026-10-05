"""Carga de las tablas generadas por 23_dashboard_data.py y estado persistente del usuario."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dashboard" / "data"
STATE = ROOT / "dashboard" / "state" / "state.json"
TABLES = ["pitchers", "arsenal", "fingerprint", "inbox", "targets", "trends", "teams", "history", "history_by_type"]


@st.cache_data(show_spinner=False)
def _read(name, mtime):
    return pd.read_parquet(DATA / f"{name}.parquet")


def load(name):
    """Tabla cacheada; se invalida sola cuando 23_dashboard_data.py regenera el archivo."""
    return _read(name, (DATA / f"{name}.parquet").stat().st_mtime)


def data_ready():
    return all((DATA / f"{t}.parquet").exists() for t in TABLES)


def read_state():
    try:
        return json.loads(STATE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write_state(**kw):
    s = read_state(); s.update(kw)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=2, ensure_ascii=False))


def seasons():
    return sorted(load("pitchers")["year"].unique(), reverse=True)


def roster_options(year):
    t = load("teams"); t = t[(t.year == year) & t.team_label.notna()]
    teams_ = (t.groupby(["team_label", "team_home_env"]).size().reset_index(name="n")
              .sort_values(["team_home_env", "team_label"], key=lambda s: s.map({"Extreme Altitude": 0, "Medium Altitude": 1, "No Altitude": 2}) if s.name == "team_home_env" else s))
    return ["Lista personalizada"] + teams_["team_label"].tolist() + ["Liga completa"]


def roster_ids(year, source):
    p = load("pitchers"); p = p[p.year == year]
    if source == "Liga completa":
        return p.loc[p.n >= 100, "pitcher_anon_id"].tolist()
    if source == "Lista personalizada":
        return [x for x in read_state().get("custom_roster", []) if x in set(p.pitcher_anon_id)]
    t = load("teams")
    return t[(t.year == year) & (t.team_label == source)]["pitcher_anon_id"].tolist()


def season_and_roster(container):
    """Selector de temporada y de roster (persistidos). Devuelve (year, source, ids)."""
    st_ = read_state()
    yrs = seasons()
    y0 = int(st.query_params.get("season", st_.get("season", yrs[0])))
    c1, c2 = container.columns([1, 1.6])
    year = c1.selectbox("Temporada", yrs, index=yrs.index(y0) if y0 in yrs else 0, key="season_sel")
    opts = roster_options(year)
    default_src = st_.get("roster_source") if st_.get("roster_source") in opts else next((o for o in opts if "Extreme" in o), opts[0])
    source = c2.selectbox("Roster", opts, index=opts.index(default_src), key="roster_sel",
                          help="El dataset es anónimo: los equipos se reconstruyen por co-aparición y se etiquetan por su entorno local. "
                               "En producción el club carga su roster real como 'Lista personalizada'.")
    if year != st_.get("season") or source != st_.get("roster_source"):
        write_state(season=int(year), roster_source=source)
    return int(year), source, roster_ids(year, source)


# ---------------- experimentos guardados (Simulator) ----------------
EXPERIMENTS = ROOT / "dashboard" / "state" / "experiments.json"


def list_experiments():
    try:
        return json.loads(EXPERIMENTS.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_experiment(exp):
    exps = list_experiments()
    exps.append(exp)
    EXPERIMENTS.parent.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS.write_text(json.dumps(exps, indent=2, ensure_ascii=False, default=float))


def delete_experiments(ids):
    exps = [e for e in list_experiments() if e["id"] not in set(ids)]
    EXPERIMENTS.write_text(json.dumps(exps, indent=2, ensure_ascii=False, default=float))


# ---------------- shortlist (Scouting) ----------------
def get_shortlist():
    return read_state().get("shortlist", [])


def save_shortlist(items):
    write_state(shortlist=items)


def add_to_shortlist(entries):
    cur = get_shortlist()
    keys = {e["key"] for e in cur}
    cur += [e for e in entries if e["key"] not in keys]
    save_shortlist(cur)
    return len(cur)


def has(name):
    return (DATA / f"{name}.parquet").exists()


def movement_map(year, min_n=50, q=0.20):
    """Mapa de movimiento de la liga: una fila por pitcher x tipo (>= min_n pitches) con su movimiento a
    nivel del mar y en Harp Helú (convención de diestro) y su grupo de efectividad: quintil alto o bajo
    de Stuff+ proyectado en Harp Helú dentro de su tipo de pitch."""
    a = load("arsenal")
    a = a[(a.year == year) & (a.n >= min_n) & a.stuff_harp.notna()]
    f = load("fingerprint")
    f = f[f.year == year].pivot_table(index=["pitcher_anon_id", "AutoPitchType"], columns="env", values=["ivb", "hb"])
    f.columns = [f"{v}_{'harp' if e == 'Extreme Altitude' else 'mar'}" for v, e in f.columns]
    d = a.merge(f.reset_index(), on=["pitcher_anon_id", "AutoPitchType"], how="inner")
    d["grupo"] = "Resto"
    for _t, g in d.groupby("AutoPitchType"):
        if len(g) < 20:
            continue
        lo, hi = g.stuff_harp.quantile([q, 1 - q])
        d.loc[g.index[g.stuff_harp >= hi], "grupo"] = "Más efectivos"
        d.loc[g.index[g.stuff_harp <= lo], "grupo"] = "Menos efectivos"
    return d[["pitcher_anon_id", "AutoPitchType", "n", "usage", "velo", "spin", "stuff_mar", "stuff_harp", "delta",
              "ivb_mar", "hb_mar", "ivb_harp", "hb_harp", "grupo"]]
