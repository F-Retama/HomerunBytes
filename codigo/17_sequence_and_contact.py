"""Bloque B: orden de pitches, plate appearances, targets faltantes y run expectancy.

1. Orden. PitchUID es UUID v1 en 99.95% de las filas: su timestamp (100 ns) ordena
   los pitches dentro de cada juego. Los UUID sin timestamp (v4) no se pueden
   ubicar; la media entrada que los contiene se marca como no ordenable (sus PA
   no reciben prev_pitch_type ni out_within_PA, y no entran al run expectancy).
2. Plate appearances. Dentro de cada media entrada (game, Inning, Top/Bottom) una
   PA nueva empieza cuando cambia el bateador o el pitch anterior fue terminal
   (PitchCall InPlay/HitByPitch o KorBB Strikeout/Walk). Validacion: la PA empieza
   en 0-0, termina en un pitch terminal, y el conteo siguiente coincide con el
   que implica PitchCall (bola -> +1 bola, strike -> +1 strike, foul -> +1 strike
   con tope 2).
3. Targets (brief): prev_pitch_type (pitch anterior de la misma PA; 'Primer_pitch'
   si es el primero); out_within_PA (la PA del pitch termino en out: pitch terminal
   con OutsOnPlay >= 1 que no sea base por bolas/HBP; PA sin pitch terminal =
   desconocida, nunca negativa). Contacto sobre bolas en juego con ExitSpeed
   medido (convenciones externas de Statcast, documentadas): hard_hit = EV >= 95
   mph; weak_contact = EV < 60 mph (categoria 'Weak' de calidad de contacto);
   hr_given_contact = home_run sobre bolas en juego. Barrel NO se implementa: su
   regla oficial exige ademas xBA >= .500 y xSLG >= 1.500 con tablas de MLB que
   no tenemos, asi que no se puede replicar exactamente.
4. Run expectancy por conteo x outs (36 estados), estimado solo con datos LMB:
   RE(b, s, o) = media de las carreras desde ese pitch hasta el fin de la media
   entrada. Solo medias entradas completas (termina con 3 outs) y sin entradas
   9+ baja (pueden cortarse por walk-off). Sin corredores en base: el RE promedia
   sobre los estados de corredores. Valor realizado por pitch (perspectiva del
   pitcher): y = -(carreras en el pitch + RE(estado siguiente) - RE(estado actual)),
   con RE = 0 al terminar la media entrada. Reemplaza los pagos externos de The
   Book (13): ya no hay constantes de MLB 2006 en el valor.
5. H3 descriptivo por altitude_category: GB%, hard-hit%, HR/BIP, HR/FB y valor
   medio (runs, perspectiva del pitcher) de un roletazo frente a un elevado, con
   IC95% bootstrap por pitcher, y contraste within-pitcher Extreme - No.

Las funciones fit_re / realized_value_re / add_sequence se importan desde 18.
"""

import argparse
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
HALF = ["game_anon_id", "Inning", "Top/Bottom"]
TERMINAL_CALLS = ["InPlay", "HitByPitch"]
HARD_HIT_MPH, WEAK_MPH = 95.0, 60.0
ALTS = ["No Altitude", "Medium Altitude", "Extreme Altitude"]


def uuid_time(u):
    try:
        x = uuid.UUID(str(u))
    except (ValueError, TypeError):
        return np.nan
    return float(x.time) if x.version == 1 else np.nan


def add_sequence(df):
    """Ordena, reconstruye PA y agrega targets de secuencia. No elimina filas."""
    d = df.copy()
    d["uuid_time"] = d["PitchUID"].map(uuid_time)
    d["half_key"] = (d["game_anon_id"].astype(str) + "|" + d["Inning"].astype(int).astype(str)
                     + "|" + d["Top/Bottom"].astype(str))
    bad_half = set(d.loc[d["uuid_time"].isna(), "half_key"])
    d["half_orderable"] = ~d["half_key"].isin(bad_half)
    d = d.sort_values(["game_anon_id", "uuid_time"], na_position="last", kind="mergesort").reset_index(drop=True)
    d["seq_in_game"] = d.groupby("game_anon_id").cumcount()
    pc = d["PitchCall"].astype(str)
    kb = d["KorBB"].astype(str)
    d["is_terminal"] = pc.isin(TERMINAL_CALLS) | kb.isin(["Strikeout", "Walk"])
    d = d.sort_values(["game_anon_id", "half_key", "uuid_time"], na_position="last", kind="mergesort").reset_index(drop=True)
    gh = d.groupby("half_key", sort=False)
    first_in_half = gh.cumcount().eq(0)
    new_pa = (first_in_half | d["batter_anon_id"].ne(gh["batter_anon_id"].shift())
              | gh["is_terminal"].shift().fillna(True).astype(bool))
    d["pa_id"] = new_pa.cumsum()
    d["pa_pitch_number"] = d.groupby("pa_id").cumcount() + 1
    prev = gh["AutoPitchType"].shift().astype(str)
    d["prev_pitch_type"] = np.where(new_pa, "Primer_pitch", prev)
    d.loc[~d["half_orderable"], "prev_pitch_type"] = np.nan

    term = d[d["is_terminal"]].groupby("pa_id").tail(1)
    walk_hbp = term["KorBB"].astype(str).eq("Walk") | term["PitchCall"].astype(str).eq("HitByPitch")
    pa_out = pd.Series(((term["OutsOnPlay"] >= 1) & ~walk_hbp).astype(float).to_numpy(), index=term["pa_id"].to_numpy())
    d["out_within_PA"] = d["pa_id"].map(pa_out)
    d.loc[~d["half_orderable"], "out_within_PA"] = np.nan

    # Estado siguiente dentro de la media entrada; fin de media entrada -> RE = 0.
    d["next_b"] = gh["Balls"].shift(-1)
    d["next_s"] = gh["Strikes"].shift(-1)
    d["next_o"] = gh["Outs"].shift(-1)
    d["half_end"] = gh.cumcount(ascending=False).eq(0)
    last = d[d["half_end"]].set_index("half_key")
    ends3 = (last["Outs"] + last["OutsOnPlay"]).ge(3)
    walkoff_zone = (last["Inning"] >= 9) & last["Top/Bottom"].astype(str).eq("Bottom")
    d["half_complete"] = d["half_key"].map(ends3).astype(bool) & d["half_orderable"]
    d["half_usable_for_re"] = d["half_complete"] & ~d["half_key"].map(walkoff_zone).astype(bool)
    # Carreras desde este pitch hasta el fin de la media entrada (incluye las del propio pitch).
    d["runs_rest_of_half"] = d.iloc[::-1].groupby("half_key", sort=False)["RunsScored"].cumsum().iloc[::-1]
    # Valor realizado definido si la media entrada es ordenable y (hay siguiente pitch o termino con 3 outs o es walk-off).
    d["value_defined"] = d["half_orderable"] & (~d["half_end"] | d["half_key"].map(ends3 | walkoff_zone).astype(bool))

    bip = d["is_ball_in_play"].eq(1)
    ev = d["ExitSpeed"]
    measured = bip & ev.notna()
    d["hard_hit"] = np.where(measured, (ev >= HARD_HIT_MPH).astype(float), np.nan)
    d["weak_contact"] = np.where(measured, (ev < WEAK_MPH).astype(float), np.nan)
    d["hr_given_contact"] = np.where(bip, d["home_run"].astype(float), np.nan)
    return d


def fit_re(frame):
    """RE(b, s, o): media de carreras hasta el fin de la media entrada (36 estados)."""
    f = frame[frame["half_usable_for_re"]]
    re = f.groupby(["Balls", "Strikes", "Outs"])["runs_rest_of_half"].mean()
    return re.reindex(pd.MultiIndex.from_product([range(4), range(3), range(3)], names=["Balls", "Strikes", "Outs"]))


def fit_re_env(frame, k=500):
    """RE por entorno con shrinkage hacia el RE general: (n*RE_env + k*RE_pool)/(n + k),
    con n = pitches del estado en ese entorno. Solo con el frame recibido (historial)."""
    pool = fit_re(frame)
    f = frame[frame["half_usable_for_re"]]
    out = {}
    for alt in ALTS:
        g = f[f["altitude_category"].astype(str).eq(alt)].groupby(["Balls", "Strikes", "Outs"])["runs_rest_of_half"]
        m, n = g.mean().reindex(pool.index), g.size().reindex(pool.index).fillna(0)
        out[alt] = ((n * m.fillna(0) + k * pool) / (n + k)).rename("RE")
    return out, pool


def realized_value_re_env(frame, re_env, pool):
    """Valor por pitch con el RE del entorno del pitch (RE general si no hay categoria valida)."""
    y = realized_value_re(frame, pool)
    alt = frame["altitude_category"].astype(str).to_numpy()
    for name, re in re_env.items():
        m = alt == name
        if m.any():
            y[m] = realized_value_re(frame[m], re)
    return y


def realized_value_re(frame, re):
    """Valor por pitch desde la perspectiva del pitcher (runs). NaN si no esta definido."""
    lut = {k: v for k, v in re.items()}
    cur = np.array([lut.get((b, s, o), np.nan) for b, s, o in
                    zip(frame["Balls"].astype(int), frame["Strikes"].astype(int), frame["Outs"].astype(int).clip(upper=2))])
    nb, ns, no = frame["next_b"].to_numpy(), frame["next_s"].to_numpy(), frame["next_o"].to_numpy()
    nxt = np.array([0.0 if np.isnan(b) else lut.get((int(b), int(s), int(min(o, 2))), np.nan)
                    for b, s, o in zip(nb, ns, no)])
    y = -(frame["RunsScored"].to_numpy(float) + nxt - cur)
    return np.where(frame["value_defined"].to_numpy(bool), y, np.nan)


def boot_by_pitcher(values, groups, stat, B, rng):
    pit = pd.Index(pd.unique(groups)); codes = pit.get_indexer(groups)
    point = stat(np.ones(len(pit)), codes)
    reps = [stat(np.bincount(rng.integers(0, len(pit), len(pit)), minlength=len(pit)).astype(float), codes) for _ in range(B)]
    return point, np.nanpercentile(reps, 2.5), np.nanpercentile(reps, 97.5)


def weighted_rate(mask_num, mask_den):
    def stat(mult, codes):
        n = np.bincount(codes, weights=mask_num.astype(float), minlength=len(mult))
        d = np.bincount(codes, weights=mask_den.astype(float), minlength=len(mult))
        return (mult @ n) / (mult @ d)
    return stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    pd.set_option("display.width", 220)
    cols = ["PitchUID", "game_anon_id", "pitcher_anon_id", "batter_anon_id", "year", "Inning", "Top/Bottom", "Outs",
            "Balls", "Strikes", "PitchCall", "KorBB", "play_result", "hit_type", "ExitSpeed", "is_ball_in_play",
            "home_run", "OutsOnPlay", "RunsScored", "AutoPitchType", "altitude_category"]
    raw = pd.read_parquet(BASE_DIR / "stuff_model_df.parquet", columns=cols)
    d = add_sequence(raw)
    require = lambda c, m: None if c else (_ for _ in ()).throw(ValueError(m))
    require(len(d) == len(raw) and d["PitchUID"].is_unique, "La secuencia cambio el numero de filas o duplico PitchUID.")

    # ---------------- Validacion ----------------
    same_pa = d["pa_id"].eq(d["pa_id"].shift(-1))
    pc = d["PitchCall"].astype(str)
    b, s = d["Balls"], d["Strikes"]
    exp_b = np.where(pc.isin(["BallCalled", "BallinDirt"]), b + 1, b)
    exp_s = np.where(pc.isin(["StrikeCalled", "StrikeSwinging"]), s + 1,
                     np.where(pc.str.startswith("FoulBall"), np.minimum(s + 1, 2), s))
    known = same_pa & d["half_orderable"] & ~pc.eq("Undefined")
    count_ok = (d["next_b"].eq(exp_b) & d["next_s"].eq(exp_s))[known]
    first = d[d["pa_pitch_number"].eq(1) & d["half_orderable"]]
    pa_last = d[d["half_orderable"]].groupby("pa_id").tail(1)
    nxt_in_half = ~d["half_end"] & d["half_orderable"]
    outs_ok = d["next_o"].eq(d["Outs"] + d["OutsOnPlay"])[nxt_in_half]
    halves = d.groupby("half_key").tail(1)
    val = pd.DataFrame([
        ("pitches totales", len(d)),
        ("pitches con UUID v1 (ordenables)", int(d["uuid_time"].notna().sum())),
        ("medias entradas", d["half_key"].nunique()),
        ("medias entradas ordenables", int(halves["half_orderable"].sum())),
        ("medias entradas completas (3 outs)", int(halves["half_complete"].sum())),
        ("PA reconstruidas (ordenables)", int(first["pa_id"].nunique())),
        ("% PA que empiezan en 0-0", round(100 * (first["Balls"].eq(0) & first["Strikes"].eq(0)).mean(), 3)),
        ("% PA que terminan en pitch terminal", round(100 * pa_last["is_terminal"].mean(), 3)),
        ("% transiciones de conteo coherentes con PitchCall", round(100 * count_ok.mean(), 3)),
        ("% transiciones de outs coherentes (Outs + OutsOnPlay)", round(100 * outs_ok.mean(), 3)),
        ("% pitches con prev_pitch_type", round(100 * d["prev_pitch_type"].notna().mean(), 3)),
        ("% pitches con out_within_PA", round(100 * d["out_within_PA"].notna().mean(), 3)),
        ("tasa out_within_PA (por pitch)", round(d["out_within_PA"].mean(), 4)),
        ("% pitches con valor realizado definido", round(100 * d["value_defined"].mean(), 3)),
    ], columns=["chequeo", "valor"])
    print("VALIDACION DE SECUENCIA Y PA\n" + val.to_string(index=False))
    val.to_csv(BASE_DIR / "17_sequence_validation.csv", index=False)

    # Tasa de out por PA (unidad PA) con cruce contra play_result, como chequeo de sentido.
    term = d[d["is_terminal"] & d["half_orderable"]].groupby("pa_id").tail(1)
    tab = term.assign(out=term["out_within_PA"]).groupby(term["play_result"].astype(str))["out"].agg(["count", "mean"])
    print("\nout_within_PA por play_result del pitch terminal (debe ser ~1 en Out/Strikeout/FieldersChoice/Sacrifice y ~0 en hits/Walk/HBP):")
    print(tab.sort_values("count", ascending=False).round(3).to_string())

    # ---------------- Run expectancy ----------------
    re_rows = []
    for name, sel in [("todas", d["year"].notna())] + [(a, d["altitude_category"].astype(str).eq(a)) for a in ALTS]:
        re = fit_re(d[sel])
        for (bb, ss, oo), v in re.items():
            re_rows.append({"muestra": name, "balls": bb, "strikes": ss, "outs": oo, "RE": v})
    re_df = pd.DataFrame(re_rows)
    re_df.to_csv(BASE_DIR / "17_run_expectancy.csv", index=False)
    piv = re_df[re_df.muestra == "todas"].pivot_table(index=["outs", "balls"], columns="strikes", values="RE")
    print("\nRun expectancy LMB por conteo x outs (todas las temporadas, carreras hasta el fin de la media entrada):")
    print(piv.round(3).to_string())
    print("\nRE(0-0) por outs y altitud:")
    print(re_df[(re_df.balls == 0) & (re_df.strikes == 0)].pivot_table(index="muestra", columns="outs", values="RE").round(3).to_string())

    re_all = fit_re(d)
    d["value_re"] = realized_value_re(d, re_all)
    print(f"\nValor realizado (RE LMB, todas las temporadas; descriptivo): media {np.nanmean(d['value_re']):+.4f}, "
          f"SD {np.nanstd(d['value_re']):.4f} runs")

    # ---------------- H3 descriptivo ----------------
    bip = d[d["is_ball_in_play"].eq(1)].copy()
    ht = bip["hit_type"].astype(str)
    h3 = []
    for alt in ALTS:
        g = bip[bip["altitude_category"].astype(str).eq(alt)]
        gh = g["hit_type"].astype(str)
        grp = g["pitcher_anon_id"].to_numpy()
        known_ht = gh.isin(["GroundBall", "FlyBall", "LineDrive", "Popup"]).to_numpy()
        row = {"altitude_category": alt, "n_bip": len(g)}
        for nm, num, den in [("gb_rate", gh.eq("GroundBall").to_numpy() & known_ht, known_ht),
                             ("hard_hit_rate", g["hard_hit"].eq(1).to_numpy(), g["hard_hit"].notna().to_numpy()),
                             ("hr_per_bip", g["home_run"].eq(1).to_numpy(), np.ones(len(g), bool)),
                             ("hr_per_flyball", (g["home_run"].eq(1) & gh.eq("FlyBall")).to_numpy(), gh.eq("FlyBall").to_numpy())]:
            pt, lo, hi = boot_by_pitcher(None, grp, weighted_rate(num, den), args.bootstrap, rng)
            row.update({nm: pt, nm + "_lo": lo, nm + "_hi": hi})
        vv = g["value_re"].to_numpy()
        ok = ~np.isnan(vv)
        for nm, t in [("value_groundball", "GroundBall"), ("value_flyball", "FlyBall"), ("value_linedrive", "LineDrive")]:
            m = ok & gh.eq(t).to_numpy()
            row[nm] = float(vv[m].mean())

        def gap(mult, codes, vv=vv, gh=gh, ok=ok):
            gb, fb = ok & gh.eq("GroundBall").to_numpy(), ok & gh.eq("FlyBall").to_numpy()
            sg = np.bincount(codes, weights=np.where(gb, vv, 0), minlength=len(mult)); ng = np.bincount(codes, weights=gb, minlength=len(mult))
            sf = np.bincount(codes, weights=np.where(fb, vv, 0), minlength=len(mult)); nf = np.bincount(codes, weights=fb, minlength=len(mult))
            return (mult @ sg) / (mult @ ng) - (mult @ sf) / (mult @ nf)
        pt, lo, hi = boot_by_pitcher(None, grp, gap, args.bootstrap, rng)
        row.update({"gb_minus_fb_value": pt, "gb_minus_fb_lo": lo, "gb_minus_fb_hi": hi})
        h3.append(row)
    h3 = pd.DataFrame(h3)
    h3.to_csv(BASE_DIR / "17_contact_by_altitude.csv", index=False)
    show = h3[["altitude_category", "n_bip", "gb_rate", "hard_hit_rate", "hr_per_bip", "hr_per_flyball",
               "value_groundball", "value_flyball", "gb_minus_fb_value", "gb_minus_fb_lo", "gb_minus_fb_hi"]]
    print("\nH3 descriptivo por altitud (valor en runs desde la perspectiva del pitcher; IC95% bootstrap por pitcher):")
    print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Within-pitcher: HR/FB, hard-hit y ventaja GB-FB en Extreme vs No (celdas pitcher x anio con >= 30 BIP en ambos).
    rows = []
    bip["alt"] = bip["altitude_category"].astype(str)
    bip["is_fb"] = ht.eq("FlyBall").to_numpy()
    bip["is_gb"] = ht.eq("GroundBall").to_numpy()
    cell = (bip[bip["alt"].isin(["No Altitude", "Extreme Altitude"])]
            .groupby(["pitcher_anon_id", "year", "alt"])
            .agg(n=("PitchUID", "size"), hr=("home_run", "sum"), fb=("is_fb", "sum"), gb=("is_gb", "sum"),
                 hard=("hard_hit", "mean")).reset_index())
    wide = cell.pivot(index=["pitcher_anon_id", "year"], columns="alt", values=["n", "hr", "fb", "gb", "hard"]).dropna()
    wide = wide[(wide[("n", "No Altitude")] >= 30) & (wide[("n", "Extreme Altitude")] >= 30)]
    grp = wide.index.get_level_values(0).to_numpy()
    wts = np.minimum(wide[("n", "No Altitude")], wide[("n", "Extreme Altitude")]).to_numpy(float)
    for nm, num_e, den_e, num_n, den_n in [
            ("hr_per_bip", ("hr", "Extreme Altitude"), ("n", "Extreme Altitude"), ("hr", "No Altitude"), ("n", "No Altitude")),
            ("hr_per_flyball", ("hr", "Extreme Altitude"), ("fb", "Extreme Altitude"), ("hr", "No Altitude"), ("fb", "No Altitude")),
            ("gb_share_of_bip", ("gb", "Extreme Altitude"), ("n", "Extreme Altitude"), ("gb", "No Altitude"), ("n", "No Altitude"))]:
        with np.errstate(invalid="ignore", divide="ignore"):
            delta = (wide[num_e] / wide[den_e] - wide[num_n] / wide[den_n]).to_numpy(float)
        ok = np.isfinite(delta)

        def st(mult, codes, delta=delta, ok=ok):
            w = np.bincount(codes, weights=np.where(ok, wts, 0), minlength=len(mult))
            v = np.bincount(codes, weights=np.where(ok, wts * delta, 0), minlength=len(mult))
            return (mult @ v) / (mult @ w)
        pt, lo, hi = boot_by_pitcher(None, grp, st, args.bootstrap, rng)
        rows.append({"metrica": nm, "n_celdas": int(ok.sum()), "delta_extreme_minus_no": pt, "ci95_lo": lo, "ci95_hi": hi})
    hard_delta = (wide[("hard", "Extreme Altitude")] - wide[("hard", "No Altitude")]).to_numpy(float)
    okh = np.isfinite(hard_delta)

    def sth(mult, codes):
        w = np.bincount(codes, weights=np.where(okh, wts, 0), minlength=len(mult))
        v = np.bincount(codes, weights=np.where(okh, wts * hard_delta, 0), minlength=len(mult))
        return (mult @ v) / (mult @ w)
    pt, lo, hi = boot_by_pitcher(None, grp, sth, args.bootstrap, rng)
    rows.append({"metrica": "hard_hit_rate", "n_celdas": int(okh.sum()), "delta_extreme_minus_no": pt, "ci95_lo": lo, "ci95_hi": hi})
    wp = pd.DataFrame(rows)
    wp.to_csv(BASE_DIR / "17_contact_within_pitcher.csv", index=False)
    print("\nWithin-pitcher Extreme - No (celdas pitcher x anio, >= 30 BIP en ambos, ponderado por el menor n):")
    print(wp.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    keep = ["PitchUID", "uuid_time", "seq_in_game", "half_key", "half_orderable", "half_complete", "half_usable_for_re",
            "pa_id", "pa_pitch_number", "is_terminal", "prev_pitch_type", "out_within_PA", "next_b", "next_s", "next_o",
            "half_end", "value_defined", "runs_rest_of_half", "hard_hit", "weak_contact", "hr_given_contact"]
    d[keep].to_parquet(BASE_DIR / "17_pitch_sequence.parquet", index=False)
    print("\nArchivos: 17_pitch_sequence.parquet, 17_sequence_validation.csv, 17_run_expectancy.csv, "
          "17_contact_by_altitude.csv, 17_contact_within_pitcher.csv")


if __name__ == "__main__":
    main()
