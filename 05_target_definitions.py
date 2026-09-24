"""Auditoría descriptiva de targets; no entrena modelos ni modifica el dataset.

Por defecto usa TODOS los pitches del parquet original, antes de filtros V1.
Las conversiones son vistas analíticas explícitas: se reportan valores no
numéricos, no finitos y no binarios; nunca se imputan ni corrigen los originales.
Las tasas excluyen eventos desconocidos y muestran siempre su denominador.
"""

import argparse
import hashlib
import io
import platform
import uuid
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
CATEGORIES = ["PitchCall", "KorBB", "play_result", "hit_type"]
BINARIES = [
    "is_swing", "is_whiff", "is_contact", "is_called_strike",
    "is_swinging_strike", "is_ball_in_play", "is_batted", "is_hit",
    "single", "double", "triple", "home_run", "base_on_balls", "strikeout",
    "is_hit_by_pitch", "in_strike_zone", "outside_strike_zone",
    "swung_outside_strike_zone",
]
NUMERICS = ["ExitSpeed", "Angle", "RunsScored", "OutsOnPlay"]
EXPECTED = CATEGORIES + NUMERICS + BINARIES
PERCENTILES = [.01, .05, .10, .25, .50, .75, .90, .95, .99]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_targets(df):
    """Return a reproducible text report, leaving df unchanged."""
    report = io.StringIO()

    def emit(value=""):
        print(value, file=report)

    def heading(title):
        emit("\n" + "=" * 72)
        emit(title)
        emit("=" * 72)

    def table(value):
        emit(value.to_string(float_format=lambda x: f"{x:.6f}"))

    def available(columns):
        missing = [c for c in columns if c not in df]
        if missing:
            emit("SKIP — faltan columnas: " + ", ".join(missing))
        return not missing

    def counts(series):
        # value_counts(dropna=False) conserva las categorías originales y NaN.
        n = series.value_counts(dropna=False)
        table(pd.DataFrame({"n": n, "pct": n / len(series) * 100
                            if len(series) else np.nan}))

    heading("A. COLUMN AVAILABILITY")
    emit(f"Dataset completo: {len(df)} filas; {len(df.columns)} columnas; sin filtros V1.")
    table(pd.DataFrame([
        {"column": c, "status": "OK" if c in df else "MISSING",
         "dtype": str(df[c].dtype) if c in df else "—"} for c in EXPECTED
    ]).set_index("column"))
    emit(f"Encontradas: {sum(c in df for c in EXPECTED)}/{len(EXPECTED)}")
    emit("Faltantes: " + (", ".join(c for c in EXPECTED if c not in df) or "ninguna"))

    heading("B. REAL CATEGORIES — value_counts(dropna=False)")
    for c in CATEGORIES:
        emit("\n" + c)
        if available([c]):
            counts(df[c])

    heading("C. BINARY TARGETS AND NUMERIC QUALITY")
    emit("Se aceptan 0/1 y bool; cadenas numéricas se convierten explícitamente.")
    emit("Cadenas true/false (sin distinguir mayúsculas) se interpretan explícitamente como 1/0.")
    emit("Otros valores se reportan y excluyen de cálculos; NaN nunca se convierte en cero.")
    numeric = {}
    quality = []
    for c in BINARIES + NUMERICS:
        if c not in df:
            continue
        raw = df[c]
        numeric_input = raw
        if not pd.api.types.is_numeric_dtype(raw.dtype):
            numeric_input = raw.astype(object).map(
                lambda x: {"true": 1, "false": 0}.get(x.lower(), x)
                if c in BINARIES and isinstance(x, str) else x
            )
            emit(f"CONVERSION EXPLICITA {c}: dtype={raw.dtype}; "
                 f"valores no nulos={raw.notna().sum()}; pd.to_numeric en copia analítica.")
        converted = pd.to_numeric(numeric_input, errors="coerce")
        values = pd.Series(converted, index=df.index, dtype="float64")
        nonnumeric = raw.notna() & values.isna()
        nonfinite = values.notna() & ~np.isfinite(values)
        nonbinary = (values.notna() & np.isfinite(values) & ~values.isin([0, 1])) if c in BINARIES else pd.Series(False, index=df.index)
        bad = nonnumeric | nonfinite | nonbinary
        numeric[c] = values.mask(bad)
        quality.append({"column": c, "non_null": int(raw.notna().sum()),
                        "missing_pct": raw.isna().mean() * 100,
                        "non_numeric": int(nonnumeric.sum()), "non_finite": int(nonfinite.sum()),
                        "non_binary": int(nonbinary.sum()), "valid_n": int(numeric[c].notna().sum()),
                        "positive_rate": numeric[c].mean() if c in BINARIES else np.nan})
        if bad.any():
            emit(f"ANOMALIA {c}: valores excluidos (frecuencias originales):")
            counts(raw[bad])
        if c in ["RunsScored", "OutsOnPlay"]:
            upper = 4 if c == "RunsScored" else 3
            unusual = values.notna() & ((values < 0) | (values > upper) | values.mod(1).ne(0))
            emit(f"{c}: fuera de entero [0,{upper}] = {int(unusual.sum())}; no se recortan.")
    if quality:
        table(pd.DataFrame(quality).set_index("column"))

    heading("D. LOGICAL CONSISTENCY")
    checks = [
        ("is_whiff != is_swinging_strike", "is_whiff", "is_swinging_strike", lambda a,b: a.ne(b)),
        ("whiff sin swing", "is_whiff", "is_swing", lambda a,b: a.eq(1) & b.eq(0)),
        ("contact sin swing", "is_contact", "is_swing", lambda a,b: a.eq(1) & b.eq(0)),
        ("called strike con swing", "is_called_strike", "is_swing", lambda a,b: a.eq(1) & b.eq(1)),
        ("chase dentro de zona", "swung_outside_strike_zone", "in_strike_zone", lambda a,b: a.eq(1) & b.eq(1)),
        ("chase sin outside zone", "swung_outside_strike_zone", "outside_strike_zone", lambda a,b: a.eq(1) & b.eq(0)),
        ("chase sin swing", "swung_outside_strike_zone", "is_swing", lambda a,b: a.eq(1) & b.eq(0)),
        ("whiff y contact simultáneos", "is_whiff", "is_contact", lambda a,b: a.eq(1) & b.eq(1)),
    ]
    for label, a, b, predicate in checks:
        if available([a, b]):
            valid = numeric[a].notna() & numeric[b].notna()
            emit(f"{label}: inconsistencias={int((predicate(numeric[a], numeric[b]) & valid).sum())}; "
                 f"comparables={int(valid.sum())}; no evaluables={int((~valid).sum())}")

    def rate(name, event, universe):
        if not available([event]):
            return
        eligible = universe & numeric[event].notna()
        events = int((eligible & numeric[event].eq(1)).sum())
        denominator = int(eligible.sum())
        value = events / denominator if denominator else np.nan
        emit(f"{name}: {value:.8f} ({value * 100:.4f}%)")
        emit(f"Universo elegible={int(universe.sum())}; evento válido/denominador={denominator}; "
             f"eventos={events}; evento ausente/inválido={int((universe & ~eligible).sum())}")
        if not denominator:
            emit("NO ESTIMABLE: denominador cero.")

    heading("E. WHIFF — TWO DIFFERENT DENOMINATORS")
    rate("P(whiff per pitch)", "is_whiff", pd.Series(True, index=df.index))
    if available(["is_swing"]):
        rate("P(whiff | swing)", "is_whiff", numeric["is_swing"].eq(1))
        emit(f"Pitches con swing desconocido/inválido: {numeric['is_swing'].isna().sum()}")
    heading("F. CHASE — OUTSIDE ZONE ONLY")
    if available(["outside_strike_zone"]):
        rate("P(chase | pitch outside zone)", "swung_outside_strike_zone", numeric["outside_strike_zone"].eq(1))
        emit(f"Pitches con outside zone desconocido/inválido: {numeric['outside_strike_zone'].isna().sum()}")

    def contact_groups(mask, key):
        if not available([key]):
            return
        # Clave original: no colapsar categorías inesperadas ni omitir missing.
        frame = pd.DataFrame({"group": df.loc[mask, key].astype(object)})
        aggregations = {"n": ("group", "size")}
        for column, prefix in [("ExitSpeed", "exit_speed"), ("Angle", "angle")]:
            if column in numeric:
                frame[column] = numeric[column].loc[mask]
                aggregations[prefix + "_count"] = (column, "count")
                aggregations[prefix + "_mean"] = (column, "mean")
                aggregations[prefix + "_median"] = (column, "median")
            else:
                emit("MISSING: " + column)
        table(frame.groupby("group", dropna=False, sort=False, observed=True).agg(**aggregations))

    heading("G. BATTED BALLS — HIT TYPE")
    batted = numeric["is_batted"].eq(1) if "is_batted" in numeric else None
    if batted is None:
        emit("SKIP G/H/I — falta is_batted; no se inventa un universo sustituto.")
    else:
        emit(f"Batted balls: {int(batted.sum())}; is_batted desconocido/inválido: {numeric['is_batted'].isna().sum()}")
        if available(["hit_type"]):
            counts(df.loc[batted, "hit_type"])
            emit("\nAngle / ExitSpeed por hit_type (n incluye filas sin medición):")
            contact_groups(batted, "hit_type")

    heading("H. CONTACT DISTRIBUTIONS — BATTED BALLS")
    if batted is not None:
        for c in ["ExitSpeed", "Angle"]:
            if available([c]):
                emit("\n" + c)
                table(numeric[c].loc[batted].describe(percentiles=PERCENTILES))
                emit(f"Missing/inválidos en batted balls: {numeric[c].loc[batted].isna().sum()}")
    else:
        emit("SKIP — falta is_batted.")

    heading("I. CONTACT METRICS BY OUTCOME — BATTED BALLS")
    if batted is not None:
        for c in ["is_hit", "home_run", "OutsOnPlay"]:
            emit("\n" + c)
            contact_groups(batted, c)
    else:
        emit("SKIP — falta is_batted.")

    heading("J. OUTS / PA STRUCTURE — ALL PITCHES")
    if available(["OutsOnPlay"]):
        counts(df["OutsOnPlay"])
    emit("\nstrikeout × OutsOnPlay (incluye ausentes; frecuencias de valores originales)")
    if available(["strikeout", "OutsOnPlay"]):
        # repr diferencia cadenas de números y evita mezclar NaN con una etiqueta real.
        label = lambda x: "<MISSING>" if pd.isna(x) else repr(x)
        table(pd.crosstab(df["strikeout"].map(label), df["OutsOnPlay"].map(label), dropna=False))
    emit("Columnas potencialmente relacionadas con PA (solo inventario, no definición terminal):")
    emit(", ".join(c for c in df if any(s in c.lower() for s in ["inning", "batter", "game", "plate", "pitchof", "pa_", "outs", "balls", "strikes"])) or "ninguna")

    heading("K. RUN VALUE RAW MATERIAL — ALL PITCHES")
    if available(["RunsScored"]):
        counts(df["RunsScored"])
    emit("\nplay_result")
    if available(["play_result"]):
        counts(df["play_result"])

    heading("L. SEMANTIC CROSS-CHECKS — NO RECODING")
    # Estas tablas ayudan a distinguir coherencia interna de significado validado.
    if "PitchCall" in df:
        cols = [c for c in ["is_swing", "is_whiff", "is_contact", "is_called_strike", "is_ball_in_play", "is_batted"] if c in numeric]
        frame = pd.DataFrame({"PitchCall": df["PitchCall"].astype(object), **{c: numeric[c] for c in cols}})
        aggs = {"n": ("PitchCall", "size"), **{c + "_mean": (c, "mean") for c in cols}}
        table(frame.groupby("PitchCall", dropna=False, sort=False).agg(**aggs))
    if batted is not None and "hit_type" in df:
        emit("\nhit_type fuera de is_batted == 1 (incluye indicador desconocido):")
        counts(df.loc[~batted, "hit_type"])
    emit("\nComparaciones semánticas adicionales (no sustituyen etiquetas):")
    for flag, column, labels in [
        ("is_whiff", "PitchCall", ["StrikeSwinging"]),
        ("is_called_strike", "PitchCall", ["StrikeCalled"]),
        ("is_hit_by_pitch", "PitchCall", ["HitByPitch"]),
        ("is_hit_by_pitch", "play_result", ["HitByPitch"]),
        ("is_hit", "play_result", ["Single", "Double", "Triple", "HomeRun"]),
        ("strikeout", "KorBB", ["Strikeout"]),
        ("base_on_balls", "KorBB", ["Walk"]),
    ]:
        if flag in numeric and column in df:
            valid = numeric[flag].notna() & df[column].notna()
            reference = df[column].isin(labels)
            emit(f"{flag} vs {column} in {labels}: "
                 f"diferencias={int((valid & numeric[flag].ne(reference.astype(int))).sum())}; "
                 f"comparables={int(valid.sum())}; no evaluables={int((~valid).sum())}")
    if batted is not None and "is_ball_in_play" in numeric:
        emit(f"is_batted=1 pero is_ball_in_play=0: {int((batted & numeric['is_ball_in_play'].eq(0)).sum())}")
    if "PitchCall" in df:
        for label in ["StrikeCalled", "StrikeSwinging", "HitByPitch"]:
            for flag in ["is_swing", "is_contact", "is_batted"]:
                if flag in numeric:
                    emit(f"PitchCall={label} y {flag}=1: {int((df['PitchCall'].eq(label) & numeric[flag].eq(1)).sum())}")

    def anomaly(mask, categories, flags):
        emit(f"Rows: {int(mask.sum())}")
        for c in categories:
            emit("\n" + c + ":")
            if available([c]):
                counts(df.loc[mask, c])
        emit("\nFlags (combinaciones originales, incluyendo NaNs):")
        present = [c for c in flags if c in df]
        available(flags)
        if present:
            table(df.loc[mask, present].value_counts(dropna=False))

    heading("M. CALLED STRIKE WITH SWING")
    if available(["is_called_strike", "is_swing"]):
        anomaly(numeric["is_called_strike"].eq(1) & numeric["is_swing"].eq(1),
                ["PitchCall", "play_result"],
                ["is_whiff", "is_contact", "is_swinging_strike", "is_ball_in_play"])

    heading("N. WHIFF AND CONTACT SIMULTANEOUS")
    if available(["is_whiff", "is_contact"]):
        anomaly(numeric["is_whiff"].eq(1) & numeric["is_contact"].eq(1),
                ["PitchCall", "play_result", "hit_type"],
                ["is_swing", "is_swinging_strike", "is_ball_in_play", "is_batted"])

    heading("O. HIT BY PITCH AUDIT")
    if available(["PitchCall", "play_result"]):
        # Unknown labels remain unknown: they are not treated as non-HBP.
        call = df["PitchCall"].astype("string").eq("HitByPitch")
        result = df["play_result"].astype("string").eq("HitByPitch")
        emit(f"PitchCall HitByPitch: {int(call.sum())}")
        emit(f"play_result HitByPitch: {int(result.sum())}")
        emit(f"PitchCall HBP but play_result not HBP: {int((call & ~result).sum())}")
        emit(f"play_result HBP but PitchCall not HBP: {int((~call & result).sum())}")
        emit(f"PitchCall HBP with unknown play_result: {int((call & result.isna()).sum())}")
        emit(f"play_result HBP with unknown PitchCall: {int((result & call.isna()).sum())}")
        # fillna only selects rows for this diagnostic; it never changes labels.
        union = (call | result).fillna(False)
        emit(f"HBP union rows: {int(union.sum())}")
        emit("\nCross-tab (solo categorías observadas; NaNs explícitos):")
        label = lambda x: "<MISSING>" if pd.isna(x) else repr(x)
        table(pd.crosstab(df.loc[union, "PitchCall"].astype(object).map(label),
                          df.loc[union, "play_result"].astype(object).map(label), dropna=False))

    heading("P. GROUND BALL — BALLS IN PLAY")
    if available(["is_ball_in_play"]):
        bip = numeric["is_ball_in_play"].eq(1)
        emit(f"Eligible BIP rows: {int(bip.sum())}")
        emit(f"BIP flag unknown/invalid: {numeric['is_ball_in_play'].isna().sum()}")
        if available(["hit_type"]):
            counts(df.loc[bip, "hit_type"])
            emit("\nBunts — describe (solo BIP):")
            bunts = bip & df["hit_type"].eq("Bunt")
            columns = [c for c in ["is_hit", "OutsOnPlay", "ExitSpeed", "Angle"] if c in numeric]
            available(["is_hit", "OutsOnPlay", "ExitSpeed", "Angle"])
            if columns:
                table(pd.DataFrame({c: numeric[c].loc[bunts] for c in columns}).describe(include="all"))
            emit("Universo V1 cerrado: BIP con hit_type conocido y no bunt; auditoría de cobertura en R.")

    heading("Q. BIP WITH UNKNOWN HIT TYPE")
    if available(["is_ball_in_play", "hit_type"]):
        unknown = numeric["is_ball_in_play"].eq(1) & df["hit_type"].isna()
        emit(f"Rows: {int(unknown.sum())}")
        columns = ["PitchCall", "play_result", "is_hit", "OutsOnPlay", "ExitSpeed", "Angle"]
        available(columns)
        present = [c for c in columns if c in df]
        if present:
            table(df.loc[unknown, present].describe(include="all"))
        for c in ["PitchCall", "play_result"]:
            if c in df:
                emit("\n" + c + ":")
                counts(df.loc[unknown, c])

    heading("R. GROUNDBALL ELIGIBLE COHORT — CLOSED V1")
    if available(["is_ball_in_play", "hit_type"]):
        bip = numeric["is_ball_in_play"].eq(1)
        eligible = bip & df["hit_type"].notna() & df["hit_type"].ne("Bunt")
        emit(f"Eligible GB rows: {int(eligible.sum())}")
        coverage = eligible.sum() / bip.sum() if bip.any() else np.nan
        emit(f"Coverage over BIP: {coverage:.8f} ({coverage * 100:.4f}%)")
        emit(f"Excluded unknown hit_type: {int((bip & df['hit_type'].isna()).sum())}")
        emit(f"Excluded Bunt: {int((bip & df['hit_type'].eq('Bunt')).sum())}")
        counts(df.loc[eligible, "hit_type"])
        # Report new categories explicitly rather than silently approving them as negatives.
        unexpected = eligible & ~df["hit_type"].isin(["GroundBall", "FlyBall", "LineDrive", "Popup"])
        emit(f"Unexpected eligible hit_type rows: {int(unexpected.sum())}")
        if unexpected.any():
            counts(df.loc[unexpected, "hit_type"])
            emit("REVISAR categorías nuevas antes de fijar negativos del target.")
        emit("Cohorte descriptiva: no se imputan hit_type ni se exportan nuevas etiquetas.")

    heading("S. CONTACT — BIP, KNOWN HIT TYPE, NON-BUNT, MEASURED")
    if available(["is_ball_in_play", "hit_type", "ExitSpeed", "Angle"]):
        base = (numeric["is_ball_in_play"].eq(1) & df["hit_type"].notna()
                & df["hit_type"].ne("Bunt"))
        contact = base & numeric["ExitSpeed"].notna() & numeric["Angle"].notna()
        emit(f"Eligible contact rows: {int(contact.sum())}")
        emit(f"Excluded missing/invalid measurements from GB cohort: {int((base & ~contact).sum())}")
        frame = pd.DataFrame({"hit_type": df.loc[contact, "hit_type"],
                              "ExitSpeed": numeric["ExitSpeed"].loc[contact],
                              "Angle": numeric["Angle"].loc[contact]})
        for c in ["ExitSpeed", "Angle"]:
            emit("\n" + c + " by hit_type")
            table(frame.groupby("hit_type", observed=True)[c].describe())
        emit("\nExitSpeed and Angle percentiles (marginal distributions on the same cohort):")
        table(frame[["ExitSpeed", "Angle"]].describe(percentiles=[.05,.10,.25,.50,.75,.90,.95]))
        emit("No se fijan thresholds; estos percentiles no constituyen definiciones de weak/hard contact.")

    heading("T. PLATE APPEARANCE IDENTIFIERS AND ORDER")
    matches = [c for c in df if any(term in c.lower() for term in ["pa", "atbat", "pitch", "plate"])]
    emit("Columns containing PA / atbat / pitch / plate:")
    emit("\n".join(matches) or "ninguna")
    # Search explicit identifiers/order, not names such as PlateLoc or PitchTrajectory.
    normalized = {c: ''.join(ch for ch in c.lower() if ch.isalnum()) for c in df}
    names = {"pa", "paid", "atbat", "atbatid", "atbatnumber", "plateappearance",
             "plateappearanceid", "plateappearancenumber", "pitchno", "pitchnumber",
             "pitchinpa", "pitchofpa", "pitchofatbat", "pitchinatbat", "timestamp",
             "datetime", "time", "date", "utctime", "localdatetime"}
    explicit = [c for c,n in normalized.items() if n in names]
    emit("Explicit PA/order/time candidates: " + (", ".join(explicit) or "ninguno"))
    context = [c for c in ["year", "game_anon_id", "batter_anon_id", "Inning", "Top/Bottom",
                           "Outs", "Balls", "Strikes", "count", "PitchUID"] if c in df]
    table(pd.DataFrame([{"column": c, "dtype": str(df[c].dtype),
                         "missing": int(df[c].isna().sum()),
                         "unique": int(df[c].nunique())} for c in dict.fromkeys(context + explicit)]))
    if "PitchUID" in df:
        emit(f"Duplicate non-null PitchUID rows: {df['PitchUID'].dropna().duplicated().sum()}")
        def uuid_version(value):
            if pd.isna(value):
                return "MISSING"
            try:
                return str(uuid.UUID(str(value)).version)
            except ValueError:
                return "not UUID"
        emit("PitchUID UUID versions:")
        counts(df["PitchUID"].map(uuid_version))
        emit("Un UUID identifica un pitch; incluso UUID v1 no demuestra límites de PA ni secuencia deportiva completa.")
    keys = ["game_anon_id", "batter_anon_id", "Inning", "Top/Bottom"]
    if available(keys):
        complete = df[keys].notna().all(axis=1)
        emit(f"Rows missing game/batter/inning/half keys: {int((~complete).sum())}")
        # Multiple recorded terminal outcomes in one candidate group refute its
        # uniqueness as a PA key. This is an audit, not a terminal-label definition.
        if all(c in numeric for c in ["is_ball_in_play", "strikeout", "base_on_balls"]):
            terminal = (numeric["is_ball_in_play"].eq(1) | numeric["strikeout"].eq(1)
                        | numeric["base_on_balls"].eq(1))
            if "PitchCall" in df:
                terminal = terminal | df["PitchCall"].eq("HitByPitch")
            grouped = df.loc[complete, keys].copy()
            grouped["terminal_candidate"] = terminal.loc[complete].astype(int)
            n = grouped.groupby(keys, observed=True)["terminal_candidate"].sum()
            emit(f"Game/batter/inning/half groups: {len(n)}")
            emit(f"Groups with >1 recorded terminal candidate: {int(n.gt(1).sum())}")
            emit("Terminal candidates = BIP OR strikeout OR walk OR PitchCall HBP; no se adopta esa unión como target.")
    emit("No se infiere el último pitch a partir del orden de filas, count ni UUID sin validar secuencia y límites de PA.")
    emit("Out-within-PA no construido: identificar outcomes por pitch no equivale a identificar una PA completa.")

    heading("U. FINAL V1 TARGET DEFINITIONS — CLOSED")
    emit("05_target_definitions.py — CERRADO por decisión metodológica ratificada.")
    emit("Incluidos: whiff_per_pitch (directo), whiff_given_swing (derivable), called_strike (directo), chase (condicional), groundball (derivable).")
    emit("Whiff por pitch: fuente is_whiff, sobre todos los pitches con evento conocido.")
    emit("Whiff dado swing: is_whiff entre is_swing == 1; estimando distinto, denominadores en E.")
    emit("Chase: swung_outside_strike_zone entre outside_strike_zone == 1.")
    emit('Called strike: fuente primaria PitchCall == "StrikeCalled". No se borran los 403 casos contradictorios.')
    emit("is_called_strike coincide con esa fuente en este dataset: cambiar la fuente no elimina la contradicción con swing/contact.")
    emit('HBP: is_hit_by_pitch descartado como fuente. Si se requiere, usar PitchCall == "HitByPitch"; documentar los 22 play_result=Walk.')
    emit("No se corrigen los 450 whiff+contact ni los 403 called strike+contact sin metadata de origen.")
    emit("Groundball V1 CERRADO: hit_type == GroundBall entre BIP con hit_type conocido y no Bunt. Ausentes excluidos, nunca negativos.")
    emit("Weak/hard contact: FUERA DE V1. Sin flag directo ni threshold defendible sin una convención adicional; no se fijan umbrales.")
    emit("Out within PA: FUERA DE V1. No hay identificador/orden de PA suficientemente confiable para identificar el pitch terminal.")
    emit("xRunValue: formalmente fuera de V1; RunsScored observado no sustituye run value esperado.")
    emit("is_batted no equivale a BIP y contiene inconsistencias: cada target de contacto requiere universo explícito.")
    emit("hit_type y Angle presentan cortes observados alrededor de 10/25/50 grados; procedencia no confirmada, no usarlos como evidencia independiente.")
    emit("Siguiente paso pendiente: 06_arsenal_incremental_value.py — Null, Physical Stuff y Physical Stuff + Historical Arsenal sobre las mismas filas elegibles season-forward para is_whiff.")
    emit("No se entrenan modelos, no se exportan columnas target, no se modifican datos ni se usan pesos externos.")
    return report.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=BASE_DIR / "stuff_model_df.parquet")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "05_target_definitions_audit.txt")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve() or args.output.suffix != ".txt":
        parser.error("La salida debe ser un .txt distinto del dataset de entrada.")
    if not args.input.is_file():
        parser.error(f"Dataset no encontrado: {args.input}. Colocar el parquet original junto al script.")
    df = pd.read_parquet(args.input)
    metadata = (f"05_target_definitions.py\nDataset: {args.input.name}\n"
                f"SHA256: {sha256(args.input)}\nPython: {platform.python_version()}\n"
                f"pandas: {pd.__version__}; numpy: {np.__version__}\n"
                "Scope: todos los registros originales; sin filtrar pitch type, pitcher o target.\n")
    text = metadata + audit_targets(df)
    args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"\nReporte guardado: {args.output}")


if __name__ == "__main__":
    main()
