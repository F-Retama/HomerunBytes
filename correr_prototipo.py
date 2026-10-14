"""Prepara y abre el prototipo del dashboard. Funciona igual en macOS, Windows y Linux.

Uso, desde la carpeta del repo y con el entorno virtual activado:

    python correr_prototipo.py              corre solo los pasos cuyos resultados faltan, regenera 23 y 24,
                                            verifica el motor contra los scores oficiales de 18 y abre el dashboard
    python correr_prototipo.py --todo       vuelve a correr toda la cadena desde 02
    python correr_prototipo.py --sin-abrir  prepara todo pero no abre Streamlit

Requiere stuff_model_df.parquet en esta carpeta (material del hackathon; no viene en GitHub). Si alguien del equipo
comparte sus salidas ya generadas (17_*, 18_*, 19_cv_oof.parquet, 20_*, 21_*, 22_*), basta con copiarlas aqui y esos
pasos se omiten. Si no, la primera corrida tarda del orden de 30 minutos (18, 19 cv y 21 son los pasos largos).

Cadena minima para el dashboard (README): 02 -> 17 -> 18 -> 19 (solo la etapa cv) -> 20 -> 21 -> 22 -> 23 -> 24.
Los logs de cada paso quedan en prototipo_logs/.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "prototipo_logs"
# (nombre, salida que indica que el paso ya corrio o None para correrlo siempre, argumentos del script)
STEPS = [
    ("02_features", "stuff_model_features_v1.parquet", ["02_feature_engineering.py"]),
    ("17_secuencia", "17_pitch_sequence.parquet", ["17_sequence_and_contact.py"]),
    ("18_scores", "18_three_scores_bundle.pkl", ["18_three_scores_v2.py"]),
    ("19_cv", "19_cv_oof.parquet", ["19_validation.py", "--stage", "cv"]),
    ("20_fisica", "20_altitude_gradient.csv", ["20_altitude_physics.py"]),
    ("21_delta", "21_evidence_levels.csv", ["21_altitude_stuff_delta.py"]),
    ("22_perfil", "22_pitch_projection.parquet", ["22_profile_recommendations_export.py"]),
    ("23_datos", None, ["23_dashboard_data.py"]),
    ("24_arsenal", None, ["24_arsenal_harp.py"]),
    ("verificacion", None, [str(Path("dashboard") / "lib" / "engine.py")]),
]
# Archivos versionados que el pipeline reescribe (fechas y hashes): avisar para no subirlos por accidente.
VERSIONADOS = ["20_transform_choice.json", "21_transform_r.json", "22_dashboard_manifest.json"]
# UTF-8 en los subprocesos. En Windows, la salida redirigida a un archivo usa la codificacion local (cp1252) y fallaria
# con caracteres que imprimen los scripts, como "→" o "≥".
ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}


def check_env():
    v = sys.version_info
    if not (3, 9) <= (v.major, v.minor) <= (3, 13):
        print(f"Aviso: Python {v.major}.{v.minor}. El proyecto se probo con Python 3.9 a 3.13; "
              "scikit-learn 1.6.x (fijado en requirements.txt) no tiene versiones para 3.14 o posterior.")
    try:
        import sklearn
        import streamlit  # noqa: F401
    except ImportError as e:
        sys.exit(f"Falta la dependencia '{e.name}'. Activa el entorno virtual e instala: python -m pip install -r requirements.txt")
    if not sklearn.__version__.startswith("1.6."):
        print(f"Aviso: scikit-learn {sklearn.__version__}. requirements.txt fija 1.6.x para que los modelos guardados (.pkl) "
              "se puedan abrir en cualquier equipo; reinstala con: python -m pip install -r requirements.txt")
    if not (ROOT / "stuff_model_df.parquet").exists():
        sys.exit(f"Falta stuff_model_df.parquet en {ROOT}. Copialo del material del hackathon (no viene en GitHub).")


def run_step(name, out, args, force):
    if not force and out and (ROOT / out).exists():
        print(f"  - {name}: ya existe {out}, se omite")
        return
    log = LOGS / f"{name}.log"
    print(f">> {name} (inicio {time.strftime('%H:%M')}; log en {log.relative_to(ROOT)})", flush=True)
    t0 = time.time()
    with open(log, "wb") as fh:
        rc = subprocess.call([sys.executable, *args], cwd=ROOT, env=ENV, stdout=fh, stderr=subprocess.STDOUT)
    if rc != 0:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
        print(f"ERROR: {name} fallo (codigo {rc}). Ultimas lineas de {log.name}:")
        print("\n".join(tail))
        sys.exit(1)
    print(f"   listo en {(time.time() - t0) / 60:.0f} min", flush=True)


def warn_tracked():
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", *VERSIONADOS],
                             cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:  # git no instalado: el aviso es opcional
        return
    if out:
        print("\nAviso: el pipeline reescribio archivos versionados (no los subas salvo que quieras actualizar los del equipo):")
        print(out)
        print("Para descartarlos: git restore " + " ".join(VERSIONADOS))


def open_dashboard():
    print("\nAbriendo el dashboard en el navegador (normalmente http://localhost:8501). Ctrl+C para cerrarlo.")
    print("La primera vez, Streamlit pide un correo en la terminal: presiona Enter para dejarlo vacio.")
    try:
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(Path("dashboard") / "app.py")], cwd=ROOT)
    except KeyboardInterrupt:
        return 0


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description="Prepara y abre el prototipo del dashboard (macOS, Windows y Linux).")
    ap.add_argument("--todo", action="store_true", help="vuelve a correr toda la cadena desde 02, aunque existan los resultados")
    ap.add_argument("--sin-abrir", action="store_true", help="prepara todo pero no abre Streamlit")
    args = ap.parse_args()
    check_env()
    LOGS.mkdir(exist_ok=True)
    for name, out, step_args in STEPS:
        run_step(name, out, step_args, args.todo)
    ver = [l for l in (LOGS / "verificacion.log").read_text(encoding="utf-8", errors="replace").splitlines() if "max |diff|" in l]
    print("Verificacion del motor contra los scores oficiales de 18 (debe ser ~1e-14):")
    print("  " + (ver[-1] if ver else "(sin resultado; revisa prototipo_logs/verificacion.log)"))
    warn_tracked()
    if not args.sin_abrir:
        sys.exit(open_dashboard())


if __name__ == "__main__":
    main()
