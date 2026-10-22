"""Parte una narración completa (un solo archivo, p. ej. exportado de ElevenLabs) en los segmentos de narracion.json.

Busca los cortes entre las pausas del audio con programación dinámica: elige, entre todas las pausas, las que hacen que
la duración de cada segmento sea proporcional a la de una referencia (la misma frase leída con `say`). Después acorta
las pausas internas largas para que el video quepa en 5:00.

Uso, desde la raíz del repo:
    python video/partir_voz.py video/voz/narracion_completa.mp3          -> video/voz/<id>.wav (uno por segmento)
    python video/grabar_video.py --voz-propia video/voz

Revisa la tabla que imprime: el cociente de cada segmento debe estar cerca de 1 (0.85-1.15).
"""

import argparse
import json
import math
import re
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parent
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
PAUSA_MAX = 0.45  # pausas internas más largas que esto se recortan a esto (s)


def duracion(f):
    o = subprocess.run([FFMPEG, "-i", str(f)], capture_output=True, text=True).stderr
    h, m, s = re.search(r"Duration: (\d+):(\d+):([\d.]+)", o).groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


def silencios(f, db=-38, d=0.25):
    o = subprocess.run([FFMPEG, "-i", str(f), "-af", f"silencedetect=noise={db}dB:d={d}", "-f", "null", "-"],
                       capture_output=True, text=True).stderr
    return list(zip(map(float, re.findall(r"silence_start: ([\d.]+)", o)), map(float, re.findall(r"silence_end: ([\d.]+)", o))))


def referencia(cfg):
    out = {}
    with tempfile.TemporaryDirectory() as t:
        for s in cfg["segmentos"]:
            f = Path(t) / f"{s['id']}.aiff"
            subprocess.run(["say", "-v", cfg["voz"], "-r", str(cfg["ritmo"]), "-o", str(f), s["texto"]], check=True)
            out[s["id"]] = duracion(f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--salida", default=str(ROOT / "voz"))
    args = ap.parse_args()
    cfg = json.loads((ROOT / "narracion.json").read_text(encoding="utf-8"))
    ids = [s["id"] for s in cfg["segmentos"]]
    ref = referencia(cfg)
    T = duracion(args.audio)
    sil = [(a, b) for a, b in silencios(args.audio) if 1 < a and b < T - 1]
    pts = [0.0] + [(a + b) / 2 for a, b in sil] + [T]
    scale = T / sum(ref.values())
    n, K = len(pts), len(ids)
    INF = float("inf")
    cost = [[INF] * n for _ in range(K + 1)]
    back = [[-1] * n for _ in range(K + 1)]
    cost[0][0] = 0.0
    for k in range(1, K + 1):
        r = ref[ids[k - 1]] * scale
        for j in range(1, n):
            for i in range(j):
                if cost[k - 1][i] < INF:
                    c = cost[k - 1][i] + math.log((pts[j] - pts[i]) / r) ** 2 * r
                    if c < cost[k][j]:
                        cost[k][j], back[k][j] = c, i
    cut, j = [n - 1], n - 1
    for k in range(K, 0, -1):
        j = back[k][j]
        cut.append(j)
    b = [pts[c] for c in reversed(cut)]

    out = Path(args.salida)
    out.mkdir(parents=True, exist_ok=True)
    trim = "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.05"
    squeeze = f"silenceremove=stop_periods=-1:stop_duration={PAUSA_MAX}:stop_threshold=-38dB:stop_silence={PAUSA_MAX}"
    total = 0.0
    print(f"{'seg':>3} {'inicio':>7} {'fin':>7} {'dur':>5} {'ref':>5} {'cociente':>8} {'final':>5}")
    for k, sid in enumerate(ids):
        f = out / f"{sid}.wav"
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-ss", f"{b[k]:.3f}", "-to", f"{b[k + 1]:.3f}", "-i", args.audio,
                        "-ac", "1", "-ar", "48000", "-af", f"{trim},areverse,{trim},areverse,{squeeze}", str(f)], check=True)
        d = duracion(f)
        total += d
        print(f"{sid:>3} {b[k]:7.2f} {b[k + 1]:7.2f} {b[k + 1] - b[k]:5.1f} {ref[sid] * scale:5.1f} "
              f"{(b[k + 1] - b[k]) / (ref[sid] * scale):8.2f} {d:5.1f}")
    print(f"Original {T:.1f} s · segmentos recortados {total:.1f} s -> {out}")


if __name__ == "__main__":
    main()
