"""Graba el video del entregable 4 recorriendo el dashboard con Playwright y le monta la narración sintética.

Requisitos: el dashboard corriendo (streamlit run dashboard/app.py --server.port 8599), playwright con Chromium
(python -m pip install playwright imageio-ffmpeg; python -m playwright install chromium) y macOS (voz con `say`).

Uso, desde la raíz del repo:
    python video/grabar_video.py               ensayo de calentamiento + grabación + montaje -> video/salida/video_entrega.mp4
    python video/grabar_video.py --voz-propia video/voz   igual, con la voz grabada por el equipo (un archivo por segmento)
    python video/grabar_video.py --remontar    rehace solo el montaje de audio sobre la última toma
    python video/grabar_video.py --ensayo      solo recorre el guion y guarda una captura por segmento en video/salida/ensayo/

La narración y las acciones están en video/narracion.json (un segmento = acciones antes, texto, acciones durante).
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "salida"
BASE = "http://localhost:8599"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
W, H = 1920, 1080
PAUSA_SEG, PAUSA_ESCENA = 0.3, 0.7  # silencio entre segmentos y entre escenas (s)
PARK = (1790, 640)  # margen derecho, fuera del contenido

# Algo sigue calculando: spinner, widget de estado o resultados atenuados (data-stale) de una corrida anterior.
BUSY = ("() => !document.querySelector('[data-testid=stSpinner], [data-testid=stStatusWidget], [data-testid=stSkeleton], "
        "[data-stale=true]')")

# Cursor visible y resaltado: la grabación headless no dibuja el puntero.
INIT_JS = r"""
(() => {
  const add = () => {
    if (document.getElementById('__cur')) return;
    const c = document.createElement('div'); c.id = '__cur';
    c.style.cssText = 'position:fixed;left:-50px;top:-50px;width:22px;height:22px;margin:-11px 0 0 -11px;border-radius:50%;' +
      'background:rgba(255,255,255,.85);border:2px solid #d4262f;box-shadow:0 0 10px rgba(0,0,0,.6);z-index:2147483647;' +
      'pointer-events:none;transition:transform .12s';
    document.body.appendChild(c);
    document.addEventListener('mousemove', e => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; }, true);
    document.addEventListener('mousedown', () => c.style.transform = 'scale(.7)', true);
    document.addEventListener('mouseup', () => c.style.transform = 'scale(1)', true);
  };
  if (document.body) add(); else document.addEventListener('DOMContentLoaded', add);
})();
"""

FIND_JS = r"""
([specs]) => {
  const vis = e => { const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0 && e.offsetParent !== null; };
  const rects = [];
  for (const [sel, texts] of specs) {
    let best = null, area = Infinity;
    for (const e of document.querySelectorAll(sel)) {
      if (!vis(e)) continue;
      const t = (e.innerText || '').toLowerCase();
      if (!texts.every(x => t.includes(x.toLowerCase()))) continue;
      const r = e.getBoundingClientRect();
      if (r.width * r.height < area) { area = r.width * r.height; best = r; }
    }
    if (best) rects.push(best);
  }
  if (!rects.length) return null;
  const x0 = Math.min(...rects.map(r => r.left)), y0 = Math.min(...rects.map(r => r.top));
  const x1 = Math.max(...rects.map(r => r.right)), y1 = Math.max(...rects.map(r => r.bottom));
  return {x: x0, y: y0, w: x1 - x0, h: y1 - y0};
}
"""

HL_JS = r"""
([r, secs]) => {
  const d = document.createElement('div');
  const pad = 7;
  d.style.cssText = `position:fixed;left:${r.x - pad}px;top:${r.y - pad}px;width:${r.w + 2 * pad}px;height:${r.h + 2 * pad}px;` +
    'border:3px solid #ff3b3b;border-radius:12px;box-shadow:0 0 0 4px rgba(255,59,59,.18),0 0 24px rgba(255,59,59,.35);' +
    'z-index:2147483646;pointer-events:none;opacity:0;transition:opacity .35s';
  document.body.appendChild(d);
  requestAnimationFrame(() => d.style.opacity = 1);
  setTimeout(() => { d.style.opacity = 0; setTimeout(() => d.remove(), 400); }, secs * 1000);
}
"""

# Qué resaltar: lista de (selector CSS, textos que debe contener); se toma el elemento visible más chico de cada uno y se une.
TARGETS = {
    "env": [(".dr-env", ["Extreme Altitude"])],
    "kpi_stuff": [(".dr-card", ["Average Stuff+"])],
    "inbox_1": [(".dr-card", ["Pitcher 00964", "Review"])],
    "player_reco": [(".dr-card", ["Current recommendation"])],
    "adapt_ff": [(".dr-card", ["Four-Seam", "Net effect"])],
    "adapt_sl": [(".dr-card", ["Slider", "Net effect"])],
    "sim_cartas": [(".dr-card", ["Stuff+ escenario", "Nivel del mar"]), (".dr-card", ["Stuff+ escenario", "Medium"]),
                   (".dr-card", ["Stuff+ escenario", "Harp Helú"])],
    "sim_harp": [(".dr-card", ["Stuff+ escenario", "Harp Helú"])],
    "sim_origen": [("[data-testid=stHorizontalBlock]", ["Origen", "Tipo de pitch", "Mano"])],
    "sim_sliders": [("[data-testid=stSlider]", ["Velocidad"]), ("[data-testid=stSlider]", ["IVB a nivel del mar"])],
    "sim_mapa": [("[data-testid=stPlotlyChart]", [])],
    "scout_mapa": [("[data-testid=stPlotlyChart]", ["Más efectivos"])],
    "scout_tabla": [(".dr-card", ["Más efectivos", "Diferencia"])],
    "game_contexto": [("[data-testid=stHorizontalBlock]", ["Bateador", "Strikes", "Entorno"])],
    "game_ff": [(".dr-card", ["Four-Seam", "Pitching+ máximo"])],
    "game_heat": [("[data-testid=stPlotlyChart]", ["Calidad esperada"])],
    "game_lectura": [(".dr-card", ["Lectura", "Costo del contacto (zona)"])],
}
# A dónde desplazar la página antes de un segmento (texto del elemento que queda arriba, con margen).
SCROLL = {"adaptacion": ("Harp Helú Adaptation", 110), "sim_mapa": ("Comparar con los Four-Seam", 380),
          "scout_mapa": ("Mapa de movimiento", 40)}


class Dash:
    def __init__(self, page):
        self.p = page

    def idle(self, extra=0.3):
        self.p.wait_for_timeout(350)
        for _ in range(2):
            self.p.wait_for_function(BUSY, timeout=300000)
            self.p.wait_for_timeout(250)
        self.p.wait_for_timeout(int(extra * 1000))

    def move_click(self, loc):
        loc.scroll_into_view_if_needed()
        b = loc.bounding_box()
        self.p.mouse.move(b["x"] + b["width"] / 2, b["y"] + b["height"] / 2, steps=28)
        self.p.wait_for_timeout(250)
        loc.click()
        self.idle()

    def click(self, name):
        p = self.p
        loc = {
            "review_1": lambda: p.locator("a.dr-btn", has_text="Review").first,
            "abrir_simulador": lambda: p.locator("a", has_text="Open in What-if Simulator").first,
            "restablecer": lambda: p.get_by_role("button", name="Restablecer al pitch actual"),
            "pitch_manual": lambda: p.get_by_text("Pitch manual", exact=True),
            "nav_scouting": lambda: p.get_by_role("link", name="Scouting"),
            "nav_roster": lambda: p.get_by_role("link", name="Roster"),
            "strikes_2": lambda: p.locator("[data-testid=stElementContainer]", has_text="Strikes").get_by_role("button", name="2", exact=True),
            "nivel_mar": lambda: p.locator("[data-testid=stElementContainer]", has_text="Entorno").get_by_role("button", name="Nivel del mar"),
        }[name]()
        self.move_click(loc)

    def scroll_main(self, y_target, secs=1.2):
        self.p.evaluate("""([y, ms]) => new Promise(res => {
            const m = document.querySelector('[data-testid=stMain]'); const y0 = m.scrollTop, t0 = performance.now();
            const f = t => { const k = Math.min(1, (t - t0) / ms), e = k < .5 ? 2*k*k : 1 - Math.pow(-2*k + 2, 2) / 2;
                             m.scrollTop = y0 + (y - y0) * e; k < 1 ? requestAnimationFrame(f) : res(); };
            requestAnimationFrame(f); })""", [y_target, secs * 1000])
        self.p.wait_for_timeout(300)

    def scroll(self, name):
        if name == "fondo":
            h = self.p.evaluate("document.querySelector('[data-testid=stMain]').scrollHeight")
            return self.scroll_main(h, 1.6)
        text, margin = SCROLL[name]
        y = self.p.evaluate("""([t, mg]) => { const m = document.querySelector('[data-testid=stMain]');
            const e = [...m.querySelectorAll('*')].find(x => x.childElementCount === 0 && (x.innerText||'').includes(t) && x.offsetParent);
            return e ? m.scrollTop + e.getBoundingClientRect().top - mg : null; }""", [text, margin])
        if y is None:
            raise RuntimeError(f"No encontré '{text}' para desplazar")
        self.scroll_main(y)

    def highlight(self, name, secs):
        r = self.p.evaluate(FIND_JS, [TARGETS[name]])
        if r is None:
            print(f"  aviso: no encontré el resaltado '{name}'")
            return
        self.p.evaluate(HL_JS, [r, secs])

    def slider(self, idx, presses):
        th = self.p.get_by_role("slider").nth(idx)
        b = th.bounding_box()
        self.p.mouse.move(b["x"] + b["width"] / 2, b["y"] + b["height"] / 2, steps=20)
        th.focus()
        key = "ArrowRight" if presses > 0 else "ArrowLeft"
        for _ in range(abs(presses)):
            self.p.keyboard.press(key)
            self.p.wait_for_timeout(70)
        self.p.wait_for_timeout(300)
        self.p.mouse.move(b["x"] + 700, b["y"], steps=8)  # quitar el foco visual del pulgar
        self.p.get_by_text("Escenario (what-if)", exact=False).first.click()
        self.idle()

    def action(self, a):
        kind = a[0]
        if kind == "clic":
            self.click(a[1])
        elif kind == "desplazar":
            self.scroll(a[1])
        elif kind == "slider":
            self.slider(a[1], a[2])
        elif kind == "ir":
            self.p.goto(BASE + a[1])
            self.idle(1.0)
        else:
            raise ValueError(a)


def duracion(f):
    o = subprocess.run([FFMPEG, "-i", str(f)], capture_output=True, text=True).stderr
    h, m, s = re.search(r"Duration: (\d+):(\d+):([\d.]+)", o).groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


def audio(cfg, propia=None):
    """Un archivo de audio por segmento. Con `propia` (carpeta con <id>.m4a/.wav/.mp3/...) usa la voz grabada por el
    equipo: se pasa a WAV 48 kHz mono y se recortan los silencios del inicio y del final; si no, voz sintética con `say`."""
    d = OUT / "audio"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    dur = {}
    for s in cfg["segmentos"]:
        f = d / f"{s['id']}.wav"
        if propia:
            src = sorted(x for x in Path(propia).glob(f"{s['id']}.*") if not x.name.startswith("."))
            if not src:
                sys.exit(f"Falta el audio del segmento {s['id']} en {propia} (por ejemplo {s['id']}.m4a)")
            trim = "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.15"
            subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(src[0]), "-ac", "1", "-ar", "48000", "-af",
                            f"{trim},areverse,{trim},areverse", str(f)], check=True)
        else:
            tmp = d / f"{s['id']}.aiff"
            subprocess.run(["say", "-v", cfg["voz"], "-r", str(cfg["ritmo"]), "-o", str(tmp), s["texto"]], check=True)
            subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(tmp), "-ac", "1", "-ar", "48000", str(f)], check=True)
            tmp.unlink()
        dur[s["id"]] = duracion(f)
    return dur


def run(cfg, dur, record, shots=None):
    marks = []
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        kw = dict(viewport={"width": W, "height": H}, device_scale_factor=1)
        if record:
            vdir = OUT / "crudo"
            shutil.rmtree(vdir, ignore_errors=True)
            kw.update(record_video_dir=str(vdir), record_video_size={"width": W, "height": H})
        ctx = b.new_context(**kw)
        ctx.add_init_script(INIT_JS)
        p = ctx.new_page()
        t0 = time.monotonic()
        d = Dash(p)
        p.goto(BASE + "/")
        d.idle(1.5)
        p.mouse.move(W * 0.62, H * 0.55)
        escena = None
        for s in cfg["segmentos"]:
            if escena is not None:
                p.wait_for_timeout(int((PAUSA_ESCENA if s["escena"] != escena else PAUSA_SEG) * 1000))
            escena = s["escena"]
            for a in s["antes"]:
                d.action(a)
            if s["antes"]:
                p.mouse.move(*PARK, steps=15)  # el puntero en el margen, sin hover sobre controles
            start = time.monotonic() - t0
            marks.append((s["id"], start))
            print(f"{s['id']:>3}  {start:7.2f}s  {dur[s['id']]:5.1f}s", flush=True)
            if shots is not None:
                p.wait_for_timeout(200)
            end = time.monotonic() + dur[s["id"]]
            # Los tiempos de 'durante' se escalan para no pasar la duración del audio de este segmento.
            plan = sum(a[1] if a[0] == "esperar" else a[2] for a in s["durante"])
            k = min(1.0, dur[s["id"]] / plan) if plan and dur[s["id"]] else 1.0
            for a in s["durante"]:
                if a[0] == "esperar":
                    p.wait_for_timeout(int(a[1] * k * 1000))
                else:
                    d.highlight(a[1], max(a[2] * k, 1.5))
                    if shots is not None:
                        p.wait_for_timeout(450)
                        p.screenshot(path=str(shots / f"{s['id']}_{a[1]}.png"))
                    p.wait_for_timeout(int(a[2] * k * 1000))
            if shots is not None and not any(a[0] == "resaltar" for a in s["durante"]):
                p.screenshot(path=str(shots / f"{s['id']}.png"))
            rest = end - time.monotonic()
            if rest > 0 and shots is None:
                p.wait_for_timeout(int(rest * 1000))
        p.wait_for_timeout(2000)
        total = time.monotonic() - t0
        video = p.video.path() if record else None
        ctx.close()
        b.close()
    return marks, total, video


def _tc(t, sep):
    h, m = int(t // 3600), int(t % 3600 // 60)
    return f"{h:02d}:{m:02d}:{t % 60:06.3f}".replace(".", sep) if sep == "," else f"{h}:{m:02d}:{t % 60:05.2f}"


def _trozos(texto, maximo=84):
    """Parte el texto en subtítulos de hasta `maximo` caracteres, cortando en fin de oración y luego en : ; ,"""
    out = []
    for frase in re.split(r"(?<=[.:;])\s+(?=[A-ZÁÉÍÓÚÑ¿0-9])", texto):
        while len(frase) > maximo:
            cortes = [m.end() for m in re.finditer(r"[,:;]\s", frase) if 20 <= m.end() <= maximo]
            if not cortes:
                cortes = [m.start() for m in re.finditer(r"\s", frase) if m.start() <= maximo] or [maximo]
            c = max(cortes)
            out.append(frase[:c].strip())
            frase = frase[c:].strip()
        if frase:
            out.append(frase)
    return out


def subtitulos(cfg, marks, start):
    """Subtítulos por segmento. Dentro de cada segmento, cada trozo dura en proporción a sus caracteres y el corte se
    ajusta a la pausa más cercana del audio. Escribe subtitulos.srt (para subir aparte) y subtitulos.ass (se quema)."""
    seg = {s["id"]: s for s in cfg["segmentos"]}
    cues = []
    for sid, t in marks:
        f = OUT / "audio" / f"{sid}.wav"
        d = duracion(f)
        o = subprocess.run([FFMPEG, "-i", str(f), "-af", "silencedetect=noise=-38dB:d=0.18", "-f", "null", "-"],
                           capture_output=True, text=True).stderr
        pausas = [(float(a) + float(b)) / 2 for a, b in zip(re.findall(r"silence_start: ([\d.]+)", o),
                                                         re.findall(r"silence_end: ([\d.]+)", o))]
        tr = _trozos(seg[sid].get("subtitulo", seg[sid]["texto"]))
        n = sum(len(x) for x in tr)
        acc, prev = 0, 0.0
        for i, x in enumerate(tr):
            acc += len(x)
            fin = d if i == len(tr) - 1 else d * acc / n
            if i < len(tr) - 1 and pausas:
                p = min(pausas, key=lambda q: abs(q - fin))
                if abs(p - fin) < 1.5 and p > prev + 0.6:
                    fin = p
            cues.append((t - start + prev, t - start + fin + (0.25 if i == len(tr) - 1 else 0), x))
            prev = fin
    srt = "\n".join(f"{i + 1}\n{_tc(a, ',')} --> {_tc(b, ',')}\n{x}\n" for i, (a, b, x) in enumerate(cues))
    (OUT / "subtitulos.srt").write_text(srt, encoding="utf-8")
    ass = ["[Script Info]", "ScriptType: v4.00+", f"PlayResX: {W}", f"PlayResY: {H}", "WrapStyle: 0", "",
           "[V4+ Styles]",
           "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
           "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
           "MarginR, MarginV, Encoding",
           "Style: Sub,Helvetica Neue,40,&H00FFFFFF,&H00FFFFFF,&H40101114,&H00000000,0,0,0,0,100,100,0,0,3,12,0,2,"
           "360,360,34,1", "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    ass += [f"Dialogue: 0,{_tc(a, '.')},{_tc(b, '.')},Sub,,0,0,0,,{x}" for a, b, x in cues]
    (OUT / "subtitulos.ass").write_text("\n".join(ass) + "\n", encoding="utf-8")
    return OUT / "subtitulos.ass"


GAP_MAX, GAP_ANTES, GAP_DESPUES = 3.2, 1.8, 1.2  # huecos sin voz más largos se cortan: se ve el inicio de la acción y el resultado


def cortes(marks, dur, total):
    """Tramos del video crudo que se conservan. Un hueco sin voz de más de GAP_MAX segundos (el dashboard recalculando)
    se recorta: quedan sus primeros GAP_ANTES s (el clic o el slider) y sus últimos GAP_DESPUES s (el resultado)."""
    keep, a = [], max(0.0, marks[0][1] - 0.8)
    for (sid, t), nxt in zip(marks, marks[1:] + [None]):
        fin = t + dur[sid]
        sig = nxt[1] if nxt else total
        if nxt and sig - fin > GAP_MAX:
            keep.append((a, fin + GAP_ANTES))
            a = sig - GAP_DESPUES
    keep.append((a, total))
    return keep


def mapa(keep):
    def f(t):
        acc = 0.0
        for a, b in keep:
            if t <= b:
                return acc + max(0.0, t - a)
            acc += b - a
        return acc
    return f


def mux(cfg, marks, total, raw):
    dur = {s["id"]: duracion(OUT / "audio" / f"{s['id']}.wav") for s in cfg["segmentos"]}
    keep = cortes(marks, dur, total)
    tmap = mapa(keep)
    nmarks = [(sid, tmap(t)) for sid, t in marks]
    length = sum(b - a for a, b in keep)
    filt = [f"[0:v]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS[v{i}]" for i, (a, b) in enumerate(keep)]
    filt.append("".join(f"[v{i}]" for i in range(len(keep))) + f"concat=n={len(keep)}:v=1:a=0[vc]")
    inputs = []
    for i, (sid, t) in enumerate(nmarks):
        inputs += ["-i", str(OUT / "audio" / f"{sid}.wav")]
        ms = int(t * 1000)
        filt.append(f"[{i + 1}:a]aresample=48000,adelay={ms}|{ms}[a{i}]")
    filt.append("".join(f"[a{i}]" for i in range(len(nmarks))) + f"amix=inputs={len(nmarks)}:normalize=0,volume=1.4,alimiter=limit=0.7:level=false[aout]")
    ass = subtitulos(cfg, nmarks, 0.0)
    filt.append(f"[vc]ass={ass.name}[vout]")
    out = OUT / "video_entrega.mp4"
    cmd = [FFMPEG, "-y", "-i", str(raw), *inputs, "-filter_complex", ";".join(filt),
           "-map", "[vout]", "-map", "[aout]", "-t", f"{length:.2f}",
           "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-r", "30",
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True, capture_output=True, cwd=OUT)  # cwd: el filtro ass toma la ruta relativa
    print("Huecos recortados:", ", ".join(f"{b0:.0f}-{a1:.0f}s" for (_, b0), (a1, _) in zip(keep, keep[1:])) or "ninguno")
    return out, length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensayo", action="store_true")
    ap.add_argument("--voz-propia", metavar="CARPETA", help="carpeta con un audio por segmento (<id>.m4a, .wav, .mp3…)")
    ap.add_argument("--remontar", action="store_true", help="rehace solo el montaje de audio sobre la última grabación")
    args = ap.parse_args()
    cfg = json.loads((ROOT / "narracion.json").read_text(encoding="utf-8"))
    OUT.mkdir(exist_ok=True)
    dur = audio(cfg, args.voz_propia) if not args.remontar else None
    if dur:
        print(f"Narración: {sum(dur.values()):.1f} s en {len(dur)} segmentos")
    if args.remontar:
        m = json.loads((OUT / "marcas.json").read_text(encoding="utf-8"))
        out, length = mux(cfg, [tuple(x) for x in m["marcas"]], m["total"], Path(m["video"]))
        print(f"Listo: {out} ({int(length // 60)}:{length % 60:04.1f})")
        return
    if args.ensayo:
        shots = OUT / "ensayo"
        shutil.rmtree(shots, ignore_errors=True)
        shots.mkdir()
        run(cfg, dur, record=False, shots=shots)
        return
    print("Calentando cachés del dashboard (recorrido sin grabar)…")
    run(cfg, {k: 0.0 for k in dur}, record=False)
    print("Grabando…")
    marks, total, raw = run(cfg, dur, record=True)
    (OUT / "marcas.json").write_text(json.dumps({"marcas": marks, "total": total, "video": str(raw)}, indent=1), encoding="utf-8")
    out, length = mux(cfg, marks, total, raw)
    print(f"Listo: {out} ({int(length // 60)}:{length % 60:04.1f})")


if __name__ == "__main__":
    sys.exit(main())
