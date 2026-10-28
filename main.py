import argparse
import json
import os
import queue
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from rich.console import Console, Group
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

load_dotenv()

from audio import LoopbackCapture, VADConfig  # noqa: E402

try:
    import msvcrt
except ImportError:  # no es Windows
    msvcrt = None

console = Console()
KEYS_ENABLED = bool(msvcrt and sys.stdin.isatty())
SILENCE_RMS = 0.002
LLM_QUEUE_SIZE = 1
LOG_DIR = Path(__file__).parent / "logs"
# Vocabulario de dominio para Whisper (nombres propios, siglas, jerga técnica). Vacío por defecto;
# personalízalo con WHISPER_INITIAL_PROMPT en .env (ver .env.example para un ejemplo).
DEFAULT_INITIAL_PROMPT = ""
ANSWERED, NOT_Q_LLM, LLM_ERROR, CANCELLED = "pregunta_respondida", "no_pregunta_llm", "error_llm", "cancelada"
FILTERED, QUEUE_DROP, NO_TEXT, PAUSED = "filtrada_local", "descartada_cola", "sin_texto", "descartada_pausa"
SENT_KINDS = {ANSWERED, NOT_Q_LLM, LLM_ERROR, CANCELLED}


class SessionLog:
    """Registro JSONL de la sesión (sin audio; solo texto y tiempos)."""

    def __init__(self):
        LOG_DIR.mkdir(exist_ok=True)
        self.path = LOG_DIR / f"sesion-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
        self._f = open(self.path, "a", encoding="utf-8")
        self._lock = threading.Lock()
        self.records = []

    def write(self, rec):
        rec = {"ts": datetime.now().isoformat(timespec="milliseconds"), **rec}
        with self._lock:
            self._f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._f.flush()
            if "tipo" not in rec:
                self.records.append(rec)

    def close(self):
        with self._lock:
            self._f.close()


def r2(x):
    return None if x is None else round(x, 3)


def read_key():
    """Una tecla pendiente de la consola (sin esperar Enter), o None."""
    if msvcrt and msvcrt.kbhit():
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):  # tecla especial: descartar su segundo código
            msvcrt.getwch()
            return None
        return ch
    return None


def bullets_of(text):
    return [re.sub(r"^\s*[-*•]\s+", "", ln).strip() for ln in text.splitlines() if ln.strip()]


def md_text(s, style=""):
    """Convierte **negrita** de Markdown en estilo Rich (sin dejar asteriscos sueltos)."""
    t = Text(style=style)
    for i, part in enumerate(s.split("**")):
        if part:
            t.append(part, style="bold" if i % 2 else "")
    return t


class DebugView:
    """Salida técnica detallada (--debug) y modo solo transcripción."""

    def __init__(self, is_speaking=None):
        pass

    def start(self):
        console.print("[dim]Ctrl+C para detener[/]")

    def stop(self):
        pass

    def set_stt(self, on):
        pass

    def set_llm(self, on):
        pass

    def set_paused(self, on):
        console.print("[yellow]PAUSADO[/] (P para reanudar)" if on else "[green]REANUDADO[/]")

    def utterance(self, info):
        console.print(
            f"[dim]#{info['n']} dur={info['dur']:.1f}s lang={info['lang']}({info['lang_p']:.2f}) "
            f"whisper={info['whisper']:.2f}s cola_audio={info['qsize']} final={info['final']}[/]"
            + (f" [yellow]descartados={info['dropped']}[/]" if info["dropped"] else "")
        )

    def partial(self):
        console.print("[dim](intervención larga: se espera su final antes de consultar al LLM)[/]")

    def question(self, i, text, opener=None):
        console.print(text or "(sin voz detectada)", style="bold cyan", markup=False, highlight=False)
        if opener:
            console.print(f"[dim]inicio sugerido: {opener}[/]")

    def skipped(self, i, text, reason):
        if text and reason != NOT_Q_LLM:
            console.print(text, style="bold cyan", markup=False, highlight=False)
        console.print(f"[dim]#{i} sin consulta al LLM: {reason}[/]")

    def answer(self, i, a, stages, total):
        show_answer(a, stages)
        console.print(f"[dim]fin real de la voz -> respuesta mostrada: {total:.2f}s[/]")

    def note(self, text):
        console.print(f"[red]{text}[/]")


class InterviewView:
    """Panel de lectura rápida con Rich Live: estado, última pregunta y las tres viñetas."""

    def __init__(self, is_speaking):
        self.is_speaking = is_speaking
        self.lock = threading.Lock()
        self.stt = self.llm = self.paused = False
        self.question_text = ""
        self.opener = None
        self.bullets = []
        self.stale = False
        self.latency = None
        self.note_text = ""
        self.note_until = 0.0
        self.live = None

    def start(self):
        self.live = Live(self, console=console, refresh_per_second=4, transient=False)
        self.live.start()

    def stop(self):
        if self.live:
            self.live.stop()

    def set_stt(self, on):
        self.stt = on

    def set_llm(self, on):
        self.llm = on

    def set_paused(self, on):
        self.paused = on

    def utterance(self, info):
        pass

    def partial(self):
        pass

    def question(self, i, text, opener=None):
        with self.lock:
            self.question_text = text
            self.stale = bool(self.bullets)
            self.opener = opener

    def skipped(self, i, text, reason):
        with self.lock:
            self.opener = None
            self.stale = False  # no habrá respuesta nueva: la anterior vuelve a ser la vigente, no "obsoleta"

    def answer(self, i, a, stages, total):
        with self.lock:
            self.bullets = bullets_of(a.text)
            self.stale = False
            self.opener = None
            self.latency = total

    def note(self, text):
        with self.lock:
            self.note_text, self.note_until = text, time.monotonic() + 10

    def _state(self):
        if self.paused:
            return "PAUSADO", "yellow"
        if self.stt:
            return "TRANSCRIBIENDO", "yellow"
        if self.llm:
            return "GENERANDO", "yellow"
        if self.is_speaking():
            return "ESCUCHANDO", "cyan"
        if self.bullets and not self.stale:
            return "RESPUESTA LISTA", "green"
        return "ESCUCHANDO", "cyan"

    def __rich__(self):
        width = max(40, min(console.width - 6, 96))
        with self.lock:
            q, bullets, stale, latency = self.question_text, list(self.bullets), self.stale, self.latency
            opener, note = self.opener, (self.note_text if time.monotonic() < self.note_until else "")
        state, color = self._state()
        status = Text()
        status.append("● ", style=color)
        status.append(state, style=f"bold {color}")
        if KEYS_ENABLED:
            status.append("   P reanuda · Ctrl+C sale" if self.paused else "   P pausa · Ctrl+C sale", style="dim")
        if note:
            status.append(f"   {note}", style="dim red")
        q_short = (q[:220] + "…") if len(q) > 220 else q
        q_panel = Panel(Text(q_short or "—", style="" if q else "dim"), title="PREGUNTA", title_align="left",
                        border_style="dim", width=width, padding=(0, 2))
        if state == "GENERANDO" and opener:
            # Frase local para empezar a hablar; se reemplaza por completo al llegar la respuesta real.
            a_panel = Panel(Text(opener, style="italic cyan"), title="INICIO SUGERIDO", title_align="left",
                            border_style="cyan", width=width, padding=(1, 2))
        else:
            grid = Table.grid(padding=(0, 1))
            grid.add_column(width=1)
            grid.add_column(ratio=1)
            for k, b in enumerate(bullets):
                if k:
                    grid.add_row("", "")
                grid.add_row("•", md_text(b, style="dim" if stale else "bold"))
            if not bullets:
                grid.add_row("", Text("—", style="dim"))
            title = "RESPUESTA ANTERIOR" if stale else "RESPUESTA"
            subtitle = f"{latency:.1f} s" if (latency and not stale) else None
            a_panel = Panel(grid, title=title, title_align="left", subtitle=subtitle, subtitle_align="right",
                            border_style="dim" if (stale or not bullets) else "green", width=width, padding=(1, 2))
        return Padding(Group(status, Text(""), q_panel, Text(""), a_panel), (1, 2))


def show_answer(a, stages=None):
    if a.error:
        console.print(f"[red]LLM:[/] {a.error} [dim]({a.latency:.2f}s)[/]")
        return
    tok = ""
    if a.input_tokens is not None:
        cached = f" (caché {a.cached_tokens})" if a.cached_tokens is not None else ""
        tok = f" tokens in={a.input_tokens}{cached} out={a.output_tokens}"
    parts = [f"{k}={v:.2f}s" for k, v in {**(stages or {}), **a.timings}.items()]
    console.print(f"[dim]LLM {a.latency:.2f}s{tok}[/]")
    if parts:
        console.print("[dim]etapas: " + " ".join(parts) + "[/]")
    for w in a.warnings:
        console.print(f"[yellow]Aviso de Codex:[/] {w}", markup=True, highlight=False)
    if a.is_question:
        console.print(a.text, style="bold green", markup=False, highlight=False)
    else:
        console.print("[dim](no es una pregunta: sin respuesta)[/]")


def llm_worker(assistant, q, stop, view, log):
    while not stop.is_set():
        try:
            item = q.get(timeout=0.5)
        except queue.Empty:
            continue
        view.set_llm(True)
        t_start = time.perf_counter()
        a = assistant.answer(item["text"])
        t_done = time.perf_counter()
        total = t_done - item["t_voice_end"]
        rec = {
            **item["rec"],
            "cola_llm_s": r2(t_start - item["t_enqueued"]),
            "codex_inicio_s": r2(a.timings.get("codex_inicio")),
            "codex_ejecucion_s": r2(a.timings.get("codex_ejecucion")),
            "codex_total_s": r2(a.timings.get("codex_total")),
            "llm_s": r2(a.latency),
            "latencia_total_s": r2(total),
            "tokens_entrada": a.input_tokens, "tokens_cache": a.cached_tokens, "tokens_salida": a.output_tokens,
            "avisos_codex": a.warnings or None,
        }
        stages = {
            "espera_silencio": item["rec"]["vad_espera_silencio_s"], "cola_audio": item["rec"]["cola_audio_s"],
            "whisper": item["rec"]["whisper_s"], "cola_llm": t_start - item["t_enqueued"],
        }
        if a.error:
            cancelled = a.error == "Cancelado." or stop.is_set()
            rec.update(kind=CANCELLED if cancelled else LLM_ERROR, error=a.error, cancelado=cancelled)
            if not cancelled:
                view.note(f"Error LLM: {a.error}")
        elif not a.is_question:
            rec["kind"] = NOT_Q_LLM
            view.skipped(rec["id"], item["text"], NOT_Q_LLM)
        else:
            rec.update(kind=ANSWERED, respuesta=a.text)
            view.answer(rec["id"], a, stages, total)
        if rec["kind"] == LLM_ERROR:
            view.skipped(rec["id"], item["text"], f"error: {a.error}")
        log.write(rec)
        view.set_llm(False)
        if stop.is_set():
            return


def put_latest(q, item):
    """Cola acotada: si está llena, descarta la entrada más antigua y la devuelve."""
    dropped = []
    while True:
        try:
            q.put_nowait(item)
            return dropped
        except queue.Full:
            try:
                dropped.append(q.get_nowait())
            except queue.Empty:
                pass


def print_summary(log, started):
    recs = log.records
    sent = [r for r in recs if r["kind"] in SENT_KINDS]
    answered = [r for r in recs if r["kind"] == ANSWERED]
    lats = [r["latencia_total_s"] for r in answered if r.get("latencia_total_s") is not None]
    count = lambda k: sum(1 for r in recs if r["kind"] == k)  # noqa: E731
    summary = {
        "intervenciones": len({r["id"] for r in recs if r.get("id") is not None}), "preguntas_procesadas": len(sent),
        "errores_transcripcion": count("error_transcripcion"),
        "respuestas_generadas": len(answered),
        "latencia_media_s": r2(sum(lats) / len(lats)) if lats else None,
        "latencia_maxima_s": r2(max(lats)) if lats else None,
        "descartadas_cola": count(QUEUE_DROP), "canceladas": count(CANCELLED), "descartadas_pausa": count(PAUSED),
        "filtradas_localmente": count(FILTERED), "no_pregunta_llm": count(NOT_Q_LLM), "errores_llm": count(LLM_ERROR),
        "duracion_s": r2(time.monotonic() - started),
    }
    log.write({"tipo": "fin", **summary})
    avg = f"{summary['latencia_media_s']:.1f} s" if lats else "—"
    mx = f"{summary['latencia_maxima_s']:.1f} s" if lats else "—"
    console.print()
    console.print("[bold]Resumen de la sesión[/]")
    console.print(f"  Intervenciones detectadas: {summary['intervenciones']}")
    console.print(f"  Preguntas procesadas: {summary['preguntas_procesadas']}   Respuestas generadas: "
                  f"{summary['respuestas_generadas']}")
    console.print(f"  Latencia (fin de voz → respuesta): media {avg}, máxima {mx}")
    console.print(f"  Consultas descartadas: {summary['descartadas_cola'] + summary['canceladas']} "
                  f"(cola: {summary['descartadas_cola']}, canceladas: {summary['canceladas']})   "
                  f"Intervenciones ignoradas por pausa: {summary['descartadas_pausa']}")
    console.print(f"  Filtradas localmente: {summary['filtradas_localmente']}   "
                  f"NO_PREGUNTA del LLM: {summary['no_pregunta_llm']}   Errores LLM: {summary['errores_llm']}   "
                  f"Errores de transcripción: {summary['errores_transcripcion']}")
    console.print(f"  Registro completo: {log.path}", soft_wrap=True, markup=False)


CANCEL = object()


def fmt_dt(dt):
    return dt.astimezone().strftime("%Y-%m-%d %H:%M") if dt else "—"


def choose_option(model_options, availability):
    """Menú inicial de modelo (solo en terminal interactiva). Devuelve la opción o CANCEL."""
    print("\nCONTEXTFLOW AI — SELECCIÓN DE MODELO\n")
    for n, o in model_options.items():
        ok, note = availability[n]
        tag = ""
        if ok is False:
            tag = f"  (no disponible: {note})"
        elif ok is None:
            tag = f"  (no verificado: {note})"
        elif note:
            tag = f"  ({note})"
        print(f"[{n}] {o['label']} — {o['mode']}{tag}")
    print()
    while True:
        try:
            raw = input("Selecciona una opción [1]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nCancelado.")
            return CANCEL
        if raw not in ("", "1", "2", "3"):
            print("Opción no válida. Escribe 1, 2 o 3 y pulsa Enter.")
            continue
        n = int(raw or "1")
        if availability[n][0] is False:
            print(f"La opción {n} no está disponible: {availability[n][1]}")
            continue
        return n


def print_context_block(assistant, option, status):
    """Resumen previo a escuchar: modelo, modo y estado verificable del contexto."""
    from assistant import MODEL_OPTIONS
    if option:
        o = MODEL_OPTIONS[option]
        model_line, mode = f"{o['label']} ({o['model']})", o["mode"]
    else:
        model_line, mode = getattr(assistant, "model", "?"), getattr(assistant, "mode", "Estándar")
    if status.get("fuente_editada"):
        updated = f"fuente {fmt_dt(status['fuente_editada'])} · copia local {fmt_dt(status.get('copiado'))}"
    else:
        updated = f"archivo local {fmt_dt(status['archivo_modificado'])} (sin fecha de sincronización)"
    color = {"Actualizado": "green", "Desactualizado": "red"}.get(status["estado"], "yellow")
    console.print()
    console.print(f"Modelo: {model_line}", markup=False)
    console.print(f"Modo: {mode}", markup=False)
    console.print(f"Contexto: {status['titulo']}", markup=False)
    console.print(f"Fuente: {status['fuente']}", markup=False)
    console.print(f"Última actualización: {updated}", markup=False)
    console.print(f"Estado: [{color}]{status['estado']}[/] [dim]— {status['detalle']}[/]")
    console.print(f"[dim]Contexto enviado a cada consulta: {assistant.context_chars} caracteres, "
                  f"huella {assistant.context_sha}[/]")
    console.print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="base")
    ap.add_argument("--check", action="store_true", help="solo carga el modelo y abre el dispositivo")
    ap.add_argument("--assistant", action="store_true", help="envía cada intervención al LLM")
    ap.add_argument("--ask", metavar="TEXTO", help="prueba el LLM con un texto, sin audio")
    ap.add_argument("--debug", action="store_true", help="salida técnica detallada en lugar del panel")
    ap.add_argument("--opcion", type=int, choices=(1, 2, 3), help="elige el modelo sin menú (1, 2 o 3)")
    args = ap.parse_args()

    assistant = None
    ctx_status = None
    option = args.opcion
    if args.assistant or args.ask:
        from assistant import (MODEL_OPTIONS, ConfigError, check_model_options, context_status,
                               create_provider)
        interactive = sys.stdin.isatty() and sys.stdout.isatty()
        if option is None and args.assistant and not args.ask and interactive:
            option = choose_option(MODEL_OPTIONS, check_model_options())
            if option is CANCEL:
                return 130
        elif option is not None:
            ok, note = check_model_options()[option]
            if ok is False:
                console.print(f"[red]La opción {option} no está disponible:[/] {note}")
                return 1
        try:
            assistant = create_provider(option)
            ctx_status = context_status()
        except ConfigError as e:
            console.print(f"[red]Configuración:[/] {e}")
            return 1
        if args.debug or args.ask:
            console.print(f"Asistente: {assistant.describe()}")
        if option is not None and not args.ask and interactive:
            o = MODEL_OPTIONS[option]
            console.print(f"\nModelo seleccionado: [bold]{o['label']} — {o['mode']}[/]")
        if not args.ask:
            print_context_block(assistant, option, ctx_status)
    if args.ask:
        try:
            show_answer(assistant.answer(args.ask))
        except KeyboardInterrupt:
            assistant.cancel()
            console.print("[yellow]Cancelado.[/]")
        return 0

    filter_on = os.getenv("QUESTION_FILTER", "on").strip().lower() not in ("off", "0", "false", "no")
    from assistant import is_probable_question, suggest_opener
    try:
        merge_s = float(os.getenv("FILTER_MERGE_SECONDS", "3"))
    except ValueError:
        console.print("[red]Configuración:[/] FILTER_MERGE_SECONDS debe ser numérico.")
        return 1

    interview = bool(args.assistant and not args.debug)
    if not interview:
        console.print(f"[bold]Cargando modelo '{args.model}' (cpu, int8)...[/]")
    else:
        console.print("[dim]Cargando modelo...[/]")
    t0 = time.perf_counter()
    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    initial_prompt = os.getenv("WHISPER_INITIAL_PROMPT", DEFAULT_INITIAL_PROMPT)
    lang_cfg = os.getenv("WHISPER_LANGUAGE", "es").strip().lower()
    language = None if lang_cfg in ("", "auto") else lang_cfg
    if not interview:
        console.print(f"Modelo listo en {time.perf_counter() - t0:.1f}s")

    try:
        cfg = VADConfig.from_env()
    except ValueError as e:
        console.print(f"[red]Configuración:[/] {e}")
        return 1
    cap = LoopbackCapture(config=cfg)
    try:
        cap.start()
    except Exception as e:
        console.print(f"[red]Error abriendo loopback:[/] {e}")
        return 1
    if not interview:
        console.print(f"Capturando: [{cap.device['index']}] {cap.device['name']} ({cap.channels} ch, {cap.rate} Hz)")
        console.print(f"[dim]VAD: silencio={cfg.silence_ms}ms preroll={cfg.preroll_ms}ms máx={cfg.max_seconds:.0f}s "
                      f"umbral={cfg.threshold} | filtro de preguntas: {'on' if filter_on else 'off'} | "
                      f"idioma Whisper: {language or 'auto'}[/]")

    if args.check:
        time.sleep(1)
        cap.stop()
        console.print("[green]Comprobación OK[/] (dispositivo abierto y cerrado, modelo cargado)")
        return 0

    log = SessionLog()
    log.write({"tipo": "inicio", "proveedor": assistant.describe() if assistant else None, "whisper": args.model,
               "vad": vars(cfg), "filtro_preguntas": filter_on, "modo": "entrevista" if interview else "debug",
               "idioma_whisper": language or "auto", "initial_prompt": initial_prompt,
               "asistente": assistant.meta() if assistant else None,
               "contexto": ({"estado": ctx_status["estado"], "detalle": ctx_status["detalle"],
                             "archivo_sha12": ctx_status["sha12"], "huella_enviada": assistant.context_sha,
                             "caracteres_enviados": assistant.context_chars,
                             "fuente_editada": ctx_status.get("fuente_editada") and ctx_status["fuente_editada"].isoformat(),
                             "copiado": ctx_status.get("copiado") and ctx_status["copiado"].isoformat()}
                            if assistant else None)})
    view = (InterviewView if interview else DebugView)(lambda: cap.speaking)

    llm_q = queue.Queue(maxsize=LLM_QUEUE_SIZE)
    llm_stop = threading.Event()
    llm_thread = None
    if assistant:
        llm_thread = threading.Thread(target=llm_worker, args=(assistant, llm_q, llm_stop, view, log), daemon=True)
        llm_thread.start()

    started = time.monotonic()
    view.start()
    n = 0
    pending, pending_stt, pending_dur = [], 0.0, 0.0
    lang_seen = None
    held = None
    paused = False
    try:
        while True:
            key = read_key() if KEYS_ENABLED else None
            while key:
                if key.lower() == "p":
                    paused = not paused
                    pending, pending_stt, pending_dur, held = [], 0.0, 0.0, None
                    view.set_paused(paused)
                    log.write({"tipo": "pausa" if paused else "reanudacion"})
                key = read_key()
            try:
                item = cap.segments.get(timeout=0.1)
            except queue.Empty:
                continue
            if paused:
                if item["final"] and item["duration"] > 0:
                    log.write({"kind": PAUSED, "id": None, "audio_s": r2(item["duration"])})
                continue
            audio_wait = time.perf_counter() - item["t_ready"]
            seg = item["audio"]
            text, dt = "", 0.0
            if len(seg) and float(np.sqrt(np.mean(seg ** 2))) >= SILENCE_RMS:
                view.set_stt(True)
                t = time.perf_counter()
                try:
                    segs, info = model.transcribe(
                        seg, beam_size=1, vad_filter=False, initial_prompt=initial_prompt,
                        condition_on_previous_text=False, language=language,
                    )
                    text = " ".join(s.text.strip() for s in segs).strip()
                except Exception as e:
                    view.set_stt(False)
                    view.note(f"Error de transcripción: {e}")
                    log.write({"kind": "error_transcripcion", "id": None, "error": str(e)[:300],
                               "audio_s": r2(item["duration"])})
                    text, info = "", None
                if info is not None:
                    dt = time.perf_counter() - t
                    view.set_stt(False)
                    lang_seen = (info.language, round(info.language_probability, 3))
                    view.utterance({"n": n + 1, "dur": item["duration"], "lang": info.language,
                                    "lang_p": info.language_probability, "whisper": dt,
                                    "qsize": cap.segments.qsize(), "final": item["final"], "dropped": cap.dropped})
            if text:
                pending.append(text)
            pending_stt += dt
            pending_dur += item["duration"]
            if not item["final"]:
                view.partial()
                continue
            n += 1
            full, stt, dur = " ".join(pending).strip(), pending_stt, pending_dur
            pending, pending_stt, pending_dur = [], 0.0, 0.0
            rec = {"id": n, "transcripcion": full, "audio_s": r2(dur),
                   "vad_espera_silencio_s": r2(item["t_ready"] - item["t_voice_end"]),
                   "cola_audio_s": r2(audio_wait), "whisper_s": r2(stt),
                   "idioma": lang_seen[0] if lang_seen else None, "idioma_prob": lang_seen[1] if lang_seen else None}
            lang_seen = None
            if not full:
                if dur > 0:
                    log.write({**rec, "kind": NO_TEXT})
                continue
            if filter_on and held and merge_s > 0:
                # Una pregunta partida por una pausa larga ("Supón que… 12.000." + "¿Qué harías?") se vuelve a unir.
                if item["t_voice_end"] - item["duration"] - held["t_voice_end"] <= merge_s:
                    full = f"{held['text']} {full}"
                    rec["transcripcion"], rec["fusionada_con"] = full, held["id"]
            held = None
            is_question_confirmed = None
            if filter_on:
                ok, why = is_probable_question(full)
                rec["clasificacion"] = "pregunta" if ok else "afirmacion"
                rec["motivo_filtro"] = why
                is_question_confirmed = ok
                if not ok:
                    log.write({**rec, "kind": FILTERED})
                    view.skipped(n, full, f"filtro local ({why})")
                    held = {"text": full, "t_voice_end": item["t_voice_end"], "id": n}
                    continue
            if not assistant:
                view.question(n, full)
                log.write({**rec, "kind": "transcrita"})
                continue
            if is_question_confirmed is None:  # filtro apagado: igual confirmamos antes de sugerir inicio
                is_question_confirmed = is_probable_question(full)[0]
            opener = suggest_opener(full) if is_question_confirmed else None
            view.question(n, full, opener)
            for old in put_latest(llm_q, {"text": full, "rec": rec, "t_voice_end": item["t_voice_end"],
                                          "t_enqueued": time.perf_counter()}):
                log.write({**old["rec"], "kind": QUEUE_DROP})
                view.note("Consulta anterior descartada")
    except KeyboardInterrupt:
        pass
    finally:
        llm_stop.set()
        if assistant:
            assistant.cancel()
        cap.stop()
        if llm_thread:
            llm_thread.join(timeout=3)
        while True:
            try:
                old = llm_q.get_nowait()
            except queue.Empty:
                break
            log.write({**old["rec"], "kind": CANCELLED, "cancelado": True})
        view.stop()
        console.print("\n[yellow]Deteniendo...[/] Cerrado limpiamente.")
        print_summary(log, started)
        log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
