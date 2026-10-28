"""Compara las opciones de modelo con las mismas preguntas y el mismo contexto (sin audio).

Uso: python benchmark.py [--rondas 2] [--opciones 1 2 3]
Las llamadas se intercalan por pregunta entre opciones; la primera llamada de cada opción se marca como fría.
Guarda el detalle en logs/benchmark-<fecha>.jsonl.

Las preguntas son genéricas y ficticias (ver context.example.md); no corresponden a ninguna entrevista real.
"""
import argparse
import json
import re
import statistics
from datetime import datetime
from pathlib import Path

from assistant import MODEL_OPTIONS, check_model_options, context_status, create_provider

LOG_DIR = Path(__file__).parent / "logs"

QUESTIONS = [
    ("presentacion", "Para partir, cuéntame un poco sobre ti y tu trayectoria profesional."),
    ("motivacion", "¿Por qué te interesa este tipo de rol?"),
    ("tecnica_metricas", "¿Qué indicadores usarías para medir la productividad y el cumplimiento del plan en un centro de distribución?"),
    ("resultados", "¿Qué resultados obtuviste en tu último proyecto?"),
    ("caso", "Proyectas 15.000 unidades de recepción, pero el centro solo puede procesar 12.000. ¿Qué haces?"),
    ("fortaleza", "¿Cuál dirías que es tu mayor fortaleza?"),
]
UNSOLICITED = {
    "presentacion": r"me interesa|me motiva|motivad|aportar|aporte|contrat|quiero (trabajar|aprender)",
    "fortaleza": r"me interesa|aportar|aporte|contrat",
}


def flags(kind, text):
    out = []
    bullets = [ln for ln in text.splitlines() if ln.strip()]
    if len(bullets) > 3:
        out.append(f"{len(bullets)} viñetas")
    if kind in UNSOLICITED and re.search(UNSOLICITED[kind], text, re.I):
        out.append("info no solicitada")
    if kind in ("presentacion", "fortaleza", "motivacion") and re.search(r"\d", text):
        out.append("cifras sin pedirlas")
    if re.search(r"(?i)excel|sql|python|power ?bi", text) and kind in ("presentacion", "fortaleza", "motivacion"):
        out.append("herramientas sin pedirlas")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rondas", type=int, default=2)
    ap.add_argument("--opciones", type=int, nargs="+", default=[1, 2, 3], choices=(1, 2, 3))
    args = ap.parse_args()

    avail = check_model_options()
    options = []
    for n in args.opciones:
        ok, note = avail[n]
        if ok is False:
            print(f"Opción {n} omitida (no disponible): {note}")
        else:
            options.append(n)
    providers = {n: create_provider(n) for n in options}
    any_p = next(iter(providers.values()))
    ctx = context_status()
    LOG_DIR.mkdir(exist_ok=True)
    path = LOG_DIR / f"benchmark-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    f = open(path, "w", encoding="utf-8")

    def write(rec):
        f.write(json.dumps({"ts": datetime.now().isoformat(timespec="milliseconds"), **rec}, ensure_ascii=False) + "\n")
        f.flush()

    write({"tipo": "inicio", "opciones": {n: providers[n].meta() for n in options},
           "contexto": {"estado": ctx["estado"], "huella_enviada": any_p.context_sha,
                        "caracteres": any_p.context_chars}, "rondas": args.rondas,
           "preguntas": [q[0] for q in QUESTIONS]})
    print(f"Contexto: {any_p.context_chars} caracteres, huella {any_p.context_sha}, estado {ctx['estado']}")
    calls = {n: 0 for n in options}
    runs = []
    for rnd in range(1, args.rondas + 1):
        for kind, q in QUESTIONS:
            for n in options:
                a = providers[n].answer(q)
                cold = calls[n] == 0
                calls[n] += 1
                rec = {"opcion": n, "modelo": providers[n].model, "modo": providers[n].mode, "ronda": rnd,
                       "pregunta": kind, "fria": cold, "error": a.error, "respuesta": a.text,
                       "es_pregunta": a.is_question, "llm_s": round(a.latency, 2),
                       "codex_inicio_s": a.timings.get("codex_inicio"), "codex_ejecucion_s": a.timings.get("codex_ejecucion"),
                       "codex_total_s": a.timings.get("codex_total"), "tokens_entrada": a.input_tokens,
                       "tokens_cache": a.cached_tokens, "tokens_salida": a.output_tokens,
                       "avisos_codex": a.warnings or None, "banderas": flags(kind, a.text) if a.text else []}
                write(rec)
                runs.append(rec)
                lat = rec["codex_total_s"]
                print(f"r{rnd} {kind:17} op{n} {'FRÍA' if cold else 'tibia'} codex={lat if lat is None else round(lat, 2)}s "
                      f"total={rec['llm_s']}s in={a.input_tokens}(c{a.cached_tokens}) out={a.output_tokens} "
                      f"{'ERROR: ' + a.error if a.error else ''} {rec['banderas'] or ''}", flush=True)

    print("\n=== Resumen por opción (segundos; excluye errores) ===")
    for n in options:
        mine = [r for r in runs if r["opcion"] == n and not r["error"]]
        cold = [r for r in mine if r["fria"]]
        warm = [r for r in mine if not r["fria"]]

        def stat(rs, key):
            vals = [r[key] for r in rs if r[key] is not None]
            return (f"{statistics.mean(vals):.2f} (med {statistics.median(vals):.2f}, "
                    f"mín {min(vals):.2f}, máx {max(vals):.2f}, n={len(vals)})") if vals else "—"
        tok = [r["tokens_entrada"] for r in mine if r["tokens_entrada"] is not None]
        cac = [r["tokens_cache"] for r in mine if r["tokens_cache"] is not None]
        out = [r["tokens_salida"] for r in mine if r["tokens_salida"] is not None]
        o = MODEL_OPTIONS[n]
        print(f"[{n}] {o['label']} — {o['mode']}")
        print(f"    fría    codex_total: {stat(cold, 'codex_total_s')} | total: {stat(cold, 'llm_s')}")
        print(f"    tibias  codex_total: {stat(warm, 'codex_total_s')} | total: {stat(warm, 'llm_s')}")
        print(f"    tokens entrada media {statistics.mean(tok) if tok else 0:.0f}, caché media "
              f"{statistics.mean(cac) if cac else 0:.0f}, salida media {statistics.mean(out) if out else 0:.0f}")
        print(f"    errores: {sum(1 for r in runs if r['opcion'] == n and r['error'])} | con banderas: "
              f"{sum(1 for r in mine if r['banderas'])}/{len(mine)} | avisos de Codex: "
              f"{sum(1 for r in mine if r['avisos_codex'])}")
    print(f"\nDetalle completo: {path}")
    f.close()


if __name__ == "__main__":
    main()
