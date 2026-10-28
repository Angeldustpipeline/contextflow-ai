"""Demostración local y determinista del pipeline de ContextFlow AI (sin micrófono, sin llamar a ningún LLM).

Reutiliza los mismos componentes que usa el asistente real: filtro local de preguntas (is_probable_question),
inicio sugerido local (suggest_opener) y carga de contexto (load_context), aplicados sobre un guion ficticio
de intervenciones de ejemplo. El paso de "generación de respuesta" se reemplaza por un DemoProvider local que
no llama a Codex CLI, a la API de OpenAI ni a ningún otro servicio: produce una respuesta fija, claramente
etiquetada como simulada, útil para ver el flujo completo (filtro -> inicio sugerido -> respuesta -> panel)
sin requerir audio real, Codex CLI autenticado ni créditos de API.

Uso:
    python demo.py
"""
import time

from rich.console import Console

from assistant import Answer, context_status, is_probable_question, load_context, suggest_opener

console = Console()

DEMO_CONTEXT_PATH = "context.example.md"

# Guion ficticio de intervenciones de ejemplo (ninguna corresponde a una entrevista real).
SCRIPT = [
    "Para partir, cuéntame un poco sobre ti y tu experiencia profesional.",
    "Muchas gracias por tu tiempo hoy.",  # no es una pregunta/petición: se filtra localmente
    "¿Qué herramientas usarías para dar seguimiento a los despachos de un centro de distribución?",
    "Cuéntame de una vez en que tuviste que manejar un plazo muy ajustado.",
]


class DemoProvider:
    """Sustituto local del proveedor LLM real. No hace llamadas de red ni ejecuta Codex CLI."""

    def __init__(self, context_text):
        self.context_chars = len(context_text)

    def answer(self, transcript):
        t0 = time.perf_counter()
        time.sleep(0.4)  # simula la latencia de un paso de generación real
        text = (
            "- [DEMO] Esta es una respuesta simulada; no se consultó ningún modelo de lenguaje.\n"
            "- [DEMO] En el asistente real, esta viñeta se generaría a partir de context.md y la "
            "transcripción real usando Codex CLI u OpenAI.\n"
            "- [DEMO] Para respuestas reales, usa `python main.py --assistant` con Codex CLI autenticado "
            "o LLM_PROVIDER=openai y una OPENAI_API_KEY válida."
        )
        return Answer(text=text, latency=time.perf_counter() - t0, is_question=True)


def main():
    console.print("[bold]ContextFlow AI — demostración local[/]")
    console.print("[dim]Guion ficticio, sin audio y sin llamadas reales a ningún LLM. "
                  "Solo para mostrar el flujo del pipeline.[/]\n")

    context_text = load_context(DEMO_CONTEXT_PATH)
    status = context_status(path=DEMO_CONTEXT_PATH)
    console.print(f"Contexto de demostración: {status['titulo']} ({len(context_text)} caracteres) "
                  f"[dim]— {status['estado']}: {status['detalle']}[/]\n")

    provider = DemoProvider(context_text)
    for n, line in enumerate(SCRIPT, start=1):
        console.print(f"[bold cyan]Intervención #{n}:[/] {line}")
        is_q, why = is_probable_question(line)
        if not is_q:
            console.print(f"[dim]-> filtrada localmente ({why}); no se envía al paso de generación.[/]\n")
            continue
        opener = suggest_opener(line)
        console.print(f"[dim]-> inicio sugerido (local, sin LLM): {opener}[/]")
        a = provider.answer(line)
        console.print(a.text, style="bold green", markup=False, highlight=False)
        console.print(f"[dim]latencia: {a.latency:.2f}s (simulada, no representativa del LLM real)[/]\n")


if __name__ == "__main__":
    main()
