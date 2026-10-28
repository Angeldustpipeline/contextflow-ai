import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import pyaudiowpatch as pyaudio

TARGET_RATE = 16000
FRAME = 512  # ventana de Silero a 16 kHz (32 ms)
FRAME_S = FRAME / TARGET_RATE
TAIL_S = 0.2  # silencio que se conserva tras la última voz


def find_loopback(p, output_name_hint=None):
    wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    default_out = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
    target = default_out
    if output_name_hint:
        for d in p.get_device_info_generator():
            if (d["hostApi"] == wasapi["index"] and d["maxOutputChannels"] > 0
                    and output_name_hint.lower() in d["name"].lower()):
                target = d
                break
    if not target.get("isLoopbackDevice"):
        for lb in p.get_loopback_device_info_generator():
            if target["name"] in lb["name"]:
                return target, lb
    return target, None


def to_mono_16k(raw, channels, rate):
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    x = x.reshape(-1, channels).mean(axis=1)
    if rate == TARGET_RATE:
        return x
    if rate % TARGET_RATE == 0:
        k = rate // TARGET_RATE
        n = len(x) // k * k
        return x[:n].reshape(-1, k).mean(axis=1)
    n_out = int(len(x) * TARGET_RATE / rate)
    return np.interp(np.linspace(0, len(x) - 1, n_out), np.arange(len(x)), x).astype(np.float32)


@dataclass
class VADConfig:
    silence_ms: int = 800
    preroll_ms: int = 300
    max_seconds: float = 30.0
    threshold: float = 0.5
    min_speech_ms: int = 250

    @classmethod
    def from_env(cls):
        d = cls()
        try:
            return cls(
                silence_ms=int(os.getenv("VAD_SILENCE_MS", d.silence_ms)),
                preroll_ms=int(os.getenv("VAD_PREROLL_MS", d.preroll_ms)),
                max_seconds=float(os.getenv("VAD_MAX_SECONDS", d.max_seconds)),
                threshold=float(os.getenv("VAD_THRESHOLD", d.threshold)),
                min_speech_ms=int(os.getenv("VAD_MIN_SPEECH_MS", d.min_speech_ms)),
            )
        except ValueError:
            raise ValueError("Las variables VAD_* de .env deben ser numéricas.")


class StreamVAD:
    """Silero VAD (modelo incluido en faster-whisper) en modo streaming, con estado persistente."""

    def __init__(self):
        from faster_whisper.vad import get_vad_model
        self.session = get_vad_model().session
        self.reset()

    def reset(self):
        self.h = np.zeros((1, 1, 128), dtype=np.float32)
        self.c = np.zeros((1, 1, 128), dtype=np.float32)
        self.ctx = np.zeros(64, dtype=np.float32)
        self.buf = np.zeros(0, dtype=np.float32)

    def feed(self, x):
        """Devuelve [(frame, prob_voz)] por cada ventana completa de 512 muestras."""
        self.buf = np.concatenate([self.buf, x])
        out = []
        while len(self.buf) >= FRAME:
            frame, self.buf = self.buf[:FRAME], self.buf[FRAME:]
            inp = np.concatenate([self.ctx, frame])[None, :]
            prob, self.h, self.c = self.session.run(None, {"input": inp, "h": self.h, "c": self.c})
            self.ctx = frame[-64:]
            out.append((frame, float(np.asarray(prob).reshape(-1)[0])))
        return out


class LoopbackCapture:
    """Captura WASAPI loopback y entrega intervenciones completas (VAD) en una cola acotada.

    Cada elemento: dict(audio, final, t_voice_end, t_ready, duration). Si final=False es un corte por
    duración máxima: el consumidor debe esperar al siguiente elemento final antes de usarlo como pregunta.
    """

    def __init__(self, device_hint="Altavoces", config=None, max_segments=4):
        self.device_hint = device_hint
        self.cfg = config or VADConfig.from_env()
        self.segments = queue.Queue(maxsize=max_segments)
        self.dropped = 0
        self.speaking = False
        self.last_error = None
        self._raw = queue.Queue(maxsize=400)
        self._stop = threading.Event()
        self._thread = None
        self._p = None
        self._stream = None
        self.device = None
        self.channels = 0
        self.rate = 0

    def start(self):
        self._p = pyaudio.PyAudio()
        _, lb = find_loopback(self._p, self.device_hint)
        if lb is None:
            self._p.terminate()
            raise RuntimeError("No se encontró el dispositivo loopback WASAPI")
        self.device = lb
        self.channels = int(lb["maxInputChannels"])
        self.rate = int(lb["defaultSampleRate"])
        self._vad = StreamVAD()
        self._stream = self._p.open(
            format=pyaudio.paInt16, channels=self.channels, rate=self.rate,
            input=True, input_device_index=lb["index"],
            frames_per_buffer=1024, stream_callback=self._callback,
        )
        self._thread = threading.Thread(target=self._segment, name="audio-segmenter", daemon=True)
        self._thread.start()

    def _callback(self, in_data, frame_count, time_info, status):
        try:
            self._raw.put_nowait((time.perf_counter(), in_data))
        except queue.Full:
            self.dropped += 1
        return (None, pyaudio.paContinue)

    def _put(self, item):
        if self.segments.full():
            try:
                self.segments.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
        self.segments.put_nowait(item)

    def _segment(self):
        cfg, vad = self.cfg, self._vad
        silence_s = cfg.silence_ms / 1000
        min_frames = max(1, int(cfg.min_speech_ms / 1000 / FRAME_S))
        max_frames = int(cfg.max_seconds / FRAME_S)
        tail = int(TAIL_S / FRAME_S)
        preroll = deque(maxlen=max(1, int(cfg.preroll_ms / 1000 / FRAME_S)))
        frames, speaking = [], False
        voiced = last_idx = 0
        last_voice_t = 0.0
        partial_sent = False

        def emit(final):
            end = min(len(frames), last_idx + 1 + tail)
            audio = np.concatenate(frames[:end]) if (end > 0 and voiced) else np.zeros(0, np.float32)
            self._put({"audio": audio, "final": final, "t_voice_end": last_voice_t,
                       "t_ready": time.perf_counter(), "duration": len(audio) / TARGET_RATE})

        while not self._stop.is_set():
            try:
                t_recv, data = self._raw.get(timeout=0.05)
            except queue.Empty:
                data = None
            if data is not None:
                try:
                    results = vad.feed(to_mono_16k(data, self.channels, self.rate))
                except Exception as e:
                    self.last_error = e
                    continue
                for fr, p in results:
                    is_voice = p >= cfg.threshold
                    if not speaking:
                        if is_voice:
                            speaking, frames = True, list(preroll) + [fr]
                            self.speaking = True
                            preroll.clear()
                            voiced, last_idx, last_voice_t = 1, len(frames) - 1, t_recv
                        else:
                            preroll.append(fr)
                        continue
                    frames.append(fr)
                    if is_voice:
                        voiced += 1
                        last_idx, last_voice_t = len(frames) - 1, t_recv
                    if len(frames) >= max_frames:
                        emit(final=False)
                        partial_sent = True
                        frames, voiced, last_idx = [], 0, -1
            # El silencio se mide por reloj: el loopback no entrega frames mientras no suena nada.
            if speaking and time.perf_counter() - last_voice_t >= silence_s:
                if voiced >= min_frames or partial_sent:
                    emit(final=True)
                speaking, frames, voiced, last_idx, partial_sent = False, [], 0, 0, False
                self.speaking = False
                preroll.clear()
                vad.reset()

    def stop(self):
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._p is not None:
            self._p.terminate()


if __name__ == "__main__":
    with pyaudio.PyAudio() as p:
        wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        print("Dispositivos WASAPI:")
        for d in p.get_device_info_generator():
            if d["hostApi"] == wasapi["index"]:
                kind = "LOOPBACK" if d.get("isLoopbackDevice") else ("OUT" if d["maxOutputChannels"] else "IN")
                print(f"  [{d['index']}] {kind:8} {d['name']} | ch={d['maxInputChannels']}/{d['maxOutputChannels']} | {int(d['defaultSampleRate'])} Hz")
        out, lb = find_loopback(p, "Altavoces")
        print("\nSalida seleccionada:", f"[{out['index']}] {out['name']}")
        if lb:
            print("Loopback:", f"[{lb['index']}] {lb['name']} | ch={lb['maxInputChannels']} | {int(lb['defaultSampleRate'])} Hz")
        else:
            print("Loopback: NO ENCONTRADO")
