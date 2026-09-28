"""Local Spanish caption server using downloadable faster-whisper."""

import json
import os
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4

import numpy as np
from faster_whisper import WhisperModel

ROOT = Path(__file__).parent
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")
MODEL = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
session_audio = {}
session_lock = threading.Lock()
model_lock = threading.Lock()
MINIMUM_SAMPLES = 16000 * 5
INITIAL_PROMPT = (
    "Conversación en español. Contexto médico y de telemedicina: "
    "paciente, médico, síntomas, dolor, medicamento, diagnóstico, presión arterial."
)


def clean_audio(pcm_bytes):
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if samples.size == 0:
        return samples
    samples -= np.mean(samples)
    peak = np.max(np.abs(samples))
    if peak > 0.02:
        samples = np.clip(samples * min(1.0, 0.85 / peak), -1.0, 1.0)
    return samples


def transcribe(pcm_bytes):
    audio = clean_audio(pcm_bytes)
    if audio.size == 0 or np.max(np.abs(audio)) < 0.008:
        return ""
    with model_lock:
        segments, _ = MODEL.transcribe(
            audio,
            language="es",
            task="transcribe",
            beam_size=5,
            best_of=5,
            temperature=0.0,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 350},
            condition_on_previous_text=True,
            initial_prompt=INITIAL_PROMPT,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


class AppHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        if self.path == "/api/health":
            body = json.dumps({
                "status": "ok",
                "recognition": "faster-whisper-local",
                "model": MODEL_SIZE,
                "device": "cpu-int8",
            }).encode("utf-8")
            self._json_response(body)
            return
        super().do_GET()

    def do_POST(self):
        if self.path != "/api/transcribe":
            self.send_error(404)
            return
        session_id = self.headers.get("X-Session-Id") or str(uuid4())
        length = int(self.headers.get("Content-Length", "0"))
        with session_lock:
            buffer = session_audio.setdefault(session_id, bytearray())
            buffer.extend(self.rfile.read(length))
            if len(buffer) < MINIMUM_SAMPLES * 2:
                self._json_response(json.dumps({"sessionId": session_id, "text": "", "final": False}).encode())
                return
            pcm_bytes = bytes(buffer)
            buffer.clear()
        try:
            text = transcribe(pcm_bytes)
        except Exception as error:
            self.send_error(500, "Transcription failed: %s" % error)
            return
        body = json.dumps({"sessionId": session_id, "text": text, "final": bool(text)}, ensure_ascii=False).encode("utf-8")
        self._json_response(body)

    def _json_response(self, body):
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        print("%s - %s" % (self.address_string(), format % args))


if __name__ == "__main__":
    port = int(os.environ.get("CAPTION_PORT", "8765"))
    server = ThreadingHTTPServer(("127.0.0.1", port), AppHandler)
    print("Spanish Live Captions: http://127.0.0.1:%d" % port)
    print("Local faster-whisper model: %s (CPU int8)" % MODEL_SIZE)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()
