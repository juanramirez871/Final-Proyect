import json
import requests
import websocket
import threading
import socket
import time
import audioop
import numpy as np
import librosa
import torch
import os
import subprocess
from silero_vad import VADIterator
import mlx_whisper


ARI_URL = "http://localhost:8088"
APP_NAME = "assistant_IA"
USERNAME = "keepcoding"
PASSWORD = "123"

WS_URL = f"ws://localhost:8088/ari/events?app={APP_NAME}&api_key={USERNAME}:{PASSWORD}"
RTP_PORT = 50000
FASTAPI_URL = "http://localhost:8000/assistant"

SILENCE_TIMEOUT = 0.8
MODEL_ID = "mlx-community/whisper-large-v3-turbo"

SOUNDS_DIR = "./sounds"
os.makedirs(SOUNDS_DIR, exist_ok=True)


mlx_whisper.transcribe(
    np.zeros(16000, dtype=np.float32),
    path_or_hf_repo=MODEL_ID,
    verbose=False
)


vad_model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    trust_repo=True
)
(get_speech_ts, _, _, _, _) = utils


class CallSession:

    def __init__(self, channel_id, bridge_id):
        self.channel_id = channel_id
        self.bridge_id = bridge_id
        self.external_channel_id = None
        self.rtp_source_addr = None
        self.speech_buffer = []
        self.last_speech_time = None
        self.active = True
        self.processing_response = False

    def stop(self):
        self.active = False

    @staticmethod
    def ulaw_to_pcm(data):
        return audioop.ulaw2lin(data, 2)

    @staticmethod
    def convert_audio_for_asterisk(input_path):
        if not os.path.exists(input_path):
            return None

        unique_name = f"assistant_{int(time.time() * 1000)}"
        output_path = os.path.abspath(os.path.join(SOUNDS_DIR, f"{unique_name}.wav"))
        command = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ar", "16000",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-f", "s16le",
            output_path.replace(".wav", ".sln16")
        ]

        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ffmpeg error: {result.stderr}")
            return None
        
        final_path = output_path.replace(".wav", ".sln16")
        return final_path if os.path.exists(final_path) else None


    def play_audio(self, audio_path):
        if not audio_path or not self.channel_id:
            return
        
        filename = os.path.splitext(os.path.basename(audio_path))[0]
        resp = requests.post(
            f"{ARI_URL}/ari/channels/{self.channel_id}/play",
            json={"media": f"sound:{filename}"},
            auth=(USERNAME, PASSWORD)
        )

    def cleanup(self):
        if self.external_channel_id:
            resp = requests.delete(
                f"{ARI_URL}/ari/channels/{self.external_channel_id}",
                auth=(USERNAME, PASSWORD)
            )

        if self.bridge_id:
            resp = requests.delete(
                f"{ARI_URL}/ari/bridges/{self.bridge_id}",
                auth=(USERNAME, PASSWORD)
            )


    def transcribe_full_audio(self, pcm_bytes):
        try:
            audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            result = mlx_whisper.transcribe(
                audio_np,
                path_or_hf_repo=MODEL_ID,
                language="es",
                task="transcribe",
                verbose=False
            )
            
            text = result["text"].strip()
            if text:
                print(f"{self.channel_id} Usuario dijo: {text}")
                return text

        except Exception as e:
            print(f"{self.channel_id} Error transcribiendo: {e}")
        return None

    def call_ai_api(self, user_text):
        try:
            response = requests.get(
                FASTAPI_URL,
                params={
                    "query": user_text,
                    "session_id": self.channel_id,
                    "max_tokens": 100,
                    "top_k": 3
                },
                timeout=60
            )
            data = response.json()
            if data["status"] == "success":
                print(f"{self.channel_id} Respuesta IA: {data['response']}")
                return data["audio"]

        except Exception as e:
            print(f"{self.channel_id} Error API: {e}")
        return None

    def process_turn(self, full_audio):
        self.processing_response = True
        try:
            user_text = self.transcribe_full_audio(full_audio)
            if user_text:
                audio_path = self.call_ai_api(user_text)
                if audio_path:
                    converted_path = self.convert_audio_for_asterisk(audio_path)
                    if converted_path:
                        self.play_audio(converted_path)

        except Exception as e:
            print(f"{self.channel_id} Error en process_turn: {e}")  

        finally:
            self.processing_response = False

    def handle_rtp_chunk(self, pcm_chunk):
        if not self.active or self.processing_response:
            return

        audio_np = np.frombuffer(pcm_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        speech_timestamps = get_speech_ts(
            torch.from_numpy(audio_np),
            vad_model,
            sampling_rate=16000
        )

        if speech_timestamps:
            self.speech_buffer.append(pcm_chunk)
            self.last_speech_time = time.time()
        else:
            if (self.last_speech_time and
                    time.time() - self.last_speech_time > SILENCE_TIMEOUT):
                full_audio = b"".join(self.speech_buffer)
                self.speech_buffer = []
                self.last_speech_time = None
                threading.Thread(
                    target=self.process_turn,
                    args=(full_audio,),
                    daemon=True
                ).start()


active_sessions: dict[str, CallSession] = {}
sessions_lock = threading.Lock()
addr_to_channel: dict[str, str] = {}
addr_lock = threading.Lock()


def rtp_dispatcher():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", RTP_PORT))
    sock.settimeout(1.0)
    temp_chunks: dict[str, bytes] = {}

    while True:
        try:
            data, addr = sock.recvfrom(4096)
        except socket.timeout:
            continue
        except Exception as e:
            print(f"RTP dispatcher error: {e}")
            continue

        addr_key = f"{addr[0]}:{addr[1]}"
        rtp_payload = data[12:]

        pcm_data = audioop.byteswap(rtp_payload, 2)
        buf = temp_chunks.get(addr_key, b"") + pcm_data

        if len(buf) < 16000:
            temp_chunks[addr_key] = buf
            continue

        temp_chunks[addr_key] = b""        
        with addr_lock:
            channel_id = addr_to_channel.get(addr_key)

        if not channel_id:
            with sessions_lock:
                for cid, session in active_sessions.items():
                    if session.rtp_source_addr is None:
                        session.rtp_source_addr = addr_key
                        channel_id = cid
                        break
            if channel_id:
                with addr_lock:
                    addr_to_channel[addr_key] = channel_id

        if not channel_id:
            continue

        with sessions_lock:
            session = active_sessions.get(channel_id)

        if session and session.active:
            session.handle_rtp_chunk(buf)


def answer_call(channel_id):
    resp = requests.post(
        f"{ARI_URL}/ari/channels/{channel_id}/answer",
        auth=(USERNAME, PASSWORD)
    )


def create_bridge():
    r = requests.post(
        f"{ARI_URL}/ari/bridges",
        json={"type": "mixing"},
        auth=(USERNAME, PASSWORD)
    )

    return r.json()["id"]


def add_channel_to_bridge(bridge_id, channel_id):
    requests.post(
        f"{ARI_URL}/ari/bridges/{bridge_id}/addChannel",
        json={"channel": channel_id},
        auth=(USERNAME, PASSWORD)
    )


def create_external_media():
    r = requests.post(
        f"{ARI_URL}/ari/channels/externalMedia",
        params={
            "app": APP_NAME,
            "external_host": f"host.docker.internal:{RTP_PORT}",
            "format": "slin16"
        },
        auth=(USERNAME, PASSWORD)
    )

    return r.json()["id"]


def on_message(ws, message):
    event = json.loads(message)
    event_type = event.get("type")
    channel = event.get("channel", {})
    channel_id = channel.get("id", "")
    channel_name = channel.get("name", "")

    if event_type == "StasisStart":

        if "UnicastRTP" in channel_name:
            return

        print(f"Llamada entrante: {channel_id}")
        answer_call(channel_id)
        bridge_id = create_bridge()
        add_channel_to_bridge(bridge_id, channel_id)

        external_channel_id = create_external_media()
        add_channel_to_bridge(bridge_id, external_channel_id)

        session = CallSession(channel_id, bridge_id)
        session.external_channel_id = external_channel_id

        with sessions_lock:
            active_sessions[channel_id] = session

    elif event_type in ("StasisEnd", "ChannelDestroyed", "ChannelHangupRequest"):

        if "UnicastRTP" in channel_name:
            return

        print(f"[{channel_id}] Evento de cierre: {event_type}")
        with sessions_lock:
            session = active_sessions.pop(channel_id, None)

        if session:
            session.stop()
            with addr_lock:
                keys_to_remove = [k for k, v in addr_to_channel.items() if v == channel_id]
                for k in keys_to_remove:
                    del addr_to_channel[k]

            session.cleanup()


def on_error(ws, error):
    print(f"WebSocket error: {error}")


def on_close(ws, close_status_code, close_msg):
    print(f"WebSocket cerrado: {close_status_code} {close_msg}")


def on_open(ws):
    print("WebSocket conectado a Asterisk ARI.")


if __name__ == "__main__":
    threading.Thread(target=rtp_dispatcher, daemon=True).start()
    ws = websocket.WebSocketApp(
        WS_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    ws.run_forever()