import json
import requests
import websocket
import threading
import socket
import time
import audioop
import numpy as np
import soundfile as sf
import mlx_whisper
import uuid
import os
from silero_vad import VADIterator, get_speech_timestamps
import torch
import librosa
import time
import librosa

ARI_URL = "http://localhost:8088"
APP_NAME = "assistant_IA"
USERNAME = "keepcoding"
PASSWORD = "123"
WS_URL = f"ws://localhost:8088/ari/events?app={APP_NAME}&api_key={USERNAME}:{PASSWORD}"
RTP_PORT = 50000

speech_buffer = []
last_speech_time = None
SILENCE_TIMEOUT = 0.8
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_ID = "mlx-community/whisper-large-v3-turbo"

start_load = time.time()
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
(get_speech_ts, _, read_audio, _, _) = utils

def ulaw_to_pcm(data):
    return audioop.ulaw2lin(data, 2)

audio_buffer = b""

def process_audio_chunk_with_vad(pcm_bytes):
    try:
        audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        audio_16k = librosa.resample(audio_np, orig_sr=8000, target_sr=16000)
        speech_timestamps = get_speech_ts(
            torch.from_numpy(audio_16k),
            vad_model,
            sampling_rate=16000
        )

        if len(speech_timestamps) == 0:
            return

        result = mlx_whisper.transcribe(
            audio_16k,
            path_or_hf_repo=MODEL_ID,
            verbose=False
        )

        text = result["text"].strip()
        if text:
            print("Usuario dijo:", text)

    except Exception as e:
        print("Error en transcripción:", e)

def transcribe_full_audio(pcm_bytes):
    try:
        audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        audio_16k = librosa.resample(audio_np, orig_sr=8000, target_sr=16000)

        result = mlx_whisper.transcribe(
            audio_16k,
            path_or_hf_repo=MODEL_ID,
            language="es",
            task="transcribe",
            verbose=False
        )

        text = result["text"].strip()

        if text:
            print("\nUsuario dijo:", text)

    except Exception as e:
        print("Error en transcripción:", e)

def rtp_listener():
    global speech_buffer, last_speech_time

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", RTP_PORT))
    temp_chunk = b""

    while True:
        data, addr = sock.recvfrom(2048)
        rtp_payload = data[12:]
        pcm_data = ulaw_to_pcm(rtp_payload)
        temp_chunk += pcm_data

        if len(temp_chunk) >= 8000:
            audio_np = np.frombuffer(temp_chunk, dtype=np.int16).astype(np.float32) / 32768.0
            audio_16k = librosa.resample(audio_np, orig_sr=8000, target_sr=16000)
            speech_timestamps = get_speech_ts(
                torch.from_numpy(audio_16k),
                vad_model,
                sampling_rate=16000
            )

            if len(speech_timestamps) > 0:
                speech_buffer.append(temp_chunk)
                last_speech_time = time.time()
            else:
                if last_speech_time and (time.time() - last_speech_time > SILENCE_TIMEOUT):
                    
                    full_audio = b"".join(speech_buffer)
                    transcribe_full_audio(full_audio)
                    speech_buffer = []
                    last_speech_time = None

            temp_chunk = b""

def wait_for_asterisk():
    while True:
        try:
            r = requests.get(f"{ARI_URL}/ari/asterisk/info", timeout=2, auth=(USERNAME, PASSWORD))
            if r.status_code == 200:
                print("Asterisk listo")
                break
        except Exception:
            pass
        
        print("Esperando Asterisk...")
        time.sleep(2)

def answer_call(channel_id):
    requests.post(f"{ARI_URL}/ari/channels/{channel_id}/answer", auth=(USERNAME, PASSWORD))

def create_bridge():
    response = requests.post(f"{ARI_URL}/ari/bridges", json={"type": "mixing"}, auth=(USERNAME, PASSWORD))
    return response.json()["id"]

def add_channel_to_bridge(bridge_id, channel_id):
    requests.post(f"{ARI_URL}/ari/bridges/{bridge_id}/addChannel", json={"channel": channel_id}, auth=(USERNAME, PASSWORD))

def create_external_media():
    response = requests.post(
        f"{ARI_URL}/ari/channels/externalMedia",
        params={
            "app": APP_NAME,
            "external_host": f"host.docker.internal:{RTP_PORT}",
            "format": "ulaw"
        },
        auth=(USERNAME, PASSWORD)
    )

    return response.json()["id"]

def on_message(ws, message):
    event = json.loads(message)
    if event["type"] == "StasisStart":
        channel = event["channel"]
        channel_id = channel["id"]
        channel_name = channel.get("name", "")

        if "UnicastRTP" in channel_name:
            return

        print("Llamada entrante:", channel_id)
        answer_call(channel_id)
        bridge_id = create_bridge()
        add_channel_to_bridge(bridge_id, channel_id)

        external_channel_id = create_external_media()
        add_channel_to_bridge(bridge_id, external_channel_id)

def on_open(ws):
    print("Conectado a ARI")

def on_error(ws, error):
    print("Error WS:", error)

def on_close(ws, close_status_code, close_msg):
    print("WS cerrado:", close_status_code, close_msg)

if __name__ == "__main__":
    threading.Thread(target=rtp_listener, daemon=True).start()
    wait_for_asterisk()

    ws = websocket.WebSocketApp(
        WS_URL,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )

    ws.on_open = on_open
    ws.run_forever(ping_interval=30, ping_timeout=10)