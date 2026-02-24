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


def ulaw_to_pcm(data):
    return audioop.ulaw2lin(data, 2)


def convert_audio_for_asterisk(input_path):
    if not os.path.exists(input_path):
        return None

    unique_name = f"assistant_{int(time.time())}.wav"
    output_path = os.path.abspath(os.path.join(SOUNDS_DIR, unique_name))
    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ar", "8000",
        "-ac", "1",
        output_path
    ]

    result = subprocess.run(command)

    if result.returncode != 0:
        print("Error ejecutando ffmpeg")
        return None

    if not os.path.exists(output_path):
        print("No se creó el archivo convertido")
        return None

    return output_path


def play_audio(channel_id, audio_path):
    if not audio_path:
        return

    filename = os.path.splitext(os.path.basename(audio_path))[0]
    requests.post(
        f"{ARI_URL}/ari/channels/{channel_id}/play",
        json={"media": f"sound:{filename}"},
        auth=(USERNAME, PASSWORD)
    )


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
            return text

    except Exception as e:
        print("Error transcribiendo:", e)

    return None


def call_ai_api(user_text):
    try:
        response = requests.get(
            FASTAPI_URL,
            params={
                "query": user_text,
                "session_id": "default",
                "max_tokens": 100,
                "top_k": 3
            },
            timeout=60
        )

        data = response.json()

        if data["status"] == "success":
            print("Response:", data["response"])
            return data["audio"]

    except Exception as e:
        print("Error llamando API:", e)

    return None


def answer_call(channel_id):
    requests.post(f"{ARI_URL}/ari/channels/{channel_id}/answer", auth=(USERNAME, PASSWORD))


def create_bridge():
    r = requests.post(f"{ARI_URL}/ari/bridges",
                      json={"type": "mixing"},
                      auth=(USERNAME, PASSWORD))

    return r.json()["id"]


def add_channel_to_bridge(bridge_id, channel_id):
    requests.post(f"{ARI_URL}/ari/bridges/{bridge_id}/addChannel",
                  json={"channel": channel_id},
                  auth=(USERNAME, PASSWORD))

def create_external_media():
    r = requests.post(
        f"{ARI_URL}/ari/channels/externalMedia",
        params={
            "app": APP_NAME,
            "external_host": f"host.docker.internal:{RTP_PORT}",
            "format": "ulaw"
        },
        auth=(USERNAME, PASSWORD)
    )

    return r.json()["id"]


speech_buffer = []
last_speech_time = None
current_channel_id = None


def rtp_listener():
    global speech_buffer, last_speech_time, current_channel_id

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
                    user_text = transcribe_full_audio(full_audio)

                    if user_text and current_channel_id:
                        audio_path = call_ai_api(user_text)
                        if audio_path:
                            converted_path = convert_audio_for_asterisk(audio_path)
                            if converted_path:
                                play_audio(current_channel_id, converted_path)

                    speech_buffer = []
                    last_speech_time = None

            temp_chunk = b""


def on_message(ws, message):
    global current_channel_id

    event = json.loads(message)
    if event["type"] == "StasisStart":

        channel = event["channel"]
        channel_id = channel["id"]
        channel_name = channel.get("name", "")

        if "UnicastRTP" in channel_name:
            return

        current_channel_id = channel_id
        print("Llamada entrante:", channel_id)

        answer_call(channel_id)
        bridge_id = create_bridge()
        add_channel_to_bridge(bridge_id, channel_id)

        external_channel_id = create_external_media()
        add_channel_to_bridge(bridge_id, external_channel_id)

def on_open(ws):
    pass


if __name__ == "__main__":
    threading.Thread(target=rtp_listener, daemon=True).start()
    ws = websocket.WebSocketApp(
        WS_URL,
        on_message=on_message
    )

    ws.on_open = on_open
    ws.run_forever()