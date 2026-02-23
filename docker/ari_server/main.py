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

ARI_URL = "http://localhost:8088"
APP_NAME = "assistant_IA"
USERNAME = "keepcoding"
PASSWORD = "123"
WS_URL = f"ws://localhost:8088/ari/events?app={APP_NAME}&api_key={USERNAME}:{PASSWORD}"
RTP_PORT = 50000

MODEL_ID = "mlx-community/whisper-large-v3-turbo"
start_load = time.time()
mlx_whisper.transcribe(
    np.zeros(16000, dtype=np.float32),
    path_or_hf_repo=MODEL_ID,
    verbose=False
)

def ulaw_to_pcm(data):
    return audioop.ulaw2lin(data, 2)

audio_buffer = b""

def process_audio_chunk(pcm_bytes):
    global MODEL_ID
    audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    temp_filename = f"/tmp/{uuid.uuid4()}.wav"

    sf.write(temp_filename, audio_np, 8000)

    try:
        result = mlx_whisper.transcribe(
            temp_filename,
            path_or_hf_repo=MODEL_ID,
            verbose=False
        )

        text = result["text"].strip()
        if text:
            print("Usuario dijo:", text)

    except Exception as e:
        print("Error en transcripción:", e)

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

def rtp_listener():
    global audio_buffer

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", RTP_PORT))
    print(f"Escuchando RTP en puerto {RTP_PORT}")

    while True:
        data, addr = sock.recvfrom(2048)
        rtp_payload = data[12:]
        pcm_data = ulaw_to_pcm(rtp_payload)
        audio_buffer += pcm_data

        if len(audio_buffer) >= 16000 * 2:
            process_audio_chunk(audio_buffer)
            audio_buffer = b""

def wait_for_asterisk():
    while True:
        try:
            r = requests.get(f"{ARI_URL}/ari/asterisk/info", timeout=2, auth=(USERNAME, PASSWORD))
            if r.status_code == 200:
                print("Asterisk listo")
                break
        except Exception:
            pass
        print("asterisk no respondió")
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