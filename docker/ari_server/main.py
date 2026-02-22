import json
import requests
import websocket
import threading
import socket
import time

ARI_URL = "http://asterisk:8088"
APP_NAME = "assistant_IA"
USERNAME = "keepcoding"
PASSWORD = "123"
WS_URL = f"ws://asterisk:8088/ari/events?app={APP_NAME}&api_key={USERNAME}:{PASSWORD}"
RTP_PORT = 50000

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

def rtp_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", RTP_PORT))
    print(f"Escuchando RTP en puerto {RTP_PORT}")

    while True:
        data, addr = sock.recvfrom(2048)
        print("Paquete RTP:", len(data))

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
            "external_host": f"ari-server:{RTP_PORT}",
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
        print("Bridge creado:", bridge_id)
        add_channel_to_bridge(bridge_id, channel_id)

        external_channel_id = create_external_media()
        print("Canal External Media creado:", external_channel_id)
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