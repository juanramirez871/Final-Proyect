import json
import requests
import websocket
import threading
from requests.auth import HTTPBasicAuth

ARI_URL = "http://152.202.182.22:8088"
WS_URL = f"ws://152.202.182.22:8088/ari/events?app=assistant_IA&api_key=keepcoding:123"
USERNAME = "keepcoding"
PASSWORD = "123"

auth = HTTPBasicAuth(USERNAME, PASSWORD)

def answer_call(channel_id):
    url = f"{ARI_URL}/ari/channels/{channel_id}/answer"
    requests.post(url, auth=auth)

def play_sound(channel_id):
    url = f"{ARI_URL}/ari/channels/{channel_id}/play"
    data = {"media": "sound:hello-world"}
    requests.post(url, json=data, auth=auth)

def on_message(ws, message):
    event = json.loads(message)

    if event["type"] == "StasisStart":
        channel_id = event["channel"]["id"]
        print("Llamada entrante:", channel_id)

        answer_call(channel_id)
        play_sound(channel_id)

def on_error(ws, error):
    print("Error:", error)

def on_close(ws, close_status_code, close_msg):
    print("Conexión cerrada")

def on_open(ws):
    print("Conectado a ARI")

if __name__ == "__main__":
    ws = websocket.WebSocketApp(
        WS_URL,
        header=[
            f"Authorization: Basic {requests.auth._basic_auth_str(USERNAME, PASSWORD)}"
        ],
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )

    ws.on_open = on_open
    ws.run_forever()