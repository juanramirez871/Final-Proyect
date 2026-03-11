from fastapi import HTTPException
import requests
from config import VITS_SERVER, LLAMA_SERVER
from pathlib import Path


def call_model_vits(text: str):

    try:
        res = requests.get(
            VITS_SERVER,
            params={"texto": text},
            stream=True
        )

        if res.status_code != 200:
            raise HTTPException(status_code=500, detail="Error en servidor TTS")

        output_path = Path("generated_audio.wav")

        with open(output_path, "wb") as f:
            for chunk in res.iter_content(chunk_size=8192):
                f.write(chunk)


        absolute_path = str(output_path.resolve())
        return absolute_path

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def call_model_llama(memory_llama: list, message: str):

    memory_llama.append({
        "role": "user",
        "content": message
    })

    try:

        r = requests.post(
            LLAMA_SERVER,
            json={"messages": memory_llama}
        )

        if r.status_code != 200:
            raise HTTPException(status_code=500, detail="Error en servidor LLM")

        data = r.json()
        text = data["response"]

        memory_llama.append({
            "role": "assistant",
            "content": text
        })

        return text

    except Exception as e:

        raise HTTPException(status_code=500, detail=str(e))