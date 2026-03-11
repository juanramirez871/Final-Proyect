from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from config import embed_model, collection, memory_llama
from service import call_model_vits, call_model_llama

import json

BASE_URL = "https://khhp4r8s0dwyry-8000.proxy.runpod.net"
app = FastAPI(
    title="KeepCoding",
    description="Prueba Final keepcoding",
    version="1.0.0"
)

templates = Jinja2Templates(
    directory=str(Path(__file__).parent / "templates")
)

app.mount("/fastAPI", StaticFiles(directory="."), name="fastAPI")
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):

    with open("database/orders.json", "r") as f:
        orders = json.load(f)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "orders": orders
        }
    )


@app.get("/knowledge")
def read_knowledge(query: str):

    query_embedding = embed_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )

    return {
        "query": query,
        "results": results
    }


@app.get("/generate_speech")
def generate_speech(query: str):

    path = call_model_vits(query)

    return {
        "query": query,
        "path": path
    }


@app.get("/generate_text")
def generate_text(message: str):

    text = call_model_llama(memory_llama, message)

    return {
        "messages": memory_llama,
        "text": text
    }


@app.get("/assistant")
def assistant(message: str):

    text = call_model_llama(memory_llama, message)
    path = call_model_vits(text)

    filename = Path(path).name
    audio_url = f"{BASE_URL}/fastAPI/{filename}"

    return {
        "text": text,
        "audio_url": audio_url
    }


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )