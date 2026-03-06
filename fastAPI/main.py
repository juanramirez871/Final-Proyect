from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from config import embed_model, collection, memory_llama
from service import call_model_vits, call_model_llama
import json


app = FastAPI(
    title="KeepCoding",
    description="Prueba Final keepcoding :)",
    version="1.0.0"
)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    
    orders = []
    with open("database/orders.json", "r") as f:
        orders = json.load(f)

    print(orders)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "orders": orders
    })


@app.get("/knowledge")
def read_knowledge(query: str, request: Request):

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
def generate_speech(query: str, request: Request):

    path = call_model_vits(query)
    return {
        "query": query,
        "path": path
    }


@app.get("/generate_text")
def generate_text(message: str, request: Request):

    text = call_model_llama(memory_llama, message)
    return {
        "messages": memory_llama,
        "text": text
    }


@app.get("/assistant")
def assistant(message: str, request: Request):
    
    text = call_model_llama(memory_llama, message)
    path = call_model_vits(text)
    return {
        "messages": memory_llama,
        "text": text,
        "path": path
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)