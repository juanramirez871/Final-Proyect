from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from transformers import pipeline

app = FastAPI(
    title="KeepCoding",
    description="Prueba Final keepcoding",
    version="1.0.0"
)


@app.get("/rag_model_colombiano")
def rag_model_colombiano():
    pass



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
