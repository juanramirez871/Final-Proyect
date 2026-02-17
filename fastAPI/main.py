from fastapi import FastAPI, HTTPException
from typing import Optional
from models import RAGResponse, KnowledgeResponse
from services import generate_response_from_model
from config import collection, embed_model

app = FastAPI(
    title="KeepCoding",
    description="Prueba Final keepcoding",
    version="1.0.0"
)


@app.get("/rag_model_colombiano", response_model=RAGResponse)
def rag_model_colombiano(prompt: str, max_tokens: Optional[int] = 100):
    try:

        if not prompt or not prompt.strip():
            raise HTTPException(
                status_code=400,
                detail="El prompt no puede estar vacío"
            )
        
        result = generate_response_from_model(
            prompt=prompt.strip(),
            max_tokens=max_tokens
        )
        
        if result.get("status") != "success":
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Error desconocido en el modelo")
            )
        
        return RAGResponse(
            prompt=prompt,
            response=result["response"],
            status="success"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar la solicitud: {str(e)}"
        )


@app.get("/knowledge", response_model=KnowledgeResponse)
def get_knowledge(query: str = "Cuanto vale los jabones de mano?", n_results: int = 5):
    try:

        query_emb = embed_model.encode([query])

        if hasattr(query_emb, "tolist"):
            query_emb = query_emb.tolist()
        if len(query_emb) == 0:
            raise HTTPException(status_code=500, detail="Embedding vacío devuelto por el modelo de embeddings.")
        if not isinstance(query_emb[0], (list, tuple)):
            q_embs = [query_emb]
        else:
            q_embs = query_emb

        results = collection.query(
            query_embeddings=q_embs,
            n_results=n_results
        )

        docs = results.get("documents")
        ids = results.get("ids")
        metadatas = results.get("metadatas")
        distances = results.get("distances")

        if isinstance(docs, list) and len(docs) > 0 and isinstance(docs[0], list):
            docs = docs[0]
        if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
            ids = ids[0]
        if isinstance(metadatas, list) and len(metadatas) > 0 and isinstance(metadatas[0], list):
            metadatas = metadatas[0]
        if isinstance(distances, list) and len(distances) > 0 and isinstance(distances[0], list):
            distances = distances[0]

        items = []
        length = 0
        if docs:
            length = len(docs)
        elif ids:
            length = len(ids)

        for i in range(length):
            item = {
                "id": ids[i] if ids and i < len(ids) else None,
                "document": docs[i] if docs and i < len(docs) else None,
                "metadata": metadatas[i] if metadatas and i < len(metadatas) else None,
                "distance": float(distances[i]) if distances and i < len(distances) else None,
            }
            items.append(item)

        return KnowledgeResponse(status="success", data=items)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)