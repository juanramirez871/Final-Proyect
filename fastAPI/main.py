import base64
from fastapi import FastAPI, HTTPException
from typing import Optional
from models import RAGResponse, KnowledgeResponse, TTSSuccessResponse, AssistantResponse
from services import generate_response_from_model
from config import collection, embed_model, SYSTEM_PROMPT
from memory import ConversationStore
from tts import generar_audio

conv_store = ConversationStore()
app = FastAPI(
    title="KeepCoding",
    description="Prueba Final keepcoding :)",
    version="1.0.0"
)


@app.get("/model_colombiano", response_model=RAGResponse)
def model_colombiano(prompt: str, max_tokens: Optional[int] = 100):
    try:
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
def get_knowledge(query: str = "", n_results: int = 5):
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



@app.get("/rag_session", response_model=RAGResponse)
def rag_session(query: str, session_id: str = "default", max_tokens: Optional[int] = 1000000, top_k: int = 3):
    try:
        q_emb = embed_model.encode([query])

        if hasattr(q_emb, "tolist"):
            q_emb = q_emb.tolist()
        if not isinstance(q_emb[0], (list, tuple)):
            q_embs = [q_emb]
        else:
            q_embs = q_emb

        results = collection.query(query_embeddings=q_embs, n_results=top_k)

        docs = results.get("documents")
        if isinstance(docs, list) and len(docs) > 0 and isinstance(docs[0], list):
            docs = docs[0]

        knowledge_text = ""
        if docs:
            knowledge_text = "\n\n--- Conocimiento interno ---\n"
            for i, d in enumerate(docs[:top_k], 1):
                knowledge_text += f"[{i}] {d}\n"

        history = conv_store.get_history(session_id)
        conv_store.add_user_message(session_id, query)
        result = generate_response_from_model(
            prompt=query,
            max_tokens=max_tokens,
            history=history,
            knowledge=knowledge_text,
        )

        if result.get("status") != "success":
            raise HTTPException(status_code=500, detail=result.get("error", "Error desconocido en el modelo"))

        model_text = result.get("response", "")
        conv_store.add_model_message(session_id, model_text)

        return RAGResponse(prompt=query, response=model_text, status="success")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en rag_session: {str(e)}")


@app.get("/tts", response_model=TTSSuccessResponse)
def tts(text: str):
    try:
        generar_audio(text, nombre="result.wav")
        with open("result.wav", "rb") as f:
            audio_bytes = f.read()

        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        return TTSSuccessResponse(status="success", audio=audio_b64)

    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="No se encontró el archivo de audio generado")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en TTS: {str(e)}")


@app.get("/assistant", response_model=AssistantResponse)
def assistant(query: str, session_id: str = "default", max_tokens: Optional[int] = 100, top_k: int = 3):
    try:
        q_emb = embed_model.encode([query])
        if hasattr(q_emb, "tolist"):
            q_emb = q_emb.tolist()
        if not isinstance(q_emb[0], (list, tuple)):
            q_embs = [q_emb]
        else:
            q_embs = q_emb

        results = collection.query(query_embeddings=q_embs, n_results=top_k)
        docs = results.get("documents")
        if isinstance(docs, list) and len(docs) > 0 and isinstance(docs[0], list):
            docs = docs[0]

        knowledge_text = ""
        if docs:
            knowledge_text = "\n\n--- Conocimiento interno ---\n"
            for i, d in enumerate(docs[:top_k], 1):
                knowledge_text += f"[{i}] {d}\n"

        history = conv_store.get_history(session_id)
        conv_store.add_user_message(session_id, query)
        result = generate_response_from_model(
            prompt=query,
            max_tokens=max_tokens,
            history=history,
            knowledge=knowledge_text,
        )

        if result.get("status") != "success":
            raise HTTPException(status_code=500, detail=result.get("error", "Error desconocido en el modelo"))

        model_text = result.get("response", "")
        conv_store.add_model_message(session_id, model_text)

        nombre = "assistant.wav"
        generar_audio(model_text, nombre=nombre)
        with open(nombre, "rb") as f:
            audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        return AssistantResponse(prompt=query, response=model_text, audio=audio_b64, status="success")

    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="No se encontró el archivo de audio generado")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en assistant: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
