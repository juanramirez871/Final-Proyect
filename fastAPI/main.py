from fastapi import FastAPI, HTTPException
from typing import Optional
from models import RAGResponse
from services import generate_response_from_model

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
