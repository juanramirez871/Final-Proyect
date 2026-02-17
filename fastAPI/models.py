from pydantic import BaseModel, Field
from typing import Optional


class RAGRequest(BaseModel):
    prompt: str = Field(..., description="Prompt o pregunta para el modelo")
    max_tokens: Optional[int] = Field(
        default=100, 
        description="Número máximo de tokens a generar"
    )


class RAGResponse(BaseModel):
    prompt: str = Field(..., description="Prompt enviado")
    response: str = Field(..., description="Respuesta generada por el modelo")
    status: str = Field(default="success", description="Estado de la ejecución")
    error: Optional[str] = Field(default=None, description="Mensaje de error si ocurre")


class KnowledgeItem(BaseModel):
    id: Optional[str] = Field(None, description="Identificador del documento")
    document: Optional[str] = Field(None, description="Texto del documento")
    metadata: Optional[dict] = Field(None, description="Metadatos asociados al documento")
    distance: Optional[float] = Field(None, description="Distancia/score de la similitud")


class KnowledgeResponse(BaseModel):
    status: str = Field(default="success", description="Estado de la consulta")
    data: Optional[list[KnowledgeItem]] = Field(default_factory=list, description="Resultados de la consulta")
