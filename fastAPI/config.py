import os
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "model_llama_fine_tuning_run" / "llama_colombiano_mlx_q4"
DEFAULT_MAX_TOKENS = 100
DEFAULT_TEMPERATURE = 0.7
DB_PATH = BASE_DIR / "chromadb" / "chroma_db"

embed_model = SentenceTransformer("BAAI/bge-m3")
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection("atomi_knowledge")