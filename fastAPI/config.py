import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DB_PATH = BASE_DIR / "chromadb" / "chroma_db"

embed_model = SentenceTransformer("BAAI/bge-m3")
client = chromadb.PersistentClient(path=str(DB_PATH))
collection = client.get_collection("products")
memory_llama = []

VITS_SERVER = "http://localhost:9001/tts"
LLAMA_SERVER = "http://localhost:9002/chat"