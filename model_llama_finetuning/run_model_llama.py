import json
import re
import random
import torch
import chromadb
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from num2words import num2words


app = FastAPI()

BASE_DIR = Path(__file__).parent.parent
DB_PATH = BASE_DIR / "chromadb" / "chroma_db"
MODEL_PATH = BASE_DIR / "model_llama_finetuning" / "llama_ventas_colombiano_merged"
ORDERS_FILE = BASE_DIR / "fastAPI" / "database" / "orders.json"

DEFAULT_MAX_TOKENS = 120
DEFAULT_TEMPERATURE = 0.2
SYSTEM_PROMPT = (
    "Eres un vendedor colombiano amigable y cercano. "
    "Hablas con palabras y expresiones típicas de Colombia como parce, mano, bacano.\n\n"

    "--- USO DE HERRAMIENTAS ---\n"

    "1. Si el cliente pregunta por productos disponibles usa 'get_products'.\n"
    "Responde SOLO con este JSON:\n"
    "{\"tool\": \"get_products\", \"query\": \"término\"}\n\n"

    "2. Si el cliente confirma compra y da teléfono usa 'create_order'.\n"
    "Responde SOLO con este JSON:\n"
    "{\"tool\": \"create_order\", \"product_name\": \"nombre\", "
    "\"quantity\": cantidad, \"price\": precio, \"customer_phone\": telefono}\n"
)

embed_model = SentenceTransformer("BAAI/bge-m3")
client = chromadb.PersistentClient(path=str(DB_PATH))
collection = client.get_collection("products")
model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_PATH),
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model.config.use_cache = True
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))

class ChatRequest(BaseModel):
    messages: list


def numero_a_palabras(n):
    try:
        return num2words(int(n), lang="es")
    except:
        return str(n)


def convertir_numeros(text):

    def reemplazar_dinero(match):

        digits = re.sub(r"[^\d]", "", match.group())
        if not digits:
            return match.group()

        palabras = numero_a_palabras(digits)
        return f"{palabras} pesos"

    text = re.sub(r"\$[\d\.]+", reemplazar_dinero, text)


    def reemplazar_litros(match):
        num = match.group(1)
        palabras = numero_a_palabras(num)

        return f"{palabras} litros"

    text = re.sub(r"(\d+)\s*[lL]", reemplazar_litros, text)

    def reemplazar_numero(match):
        return numero_a_palabras(match.group())
    
    text = re.sub(r"\b\d+\b", reemplazar_numero, text)
    return text


def precio_a_entero(precio_raw):

    solo_digitos = re.sub(r"[^\d]", "", str(precio_raw).split(".")[0])
    return int(solo_digitos) if solo_digitos else 0


def precio_colombiano(precio_raw):

    entero = precio_a_entero(precio_raw)
    return "$" + f"{entero:,}".replace(",", ".")



def call_tool(tool_name, args):

    if tool_name == "get_products":
        
        query = args.get("query", "")
        query_emb = embed_model.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=1
        )

        productos = []
        for doc in results["documents"][0]:

            lines = [l.strip() for l in doc.split("\n") if l.strip()]
            data = {}

            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    data[key.strip()] = value.strip()

            nombre = data.get("Nombre", "Producto")
            precio = precio_colombiano(data.get("Precio", "0"))

            productos.append({
                "nombre": nombre,
                "precio": precio
            })

        return json.dumps(
            {"status": "ok", "productos": productos},
            ensure_ascii=False
        )

    elif tool_name == "create_order":

        order_id = random.randint(10000, 99999)
        new_order = {
            "id": order_id,
            "product_name": args.get("product_name", ""),
            "quantity": args.get("quantity", 1),
            "price": precio_a_entero(args.get("price", 0)),
            "customer_phone": args.get("customer_phone", ""),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(ORDERS_FILE, "r+") as f:

            data = json.load(f)
            data.append(new_order)

            f.seek(0)
            json.dump(data, f, indent=4)

        return json.dumps({"status": "success", "order_id": order_id})

    return json.dumps({"status": "error"})


def build_prompt(messages):

    prompt = (
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
    )

    for msg in messages:

        prompt += (
            f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n"
            f"{msg['content']}<|eot_id|>"
        )

    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt



def call_model(messages):

    prompt = build_prompt(messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    eos_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    with torch.no_grad():

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model.generate(
                **inputs,
                max_new_tokens=DEFAULT_MAX_TOKENS,
                temperature=DEFAULT_TEMPERATURE,
                top_p=0.85,
                top_k=40,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=128004,
                eos_token_id=[tokenizer.eos_token_id, eos_id],
            )

    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()


def preparar_para_tts(text):

    text = convertir_numeros(text)
    text = re.sub(r"[¡!¿?:]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text



def generate_response(messages):

    messages = list(messages)
    for _ in range(3):

        response = call_model(messages)

        try:
            tool_call = json.loads(response)
        except:
            tool_call = None

        if isinstance(tool_call, dict) and "tool" in tool_call:

            print(tool_call["tool"])
            tool_result = call_tool(tool_call["tool"], tool_call)

            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"[TOOL RESULT]: {tool_result}"})

            continue

        break

    response = re.sub(r"\n+", " ", response)
    response = re.sub(r"\s+", " ", response).strip()
    response = preparar_para_tts(response)

    return response


@app.post("/chat")
def chat(req: ChatRequest):

    try:

        response = generate_response(req.messages)
        return {"response": response}

    except Exception as e:

        raise HTTPException(status_code=500, detail=str(e))