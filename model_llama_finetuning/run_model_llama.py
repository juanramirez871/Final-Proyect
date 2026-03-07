import json
import argparse
import re
import random
import torch
import chromadb
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer


parser = argparse.ArgumentParser()
parser.add_argument("--messages", required=True)
args = parser.parse_args()
conversation = json.loads(args.messages)

BASE_DIR = Path(__file__).parent.parent
DB_PATH = BASE_DIR / "chromadb" / "chroma_db"
MODEL_PATH = BASE_DIR / "model_llama_finetuning" / "llama_ventas_colombiano_merged"
ORDERS_FILE = BASE_DIR / "fastAPI" / "database" / "orders.json"

DEFAULT_MAX_TOKENS = 120
DEFAULT_TEMPERATURE = 0.2

SYSTEM_PROMPT = (
    "Eres un vendedor colombiano amigable y cercano. "
    "Hablas con palabras y expresiones típicas de Colombia como parce, mano, bacano, etc. "
    "También usas diminutivos como cosita, platica, etc.\n\n"

    "--- USO DE HERRAMIENTAS ---\n"

    "1. Si el cliente pregunta por productos disponibles usa 'get_products'.\n"
    "Responde SOLO con este JSON:\n"
    "{\"tool\": \"get_products\", \"query\": \"término\"}\n\n"

    "2. Si el cliente confirma la compra y da su teléfono usa 'create_order'.\n"
    "Responde SOLO con este JSON:\n"
    "{\"tool\": \"create_order\", \"product_name\": \"nombre\", "
    "\"quantity\": cantidad, \"price\": precio, \"customer_phone\": telefono}\n\n"

    "3. Si el cliente saluda o conversa, responde normalmente como vendedor colombiano "
    "sin usar herramientas.\n"
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


def precio_a_entero(precio_raw) -> int:
    solo_digitos = re.sub(r'[^\d]', '', str(precio_raw).split('.')[0])
    return int(solo_digitos) if solo_digitos else 0


def precio_colombiano(precio_raw) -> str:
    entero = precio_a_entero(precio_raw)
    return "$" + f"{entero:,}".replace(",", ".")


def call_tool(tool_name: str, args: dict) -> str:
    
    if tool_name == "get_products":
        query = args.get("query", "")
        query_emb = embed_model.encode(query).tolist()
        results = collection.query(query_embeddings=[query_emb], n_results=1)

        productos = []
        for doc in results["documents"][0]:
            lines = [l.strip() for l in doc.split("\n") if l.strip()]
            data  = {}

            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    data[key.strip()] = value.strip()

            nombre = data.get("Nombre", "Producto")
            precio = precio_colombiano(data.get("Precio", "0"))  # string: "$670.000"
            productos.append({"nombre": nombre, "precio": precio})

        tool_result = json.dumps({"status": "ok", "productos": productos}, ensure_ascii=False)
        print(f"Knowledge={tool_result}")
        return tool_result

    elif tool_name == "create_order":
        order_id  = random.randint(10000, 99999)
        new_order = {
            "id": order_id,
            "product_name": args.get("product_name", ""),
            "quantity": args.get("quantity", 1),
            "price": precio_a_entero(args.get("price", 0)),
            "customer_phone": args.get("customer_phone", ""),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": "",
        }
        
        print("Orden a crear:", new_order)
        with open(ORDERS_FILE, "r+") as f:
            data = json.load(f)
            data.append(new_order)
            f.seek(0)
            json.dump(data, f, indent=4)

        tool_result = {"status": "success", "order_id": order_id}
        return json.dumps(tool_result, ensure_ascii=False)

    return json.dumps({"status": "error", "message": "herramienta desconocida"})


def build_prompt(messages: list) -> str:
    
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


def call_model(messages: list) -> str:
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


DIGITOS_TTS = {
    "0": "cero", "1": "uno", "2": "dos", "3": "tres", "4": "cuatro",
    "5": "cinco", "6": "seis", "7": "siete", "8": "ocho", "9": "nueve",
}

def num_a_palabras(n: int) -> str:
    if n == 0:
        return "cero"
    unidades = ["", "un", "dos", "tres", "cuatro", "cinco", "seis", "siete",
                "ocho", "nueve", "diez", "once", "doce", "trece", "catorce",
                "quince", "dieciséis", "diecisiete", "dieciocho", "diecinueve",
                "veinte", "veintiún", "veintidós", "veintitrés", "veinticuatro",
                "veinticinco", "veintiséis", "veintisiete", "veintiocho", "veintinueve"]

    decenas = ["", "", "veinte", "treinta", "cuarenta", "cincuenta","sesenta", "setenta", "ochenta", "noventa"]
    centenas = ["", "cien", "doscientos", "trescientos", "cuatrocientos", "quinientos", "seiscientos", "setecientos", "ochocientos", "novecientos"]

    def menos_mil(n):
        if n < 30:
            return unidades[n]
        elif n < 100:
            d, u = divmod(n, 10)
            return decenas[d] + (" y " + unidades[u] if u else "")
        else:
            c, r = divmod(n, 100)
            base = "ciento " + menos_mil(r) if (c == 1 and r) else centenas[c]
            return base + (" " + menos_mil(r) if r and c != 1 else "")

    def convertir(n):
        if n < 1_000:
            return menos_mil(n)
        elif n < 1_000_000:
            miles, r = divmod(n, 1_000)
            prefijo = "mil" if miles == 1 else menos_mil(miles) + " mil"
            return prefijo + (" " + menos_mil(r) if r else "")
        else:
            mils, r = divmod(n, 1_000_000)
            prefijo = "un millón" if mils == 1 else menos_mil(mils) + " millones"
            return prefijo + (" " + convertir(r) if r else "")

    return convertir(n)


def fix_prices_in_response(response: str, tool_result_json: str) -> str:
    try:
        data = json.loads(tool_result_json)
        productos = data.get("productos", [])
    except Exception:
        return response

    for p in productos:
        nombre = p["nombre"]
        precio_correcto = p["precio"]
        pattern = r'(' + re.escape(nombre) + r'\s+a\s+)\$[\d\.]*'
        response = re.sub(pattern, r'\g<1>' + precio_correcto, response, flags=re.IGNORECASE)

    return response


def preparar_para_tts(text: str) -> str:
    
    def reemplazar_precio(m):
        digits = re.sub(r'[^\d]', '', m.group(0))
        return (num_a_palabras(int(digits)) + " pesos") if digits else m.group(0)
    text = re.sub(r'\$[\d\.]+', reemplazar_precio, text)

    def reemplazar_codigo(m):
        return m.group(1) + " ".join(DIGITOS_TTS[d] for d in m.group(2))

    text = re.sub(r'([A-Z]+-\s*)(\d+)', reemplazar_codigo, text)
    text = re.sub(r'\b(\d+)\b', lambda m: num_a_palabras(int(m.group(1))), text)
    text = re.sub(r'[¡!¿?:]', '', text)
    text = re.sub(r'[/\\|#@%^&*_=+<>~`]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


COMPRA_KEYWORDS = [
    "me lo llevo", "me la llevo", "los llevo", "las llevo",
    "quiero comprar", "quiero pedir", "quiero ordenar", "quiero ese", "quiero esa",
    "lo quiero", "la quiero", "los quiero", "las quiero",
    "listo lo agarro", "listo la agarro", "de una lo quiero", "de una la quiero",
    "confirmado", "procésame", "procesame", "hágame el pedido", "hagame el pedido",
    "apúntame", "apuntame", "me quedo con", "voy a llevar", "vamos con",
    "listo me llevo", "dale que sea", "me convencí", "me convenci",
]

def tiene_intencion_compra(texto: str) -> bool:
    t = texto.lower()
    return any(kw in t for kw in COMPRA_KEYWORDS)

def tiene_telefono(texto: str) -> bool:
    return bool(re.search(r'\b3\d{9}\b', texto))

PEDIR_TELEFONO_RESPUESTAS = [
    "Bacano parce, con gusto le proceso el pedido, me regala su número de WhatsApp para confirmarle",
    "Perfecto mano, para cerrar el pedido me comparte su numerito de celular",
    "Qué buena decisión parce, solo me falta su número de WhatsApp para procesarlo",
    "De una, para apuntarle el pedido me regala su número de contacto",
    "Listo parce, ya casi, solo me da su celular y lo procesamos de una",
]


def generate_response_from_model():

    messages = [*conversation]
    last_tool_result = None
    ultimo_mensaje = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )
    
    if tiene_intencion_compra(ultimo_mensaje) and not tiene_telefono(ultimo_mensaje):
        response = random.choice(PEDIR_TELEFONO_RESPUESTAS)
        response = preparar_para_tts(response)
        response = re.sub(r'([a-záéíóúüñ])([A-ZÁÉÍÓÚÜÑ])', r'\1 \2', response)
        words = response.split()
        for i in range(3, len(words) - 1, 4):
            if not words[i].endswith(','):
                words[i] += ','
        print(f"RESULT_TEXT={' '.join(words)}")
        return

    for _ in range(3):

        response = call_model(messages)
        try:
            tool_call = json.loads(response)
        except json.JSONDecodeError:
            tool_call = None

        if isinstance(tool_call, dict) and "tool" in tool_call:
            tool_result = call_tool(tool_call["tool"], tool_call)
            if tool_call["tool"] == "get_products":
                last_tool_result = tool_result
                
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"[TOOL RESULT]: {tool_result}"})
            continue

        break

    response = re.sub(r'\b\d+\.\s*', '', response)
    response = re.sub(r'\n+', ' ', response)
    response = re.sub(r'\s+', ' ', response).strip()

    if last_tool_result:
        response = fix_prices_in_response(response, last_tool_result)

    response = preparar_para_tts(response)
    PROTEGER = {"WhatsApp": "WHATSAPP_PLACEHOLDER"}
    for word, placeholder in PROTEGER.items():
        response = response.replace(word, placeholder)

    response = re.sub(r'([a-záéíóúüñ])([A-ZÁÉÍÓÚÜÑ])', r'\1 \2', response)

    for placeholder, word in {v: k for k, v in PROTEGER.items()}.items():
        response = response.replace(placeholder, word)

    words = response.split()
    for i in range(3, len(words) - 1, 4):
        if not words[i].endswith(','):
            words[i] += ','

    response = ' '.join(words)
    print(f"RESULT_TEXT={response}")


if __name__ == "__main__":
    generate_response_from_model()