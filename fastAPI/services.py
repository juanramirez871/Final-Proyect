import json
from typing import Dict, List, Optional, Any
from mlx_lm import load, generate
from config import MODEL_PATH, DEFAULT_MAX_TOKENS, SYSTEM_PROMPT, embed_model, collection
from mlx_lm.sample_utils import make_sampler
import datetime

model, tokenizer = load(str(MODEL_PATH))

def _build_messages(
    user_prompt: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    if history:
        for m in history:
            role = m.get("role", "user")
            text = m.get("text", "")
            if not text:
                continue
            if role not in ("user", "assistant", "system"):
                role = "user"
            messages.append({"role": role, "content": text})

    messages.append({"role": "user", "content": user_prompt})
    return messages


def generate_response_from_model(
    prompt: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    history: Optional[List[Dict[str, str]]] = None,
    top_k: int = 3,
) -> Dict[str, str]:
    try:
        messages = _build_messages(prompt, history=history)

        print("\n--- INICIO RAZONAMIENTO ---")
        print("Messages:", messages)
        
        chat_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        response = generate(
            model,
            tokenizer,
            prompt=chat_prompt,
            max_tokens=max_tokens,
            verbose=False,
        )

        text = response.strip()
        print(f"Model output: {text}")

        try:
            tool_call = json.loads(text)
            if isinstance(tool_call, dict) and "tool" in tool_call:
                tool_name = tool_call.get("tool")
                
                if tool_name == "get_products":
                    query_tool = tool_call.get("query", "")
                    print(f"Tool call detectado: get_products('{query_tool}')")
                    
                    knowledge = get_internal_knowledge(query_tool, top_k=top_k)
                    print(f"Conocimiento obtenido: {knowledge[:100]}...")

                    messages.append({"role": "assistant", "content": text})
                    messages.append({"role": "system", "content": f"RESULTADO DE LA HERRAMIENTA:\n{knowledge}"})

                elif tool_name == "create_order":
                    print(f"Tool call detectado: create_order")
                    
                    order_status = create_order(
                        product_name=tool_call.get("product_name"),
                        quantity=tool_call.get("quantity"),
                        price=tool_call.get("price"),
                        customer_phone=tool_call.get("customer_phone")
                    )
                    
                    messages.append({"role": "assistant", "content": text})
                    messages.append({"role": "system", "content": f"RESULTADO DE LA HERRAMIENTA:\n{order_status}"})

                chat_prompt_final = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                
                final_response = generate(
                    model,
                    tokenizer,
                    prompt=chat_prompt_final,
                    max_tokens=max_tokens,
                    verbose=False,
                )
                text = final_response.strip()
                print(f"Respuesta final: {text}")
                
        except (json.JSONDecodeError, TypeError):
            pass

        print("--- FIN RAZONAMIENTO ---\n")
        return {"response": text, "status": "success"}

    except Exception as e:
        return {
            "response": "",
            "status": "error",
            "error": str(e),
        }
        

def get_internal_knowledge(query: str, top_k: int) -> str:
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
            
    return knowledge_text


def create_order(product_name: str, quantity: int, price: float, customer_phone: str) -> str:
    try:
        file_path = "database/orders.json"
        try:
            with open(file_path, "r") as f:
                orders = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            orders = []

        new_order = {
            "id": len(orders) + 1,
            "product_name": product_name,
            "description": f"Orden para {product_name}",
            "price": price * quantity,
            "customer_phone": customer_phone,
            "quantity": quantity,
            "date": datetime.date.today().isoformat()
        }

        orders.append(new_order)

        with open(file_path, "w") as f:
            json.dump(orders, f, indent=4)
            
        return f"¡Listo, parce! La orden para {quantity} de '{product_name}' ha sido creada con éxito."
    
    except Exception as e:
        return f"Hubo un error al crear la orden: {str(e)}"