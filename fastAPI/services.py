from typing import Dict
from mlx_lm import load, generate
from config import MODEL_PATH, DEFAULT_MAX_TOKENS, SYSTEM_PROMPT


model, tokenizer = load(str(MODEL_PATH))


def _build_messages(user_prompt: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def generate_response_from_model(prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> Dict[str, str]:
    try:
        messages = _build_messages(prompt)
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
            verbose=True,
        )

        if isinstance(response, str):
            text = response
        else:
            text = str(response)

        return {"response": text, "status": "success"}
    except Exception as e:
        return {
            "response": "",
            "status": "error",
            "error": str(e),
        }
