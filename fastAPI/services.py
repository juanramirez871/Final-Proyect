import subprocess
import json
from typing import Dict
from config import MODEL_PATH, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE


def generate_response_from_model(prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> Dict[str, str]:
    try:
        cmd = [
            "python", "-m", "mlx_lm.generate",
            "--model", str(MODEL_PATH),
            "--prompt", prompt,
            "--max-tokens", str(max_tokens)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode != 0:
            raise RuntimeError(
                f"Error en la ejecución del modelo: {result.stderr}"
            )
        
        response_text = result.stdout.strip()
        return {
            "response": response_text,
            "status": "success"
        }
        
    except subprocess.TimeoutExpired:
        return {
            "response": "",
            "status": "timeout",
            "error": "La ejecución del modelo excedió el tiempo límite"
        }
    except Exception as e:
        return {
            "response": "",
            "status": "error",
            "error": str(e)
        }
