import subprocess
import json
from typing import Dict
from config import MODEL_PATH, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE


def generate_response_from_model(prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> Dict[str, str]:
    try:
        # Use the recommended invocation form to avoid deprecation warnings
        cmd = [
            "python", "-m", "mlx_lm", "generate",
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
        
        raw_output = result.stdout or ""

        def clean_model_output(text: str) -> str:
            # Remove common warnings about invocation and other noise, then strip trailing metrics
            lines = text.splitlines()
            # filter out deprecation/invocation hint lines
            filtered = []
            for ln in lines:
                s = ln.strip()
                if not s:
                    filtered.append(ln)
                    continue
                # drop common deprecation/invocation hints
                if (
                    "deprecated" in s.lower()
                    or s.startswith("Calling `python -m mlx_lm.generate")
                    or s.startswith("Use `mlx_lm.generate")
                    or s.startswith("Use `python -m mlx_lm generate")
                ):
                    continue
                filtered.append(ln)

            # find separator lines composed of '=' to isolate main content
            sep_idxs = [i for i, l in enumerate(filtered) if set(l.strip()) == {"="} and len(l.strip()) >= 3]
            if len(sep_idxs) >= 2:
                content_lines = filtered[sep_idxs[0] + 1: sep_idxs[1]]
            elif len(sep_idxs) == 1:
                # take content after first separator until metrics
                start = sep_idxs[0] + 1
                end = len(filtered)
                for i in range(start, len(filtered)):
                    if filtered[i].strip().startswith(("Prompt:", "Generation:", "Peak memory:")):
                        end = i
                        break
                content_lines = filtered[start:end]
            else:
                # no separators: drop trailing metric lines
                end = len(filtered)
                for i, ln in enumerate(filtered):
                    if ln.strip().startswith(("Prompt:", "Generation:", "Peak memory:")):
                        end = i
                        break
                content_lines = filtered[:end]

            content = "\n".join(content_lines).strip()
            return content

        response_text = clean_model_output(raw_output)

        return {"response": response_text, "status": "success"}
        
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
