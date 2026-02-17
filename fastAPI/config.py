import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "model_llama_fine_tuning_run" / "llama_colombiano_mlx_q4"
DEFAULT_MAX_TOKENS = 100
DEFAULT_TEMPERATURE = 0.7
