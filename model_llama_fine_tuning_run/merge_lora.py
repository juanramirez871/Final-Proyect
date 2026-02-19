import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import login

BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LORA_PATH = "../model_llama_fine_tuning/llama_colombiano_LoRA"
OUTPUT_PATH = "./llama_colombiano_merged"

login("API_KEY")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="cpu"
)

model = PeftModel.from_pretrained(model, LORA_PATH)
model = model.merge_and_unload()
model.save_pretrained(OUTPUT_PATH)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.save_pretrained(OUTPUT_PATH)