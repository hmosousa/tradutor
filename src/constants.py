import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent

load_dotenv(ROOT / ".env")
MICROSOFT_API_KEY = os.getenv("MICROSOFT_API_KEY")
MICROSOFT_LOCATION = os.getenv("MICROSOFT_LOCATION")
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
ANN_HF_TOKEN = os.getenv("ANN_HF_TOKEN")

MODELS_PATH = Path(os.getenv("MODELS_DIR"))

CACHE_PATH = ROOT / "cache"

RESULTS_PATH = ROOT / "results"

CHECKPOINT_PATH = ROOT / "checkpoints"

PAD_TOKEN = "<|pad|>"
TRANSLATE_TOKEN = "<|translate|>"


VALID_MODELS = {
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama2_13b": "meta-llama/Llama-2-13b-chat-hf",
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "gemma": "google/gemma-2b-it",
}


TOKENIZER_PATH = {
    "llama3": MODELS_PATH / "Meta-Llama-3-8B-Instruct" / "original" / "tokenizer.model",
    "llama2_13b": MODELS_PATH / "Llama-2-13b-chat-hf" / "tokenizer.model",
    "llama3_lora": MODELS_PATH / "Meta-Llama-3-8B-Instruct" / "original" / "tokenizer.model",
    "phi3": MODELS_PATH / "Phi-3-mini-4k-instruct" / "tokenizer.model",
    "phi3_lora": MODELS_PATH / "Phi-3-mini-4k-instruct" / "tokenizer.model",
    "phi3_tokenizer": MODELS_PATH / "phi3-mini-it" / "tokenizer.model",
}


MODEL_PATH = {
    "llama3": MODELS_PATH / "Meta-Llama-3-8B-Instruct" / "original",
    "llama2_13b": MODELS_PATH / "Llama-2-13b-chat-hf",
    "llama3_base": MODELS_PATH / "Meta-Llama-3-8B-Instruct",
    "llama3_lora": MODELS_PATH / "Meta-Llama-3-8B-Instruct" / "original",
    "phi3": MODELS_PATH / "Phi-3-mini-4k-instruct",
    "phi3_base": MODELS_PATH / "Phi-3-mini-4k-instruct",
    "phi3_lora": MODELS_PATH / "Phi-3-mini-4k-instruct",
}


DATASET_NAME = "u1537782/PTradutor"
