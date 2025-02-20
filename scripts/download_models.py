import logging

import fire
from huggingface_hub import snapshot_download
from requests.exceptions import HTTPError

from src.constants import HF_TOKEN, MODELS_PATH, VALID_MODELS

logging.basicConfig(level=logging.INFO)


def hf_download(model_name: str = "llama3") -> None:
    repo_id = VALID_MODELS.get(model_name)

    match model_name:
        case "llama3":
            allow_patterns = ["original/consolidated.00.pth", "original/tokenizer.model", "*.json", "*.safetensors"]

        case "llama2_13b":
            allow_patterns = ["*.safetensors", "*.model", "*.json"]

        case "phi3":
            allow_patterns = ["*.safetensors", "*.model", "*.json"]

        case "phi3_tokenizer":
            allow_patterns = ["*.model", "*.json"]

        case "gemma":
            allow_patterns = ["*.safetensors", "*.model", "config.json", "*.json"]

        case _:
            raise ValueError(f"Invalid model name: {model_name}. Valid models are: {VALID_MODELS.keys()}")

    local_dir = MODELS_PATH / repo_id.split("/")[-1]
    logging.info(f"Downloading model {model_name} to {local_dir}")
    try:
        snapshot_download(
            repo_id, local_dir=local_dir, local_dir_use_symlinks=False, allow_patterns=allow_patterns, token=HF_TOKEN
        )
    except HTTPError as e:
        raise e


if __name__ == "__main__":
    fire.Fire(hf_download)
