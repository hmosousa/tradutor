import json
from abc import ABC
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torchtune.data import Message
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.constants import CACHE_PATH, CHECKPOINT_PATH, MODEL_PATH, TOKENIZER_PATH


class BaseModel(ABC):
    _NAME: str = None

    def __call__(self, dataset, **kwargs) -> Any:
        return self.translate_dataset(dataset, **kwargs)

    @classmethod
    def translate(self, text: str, **kwargs) -> str:
        raise NotImplementedError

    def translate_dataset(self, dataset, **kwargs) -> str:
        translated = []
        cache = self._check_cache(dataset.name)
        for idx, source, _ in tqdm(dataset):
            idx = str(idx)
            if cache and idx in cache:
                translated.append(cache[idx])
                continue
            translation = self.translate(source, **kwargs)
            translated.append(translation)
            cache[idx] = translation
            self._save_cache(dataset.name, cache)
        return translated

    def _check_cache(self, dataset_name: str) -> bool:
        cache_fp = self._cache_filepath(dataset_name)
        if cache_fp.exists():
            return json.load(cache_fp.open())
        return {}

    def _save_cache(self, dataset_name: str, data: Any) -> None:
        if not CACHE_PATH.exists():
            CACHE_PATH.mkdir(parents=True, exist_ok=True)
        cache_fp = self._cache_filepath(dataset_name)
        json.dump(data, cache_fp.open("w", encoding="utf-8"), indent=4, ensure_ascii=False)

    def _cache_filepath(self, dataset_name: str):
        if hasattr(self, "_checkpoint"):
            return CACHE_PATH / dataset_name / f"{self._checkpoint}.json"
        else:
            return CACHE_PATH / dataset_name / f"{self.name}.json"

    @property
    def name(self):
        return self._NAME


class TuneBaseModel(BaseModel):
    _NAME = None

    def __init__(
        self,
        checkpoint: str,
        device: Optional[str] = None,
        **kwargs,
    ):
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = device

        self._checkpoint = checkpoint
        self._ckpt_path = CHECKPOINT_PATH / checkpoint
        self._tkn_path = TOKENIZER_PATH[self._NAME]
        for k, v in kwargs.items():
            setattr(self, f"_{k}", v)

        self._model = self._load_model()
        self._tokenizer = self._load_tokenizer()

        self._stop_tokens = self._get_stop_tokens()

    def _get_stop_tokens(self):
        linebreak_token_id = self._tokenizer.encode("\n", add_bos=False, add_eos=False)[-1]
        return [linebreak_token_id] + self._tokenizer.stop_tokens

    @property
    def stop_tokens(self):
        return self._stop_tokens

    @torch.no_grad()
    def _inference(
        self, prompt: str, top_p: Optional[float] = None, top_k: Optional[int] = None, penalty: Optional[float] = None
    ):
        messages = [
            Message(
                role="system",
                content="You are a translator from English to European Portuguese",
            ),
            Message(
                role="user",
                content=f"Translate this text from English to European Portuguese: {prompt}",
            ),
            Message(role="assistant", content=""),
        ]

        input_ids = self._encode_messages(messages)
        n_input_tokens = len(input_ids)
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(self._device)

        max_new_tokens = self._compute_max_new_tokens(input_ids)
        for _ in range(max_new_tokens):
            logits = self._model(input_ids)[:, -1]  # to select the last one

            if penalty is not None:
                score = torch.gather(logits, 1, input_ids)
                score = torch.where(score < 0, score * penalty, score / penalty)
                logits = logits.scatter(1, input_ids, score)

            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k, dim=1)[0][..., -1, None]
                logits[indices_to_remove] = -float("Inf")  # so they disappear in softmax

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                sorted_logits[sorted_indices_to_remove] = -float("Inf")
                logits = torch.gather(sorted_logits, 1, sorted_indices.argsort(-1))
                new_token = torch.multinomial(F.softmax(logits, -1), 1)
                input_ids = torch.cat([input_ids, new_token], dim=-1)

            new_token = logits.argmax(dim=-1)
            if new_token.item() in self._stop_tokens:
                break

            input_ids = torch.cat([input_ids, new_token.unsqueeze(0)], dim=-1)
        output_ids = input_ids[0, n_input_tokens:]
        answer = self._tokenizer.decode(output_ids.tolist())
        return answer.strip()

    def translate(
        self, text: str, top_p: Optional[float] = None, top_k: Optional[int] = None, penalty: Optional[float] = None
    ) -> str:
        answer = self._inference(text, top_p, top_k, penalty)
        return answer

    def _compute_max_new_tokens(self, input_ids: torch.Tensor):
        """Current heuristic: double the input size, but not exceeding the model size."""
        input_size = input_ids.size(1)
        model_size = self._model.max_seq_len
        return min(input_size * 2, model_size - input_size)

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def tokenizer_path(self):
        return self._tkn_path

    def _encode_messages(self, messages: str):
        raise NotImplementedError

    def push2hf(self, repo_name: str):
        hf_model, hf_tokenizer = self.to_hf()
        hf_model.push_to_hub(repo_name)
        hf_tokenizer.push_to_hub(repo_name, commit_message="Add tokenizer")

    def save_local_hf(self, folder_name: str):
        hf_model, hf_tokenizer = self.to_hf()
        hf_model.save_pretrained(folder_name)
        hf_tokenizer.save_pretrained(folder_name)


class HuggingFaceBaseModel(BaseModel):
    _NAME = None

    def __init__(self, model_path=None, checkpoint=None) -> None:
        if model_path is None:
            model_path = MODEL_PATH[self._NAME]
        if checkpoint is not None:
            model_path = CHECKPOINT_PATH / checkpoint

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16,
        )
        self._eos_token_ids = self._get_stop_tokens()

    def _chat_template(self, text: str):
        raise NotImplementedError

    def translate(self, text: str) -> str:
        chat = self._chat_template(text)

        input_ids = self._tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            max_length=1024,
        ).to(self._model.device)

        input_ids = input_ids.to(self._model.device)
        output_ids = self._model.generate(
            input_ids,
            max_length=1024,
            num_return_sequences=1,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        generated_ids = output_ids[0, input_ids.shape[1] :]
        translation = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        return translation.strip()

    def _get_stop_tokens(self):
        line_break_token_id = self._tokenizer.encode("\n", add_special_tokens=False)[0]
        double_line_break_token_id = self._tokenizer.encode("\n\n", add_special_tokens=False)[0]
        return [self._tokenizer.eos_token_id, line_break_token_id, double_line_break_token_id]

    def translate_dataset(self, dataset, **kwargs) -> str:
        translated = []
        cache = self._check_cache(dataset.name)
        for idx, source, _ in tqdm(dataset):
            idx = str(idx)
            if cache and idx in cache:
                translated.append(cache[idx])
                continue
            translation = self.translate(source, **kwargs)
            translated.append(translation)
            cache[idx] = translation
            self._save_cache(dataset.name, cache)
        return translated
