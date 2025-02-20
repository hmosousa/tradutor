import torch
from torchtune.models.convert_weights import meta_to_tune, tune_to_hf
from torchtune.models.llama3 import llama3_8b, llama3_tokenizer, lora_llama3_8b
from transformers import AutoTokenizer, LlamaForCausalLM
from peft import PeftModel

from src.constants import MODEL_PATH, TOKENIZER_PATH
from src.models.base import TuneBaseModel


class LLama3(TuneBaseModel):
    _NAME = "llama3"

    def _load_model(self):
        model = llama3_8b()
        state_dict = self._load_state_dict()
        model.load_state_dict(state_dict, strict=True)
        model = model.to(self._device)
        return model

    def _load_state_dict(self):
        state_dict = torch.load(self._ckpt_path / "meta_model_0.pt", map_location="cpu")
        state_dict = meta_to_tune(state_dict)
        return state_dict

    def _load_tokenizer(self):
        tokenizer = llama3_tokenizer(str(self._tkn_path))
        return tokenizer

    def _encode_messages(self, messages: str):
        input_ids, _ = self._tokenizer.tokenize_messages(messages)
        # drop the last <|eot_id|> token (with id 128009) so that the model generates the translation
        input_ids = input_ids[:-2]
        return input_ids

    def to_hf(self):
        hf_state_dict = tune_to_hf(self._model.state_dict(), num_heads=32, num_kv_heads=8, dim=4096)
        hf_model = LlamaForCausalLM.from_pretrained(MODEL_PATH["llama3"].parent)
        hf_model.load_state_dict(hf_state_dict, strict=True)
        hf_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH["llama3"].parent.parent)
        return hf_model, hf_tokenizer


class LoraLLama3(LLama3):
    _NAME = "llama3_lora"

    def _load_model(self):
        model = lora_llama3_8b(
            lora_attn_modules=self._lora_attn_modules,
            apply_lora_to_mlp=self._apply_lora_to_mlp,
            apply_lora_to_output=self._apply_lora_to_output,
            lora_rank=self._lora_rank,
            lora_alpha=self._lora_alpha,
        )

        state_dict = self._load_state_dict()
        model.load_state_dict(state_dict, strict=True)
        model = model.to(self._device)
        return model

    def _load_state_dict(self):
        state_dict = torch.load(self._ckpt_path / "meta_model_0.pt", map_location="cpu")
        state_dict = meta_to_tune(state_dict)

        adapter_state_dict = torch.load(self._ckpt_path / "adapter_0.pt", map_location="cpu")
        state_dict.update(adapter_state_dict)
        return state_dict

    def _load_tokenizer(self):
        tokenizer = llama3_tokenizer(str(self._tkn_path))
        return tokenizer

    def to_hf(self):
        hf_model = LlamaForCausalLM.from_pretrained(MODEL_PATH["llama3"].parent)
        peft_model = PeftModel.from_pretrained(hf_model, self._ckpt_path)
        hf_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH["llama3_tokenizer"].parent)
        return peft_model, hf_tokenizer
