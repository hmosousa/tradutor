import torch
from torchtune.models.phi3 import lora_phi3_mini, phi3_hf_to_tune, phi3_mini, phi3_mini_tokenizer, phi3_tune_to_hf
from transformers import AutoTokenizer, Phi3ForCausalLM
from peft import LoraConfig, PeftModel
from torchtune.modules.peft.peft_utils import get_lora_module_names

from src.constants import MODEL_PATH, TOKENIZER_PATH
from src.models.base import TuneBaseModel


class Phi3(TuneBaseModel):
    _NAME = "phi3"

    def _load_model(self):
        model = phi3_mini()
        state_dict = self._load_state_dict()
        model.load_state_dict(state_dict, strict=True)
        model = model.to(self._device)
        return model.eval()

    def _load_state_dict(self):
        state_dict = {}
        for filepath in self._ckpt_path.glob("*.pt"):
            state_dict.update(torch.load(filepath, map_location="cpu"))
        state_dict = phi3_hf_to_tune(state_dict)
        return state_dict

    def _load_tokenizer(self):
        tokenizer = phi3_mini_tokenizer(str(self._tkn_path))
        return tokenizer

    def _encode_messages(self, messages: str):
        input_ids, _ = self._tokenizer.tokenize_messages(messages)
        # drop the last <|end|> token (with id 32007) so that the model generates the translation
        input_ids = input_ids[:-3]
        return input_ids

    def to_hf(self):
        hf_state_dict = phi3_tune_to_hf(self._model.state_dict())
        hf_model = Phi3ForCausalLM.from_pretrained(MODEL_PATH["phi3"])
        hf_model.load_state_dict(hf_state_dict, strict=True)
        hf_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH["phi3_tokenizer"].parent)
        return hf_model, hf_tokenizer


class LoraPhi3(Phi3):
    _NAME = "phi3_lora"

    def _load_model(self):
        model = lora_phi3_mini(
            lora_attn_modules=self._lora_attn_modules,
            apply_lora_to_mlp=self._apply_lora_to_mlp,
            apply_lora_to_output=self._apply_lora_to_output,
            lora_rank=self._lora_rank,
            lora_alpha=self._lora_alpha,
        )
        state_dict = self._load_state_dict()
        model.load_state_dict(state_dict, strict=True)
        model = model.to(self._device)
        return model.eval()

    def _load_state_dict(self):
        state_dict = {}
        for filepath in self._ckpt_path.glob("hf*.pt"):
            state_dict.update(torch.load(filepath, map_location="cpu"))
        state_dict = phi3_hf_to_tune(state_dict)

        adapter_state_dict = torch.load(self._ckpt_path / "adapter_0.pt", map_location="cpu")
        state_dict.update(adapter_state_dict)
        return state_dict

    def to_hf(self):
        hf_model = Phi3ForCausalLM.from_pretrained(MODEL_PATH["phi3"])
        target_modules = get_lora_module_names(
            self._lora_attn_modules,
            self._apply_lora_to_mlp,
            self._apply_lora_to_output,
        )
        adapter_config = LoraConfig(r=self._lora_rank, lora_alpha=self._lora_alpha, target_modules=target_modules)
        peft_model = PeftModel.from_pretrained(hf_model, self._ckpt_path, config=adapter_config)
        hf_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH["phi3_tokenizer"].parent)
        return peft_model, hf_tokenizer
