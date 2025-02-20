from typing import Dict

import torch
from torchtune.models.convert_weights import _TO_PEFT_KEYS, _FROM_HF, get_mapped_key


def tune_to_peft_phi3_adapter_weights(
    state_dict: Dict[str, torch.Tensor],
    num_heads: int = 32,
    num_kv_heads: int = 32,
    dim: int = 3072,
):
    converted_state_dict = {}
    full_mapping = {}
    # Rather than recreate a separate mapping for LoRA adapter weights, we just
    # re-use the _FROM_HF mapping for base model weights. We itgerate over it twice:
    # once to add mappings for LoRA A matrices and once to add mappings for LoRA B matrices.
    for k, v in _TO_PEFT_KEYS.items():
        full_mapping.update(
            {
                vv.replace(".weight", f".{k}.weight"): kk.replace(".weight", f".{v}.weight")
                for kk, vv in _FROM_HF.items()
                if vv is not None
            }
        )

    head_dim = dim // num_heads

    def _permute_lora_matrix(t, n_heads):
        rank = t.shape[-1]
        return t.view(n_heads, head_dim // 2, 2, rank).transpose(1, 2).reshape((head_dim * n_heads), rank)

    for key, value in state_dict.items():
        new_key = get_mapped_key(key, full_mapping)
        if "q_proj" in new_key and "lora_B" in new_key:
            value = _permute_lora_matrix(value, num_heads)
        elif "k_proj" in new_key and "lora_B" in new_key:
            value = _permute_lora_matrix(value, num_kv_heads)
        parts = new_key.split(".")
        parts = parts[:-1] + ["default"] + parts[-1:]
        new_key = ".".join(parts)
        converted_state_dict["base_model.model." + new_key] = value

    return converted_state_dict
