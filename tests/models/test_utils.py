from pathlib import Path

import torch
from src.models.utils import tune_to_peft_phi3_adapter_weights

ASSETS_PATH = Path(__file__).parent.parent / "assets"


def test_tune_to_peft_phi3_adapter_weights():
    adapter_state_dict = torch.load(ASSETS_PATH / "phi3_adapter_0.pt")
    converted_state_dict = tune_to_peft_phi3_adapter_weights(adapter_state_dict)
    adapter_keys = set(converted_state_dict.keys())

    expected_adapter_keys = (ASSETS_PATH / "phi3_lora_adapter_keys.txt").read_text().splitlines()
    expected_adapter_keys = set(expected_adapter_keys)

    n_intersection = len(adapter_keys.intersection(expected_adapter_keys))
    assert n_intersection == len(adapter_keys)

    len(expected_adapter_keys)
    len(adapter_keys)
    adapter_keys - expected_adapter_keys
    expected_adapter_keys
