from src.data.eval import DSLTL, FRMT, NTREX, Publico
from src.models import (
    DeepLModel,
    GoogleModelBR,
    GoogleModelPT,
    LLama3,
    Llama3Baseline,
    LLama2,
    LoraLLama3,
    LoraPhi3,
    MicrosoftModel,
    Phi3,
    Phi3Baseline,
    ArgoModel,
    OpusMTModel,
)


def load_data(dataset):
    match dataset:
        case "publico":
            return Publico()
        case "frmt":
            return FRMT()
        case "dsl_tl":
            return DSLTL()
        case "ntrex":
            return NTREX()
        case _:
            raise ValueError("Invalid dataset")


def load_model(model: str, checkpoint: str = None):
    match model:
        case "google_br":
            return GoogleModelBR()
        case "google_pt":
            return GoogleModelPT()
        case "microsoft":
            return MicrosoftModel()
        case "deepl":
            return DeepLModel()
        case "phi3":
            return Phi3(checkpoint=checkpoint)
        case "phi3_lora":
            return LoraPhi3(
                checkpoint=checkpoint,
                lora_attn_modules=["q_proj", "v_proj", "k_proj", "ouput_proj"],
                apply_lora_to_mlp=True,
                apply_lora_to_output=True,
                lora_rank=128,
                lora_alpha=256,
            )
        case "llama3":
            return LLama3(checkpoint=checkpoint)
        case "llama3_lora":
            return LoraLLama3(
                checkpoint=checkpoint,
                lora_attn_modules=["q_proj", "v_proj", "k_proj", "ouput_proj"],
                apply_lora_to_mlp=True,
                apply_lora_to_output=True,
                lora_rank=128,
                lora_alpha=256,
            )
        case "llama2_13b":
            return LLama2(checkpoint=checkpoint)
        case "phi3_base":
            return Phi3Baseline()
        case "llama3_base":
            return Llama3Baseline()
        case "argo":
            return ArgoModel()
        case "opus_mt":
            return OpusMTModel()
        case _:
            raise ValueError("Invalid model")
