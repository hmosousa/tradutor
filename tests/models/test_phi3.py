from src.models import LoraPhi3, Phi3


class TestPhi3:
    def test_tokenizer(self):
        model = Phi3(checkpoint="phi3", device="cpu")
        assert model.tokenizer is not None

    def test_stop_tokens(self):
        model = Phi3(checkpoint="phi3", device="cpu")
        assert model.stop_tokens == [13, 32000]

    def test_translate(self):
        model = Phi3(checkpoint="phi3", device="cpu")
        text = "Hello, world!"
        translation = model.translate(text)
        assert "Olá, mundo!" in translation

    def test_to_hf(self):
        model = Phi3(checkpoint="phi3_clean")
        hf_model, hf_tokenizer = model.to_hf()
        assert hf_model is not None
        assert hf_tokenizer is not None


class TestLoraPhi3:
    def test_name(self):
        model = LoraPhi3(
            checkpoint="phi3_lora",
            lora_attn_modules=["q_proj", "v_proj", "k_proj", "ouput_proj"],
            apply_lora_to_mlp=True,
            apply_lora_to_output=True,
            lora_rank=128,
            lora_alpha=256,
            device="cpu",
        )
        assert model.name == "phi3_lora"

    def test_tokenizer(self):
        model = LoraPhi3(
            checkpoint="phi3_lora",
            device="cpu",
            lora_attn_modules=["q_proj", "v_proj", "k_proj", "ouput_proj"],
            apply_lora_to_mlp=True,
            apply_lora_to_output=True,
            lora_rank=128,
            lora_alpha=256,
        )
        assert model.tokenizer is not None

    def test_translate(self):
        model = LoraPhi3(
            checkpoint="phi3_lora",
            device="cpu",
            lora_attn_modules=["q_proj", "v_proj", "k_proj", "ouput_proj"],
            apply_lora_to_mlp=True,
            apply_lora_to_output=True,
            lora_rank=128,
            lora_alpha=256,
        )
        text = "Hello, world!"
        translation = model.translate(text)
        assert isinstance(translation, str)

    def test_translate_empty(self):
        model = LoraPhi3(
            checkpoint="phi3_lora",
            device="cpu",
            lora_attn_modules=["q_proj", "v_proj", "k_proj", "ouput_proj"],
            apply_lora_to_mlp=True,
            apply_lora_to_output=True,
            lora_rank=128,
            lora_alpha=256,
        )
        text = "This was followed by the club's first major success in 1935."
        translation = model.translate(text)
        assert isinstance(translation, str)

    def test_to_hf(self):
        model = LoraPhi3(
            checkpoint="phi3_lora_raw",
            lora_attn_modules=["q_proj", "v_proj", "k_proj", "ouput_proj"],
            apply_lora_to_mlp=True,
            apply_lora_to_output=True,
            lora_rank=128,
            lora_alpha=256,
        )
        hf_model, hf_tokenizer = model.to_hf()
        assert hf_model is not None
        assert hf_tokenizer is not None
