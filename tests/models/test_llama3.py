from src.models import LLama3, LoraLLama3


class TestLLama3:
    def test_translate(self):
        model = LLama3(checkpoint="llama3")
        text = "Hello, world!"
        translation = model.translate(text)
        assert "Olá, mundo!" in translation

    def test_stop_tokens(self):
        model = LLama3(checkpoint="llama3")
        assert model.stop_tokens == [198, 128001, 128009]

    def test_end_of_sentence(self):
        model = LLama3(checkpoint="llama3")
        text = "Many of these tourists own land in the region."
        translation = model.translate(text)
        assert "<|eot_id|>" not in translation

    def test_to_hf(self):
        model = LLama3(checkpoint="llama3_clean_10k")
        hf_model, hf_tokenizer = model.to_hf()
        assert hf_model is not None
        assert hf_tokenizer is not None


class TestLoraLLama3:
    def test_translate(self):
        model = LoraLLama3(
            checkpoint="llama3_lora",
            device="cpu",
            lora_attn_modules=["q_proj", "v_proj", "k_proj", "ouput_proj"],
            apply_lora_to_mlp=True,
            apply_lora_to_output=True,
            lora_rank=64,
            lora_alpha=128,
        )
        text = "Hello, world!"
        translation = model.translate(text)
        assert "Olá, mundo!" in translation

    def test_stop_tokens(self):
        model = LoraLLama3(
            checkpoint="llama3_lora",
            device="cpu",
            lora_attn_modules=["q_proj", "v_proj", "k_proj", "ouput_proj"],
            apply_lora_to_mlp=True,
            apply_lora_to_output=True,
            lora_rank=64,
            lora_alpha=128,
        )
        assert model.stop_tokens == [198, 128001, 128009]

    def test_end_of_sentence(self):
        model = LoraLLama3(
            checkpoint="llama3_lora",
            device="cpu",
            lora_attn_modules=["q_proj", "v_proj", "k_proj", "ouput_proj"],
            apply_lora_to_mlp=True,
            apply_lora_to_output=True,
            lora_rank=64,
            lora_alpha=128,
        )
        text = "Many of these tourists own land in the region."
        translation = model.translate(text)
        assert "<|eot_id|>" not in translation
