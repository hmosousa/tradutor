from src.models import LLama3, LoraLLama3, LoraPhi3, Phi3


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
