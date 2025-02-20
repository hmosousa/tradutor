from src.models import LLama2


class TestLLama2:
    def test_translate(self):
        model = LLama2(checkpoint="llama2_13b_fft")
        text = "Hello, world!"
        translation = model.translate(text)
        assert "Olá, mundo!" in translation

    def test_end_of_sentence(self):
        model = LLama2(checkpoint="llama2_13b_fft")
        text = "Many of these tourists own land in the region."
        translation = model.translate(text)
        assert "</s>" not in translation

    def test_to_hf(self):
        model = LLama2(checkpoint="llama2_13b_fft")
        hf_model, hf_tokenizer = model.to_hf()
        assert hf_model is not None
        assert hf_tokenizer is not None
