from src.models import (
    DeepLModel,
    GoogleModelBR,
    Llama3Baseline,
    MicrosoftModel,
    Phi3Baseline,
    YandexModel,
    ArgoModel,
    OpusMTModel,
)
from src.data.eval import Publico


class TestGoogleModelBR:
    def test_init(self):
        model = GoogleModelBR()
        assert model.name == "google"

    def test_translate(self):
        model = GoogleModelBR()
        text = "Hello, world!"
        translation = model.translate(text)
        assert translation == "Olá Mundo!"

    def test_translate_dataset(self):
        model = GoogleModelBR()
        dataset = Publico("test[:5]")
        translation = model.translate_dataset(dataset)
        assert isinstance(translation, list)
        assert all(isinstance(t, str) for t in translation)
        assert len(translation) == len(dataset)


class TestMicrosoftModel:
    def test_translate(self):
        model = MicrosoftModel()
        text = "Hello, world!"
        translation = model.translate(text)
        assert translation == "Olá, mundo!"


class TestDeepLModel:
    def test_translate(self):
        model = DeepLModel()
        text = "Hello, world!"
        translation = model.translate(text)
        assert translation == "Olá, mundo!"


class TestYandexModel:
    def test_translate(self):
        model = YandexModel()
        text = "Hello, world!"
        translation = model.translate(text)
        assert translation == "Olá, mundo!"


class TestArgoModel:
    def test_translate(self):
        model = ArgoModel()
        text = "Hello, world!"
        translation = model.translate(text)
        assert translation == "Olá Mundo"


class TestOpusMTModel:
    def test_translate(self):
        model = OpusMTModel()
        text = "Hello, world!"
        translation = model.translate(text)
        assert translation == "Olá, mundo!"


class TestPhi3Baseline:
    def test_translate(self):
        model = Phi3Baseline()
        text = "Hello, world!"
        translation = model.translate(text)
        assert translation == "Olá, mundo!"


class TestLlama3Baseline:
    def test_translate(self):
        model = Llama3Baseline()
        text = "Hello, world!"
        translation = model.translate(text)
        assert translation == "Olá, mundo!"
