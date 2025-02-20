import argostranslate.package
import argostranslate.translate
from deep_translator import DeeplTranslator, GoogleTranslator, MicrosoftTranslator, YandexTranslator
from transformers import MarianMTModel, MarianTokenizer
from transformers import pipeline

from src.constants import DEEPL_API_KEY, MICROSOFT_API_KEY, MICROSOFT_LOCATION, YANDEX_API_KEY
from src.models.base import BaseModel, HuggingFaceBaseModel


class GoogleModelBR(BaseModel):
    _NAME = "google_br"

    def __init__(self):
        self._engine = GoogleTranslator(source="en", target="pt")

    def translate(self, text: str) -> str:
        return self._engine.translate(text)


class GoogleModelPT(BaseModel):
    _NAME = "google_pt"

    def __init__(self):
        self._engine = GoogleTranslator(source="en", target="pt-PT")

    def translate(self, text: str) -> str:
        return self._engine.translate(text)


class MicrosoftModel(BaseModel):
    _NAME = "microsoft"

    def __init__(self):
        self._engine = MicrosoftTranslator(
            source="en", target="pt", api_key=MICROSOFT_API_KEY, region=MICROSOFT_LOCATION
        )

    def translate(self, text: str) -> str:
        return self._engine.translate(text)


class YandexModel(BaseModel):
    _NAME = "yandex"

    def __init__(self):
        self._engine = YandexTranslator(source="en", target="pt", api_key=YANDEX_API_KEY)

    def translate(self, text: str) -> str:
        return self._engine.translate(text)


class ArgoModel(BaseModel):
    _NAME = "argo"

    def __init__(self):
        self._from_code = "en"
        self._to_code = "pt"

        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()
        package_to_install = next(
            filter(lambda x: x.from_code == self._from_code and x.to_code == self._to_code, available_packages)
        )
        argostranslate.package.install_from_path(package_to_install.download())

    def translate(self, text: str) -> str:
        translatedText = argostranslate.translate.translate(text, self._from_code, self._to_code)
        return translatedText


class OpusMTModel(BaseModel):
    _NAME = "opus_mt"

    def __init__(self):
        model_name = "Helsinki-NLP/opus-mt-tc-big-en-pt"
        self._pipe = pipeline("translation", model=model_name, device="cuda")

    def translate(self, text: str) -> str:
        outputs = self._pipe(text)
        return outputs[0]["translation_text"]


class DeepLModel(BaseModel):
    _NAME = "deepl"

    def __init__(self):
        self._engine = DeeplTranslator(source="en", target="pt", api_key=DEEPL_API_KEY)

    def translate(self, text: str) -> str:
        return self._engine.translate(text)


class Phi3Baseline(HuggingFaceBaseModel):
    _NAME = "phi3_base"

    def _chat_template(self, text: str) -> str:
        return [
            {"role": "user", "content": f"Translate this text from English to European Portuguese: {text}"},
        ]


class Llama3Baseline(HuggingFaceBaseModel):
    _NAME = "llama3_base"

    def _chat_template(self, text: str) -> str:
        return [
            {"role": "system", "content": "You are a translator from English to European Portuguese"},
            {"role": "user", "content": f"Translate this text from English to European Portuguese: {text}"},
        ]
