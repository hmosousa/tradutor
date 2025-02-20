from src.models.base import HuggingFaceBaseModel


class LLama2(HuggingFaceBaseModel):
    _NAME = "llama2_13b"

    def _chat_template(self, text: str) -> str:
        return [
            {"role": "system", "content": "You are a translator from English to European Portuguese"},
            {"role": "user", "content": f"Translate this text from English to European Portuguese: {text}"},
        ]
