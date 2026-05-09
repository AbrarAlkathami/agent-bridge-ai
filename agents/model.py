from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from core.config import settings


class ModelProvider:
    def __init__(self, model_name: str, provider: str, api_key: str) -> None:
        self._model_name = model_name
        self._provider = provider
        self._api_key = api_key
        self._model: BaseChatModel | None = None

    def get_model(self) -> BaseChatModel:
        if self._model is None:
            self._model = init_chat_model(
                model=self._model_name,
                model_provider=self._provider,
                openai_api_key=self._api_key,
            )
        return self._model


model_provider = ModelProvider(
    model_name=settings.model_name,
    provider="openai",
    api_key=settings.openai_api_key,
)
