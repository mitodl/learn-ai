"""System checks for the ai_chatbots app"""

from collections.abc import Iterator

import ol_llm
from django.conf import settings
from django.core.checks import Error, register

from ai_chatbots.chatbots import BaseChatbot


def _chatbot_classes(cls: type[BaseChatbot]) -> Iterator[type[BaseChatbot]]:
    for subclass in cls.__subclasses__():
        yield subclass
        yield from _chatbot_classes(subclass)


@register()
def check_default_models(app_configs, **kwargs):  # noqa: ARG001
    """
    Refuse a default model that ol-llm's catalog has retired.

    Every bot runs check_chat_model before it calls the model, so a retired
    default fails every answer. migrate runs system checks, so this fails the
    deploy instead.
    """
    errors = []
    for bot in _chatbot_classes(BaseChatbot):
        if not (bot.CALLS_THE_MODEL and hasattr(bot, "USE_CASE_NAME")):
            continue
        model = getattr(settings, bot.DEFAULT_MODEL_SETTING)
        try:
            ol_llm.check_chat_model(bot.chat_use_case(model))
        except ValueError as error:
            errors.append(
                Error(
                    f"{bot.DEFAULT_MODEL_SETTING}={model!r}: {error}",
                    obj=bot,
                    id="ai_chatbots.E001",
                )
            )
    return errors
