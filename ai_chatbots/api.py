"""AI-specific functions for ai_agents."""

from typing import Optional

from django.conf import settings
from llama_index.core.agent import AgentRunner
from llama_index.core.llms.llm import LLM

from ai_chatbots.constants import AgentClassEnum, LLMClassEnum
from ai_chatbots.proxies import AIProxy


def get_llm(model_name: Optional[str] = None, proxy: Optional[AIProxy] = None) -> LLM:
    """
    Get the LLM from the given model name,
    incorporating a proxy if passed.

    Args:
        model_name: The name of the model
        proxy: The proxy to use

    Returns:
        The LLM

    """
    if not model_name:
        model_name = settings.AI_MODEL
    try:
        llm_class = LLMClassEnum[settings.AI_PROVIDER].value
        return llm_class(
            model=model_name,
            **(proxy.get_api_kwargs() if proxy else {}),
            additional_kwargs=(proxy.get_additional_kwargs() if proxy else {}),
        )
    except KeyError as ke:
        msg = f"{settings.AI_PROVIDER} not supported"
        raise NotImplementedError(msg) from ke
    except Exception as ex:
        msg = f"Error instantiating LLM: {model_name}"
        raise ValueError(msg) from ex


def get_agent() -> AgentRunner:
    """Get the appropriate chatbot agent for the AI provider"""
    try:
        return AgentClassEnum[settings.AI_PROVIDER].value
    except KeyError as ke:
        msg = f"{settings.AI_PROVIDER} not supported"
        raise NotImplementedError(msg) from ke
