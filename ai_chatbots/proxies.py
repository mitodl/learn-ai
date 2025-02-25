"""AI proxy helper classes"""

import logging
from abc import ABC, abstractmethod
from urllib.parse import urljoin

import requests
from django.conf import settings
from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders

from ai_chatbots.chatbots import BaseChatbot
from ai_chatbots.constants import AI_ANONYMOUS_USER

log = logging.getLogger(__name__)


class AIProxy(ABC):
    """Abstract base helper class for an AI proxy/gateway."""

    REQUIRED_SETTINGS = []

    # if the proxy model needs to be prefixed, set this here
    PROXY_MODEL_PREFIX = ""

    def __init__(self):
        """Raise an error if required settings are missing."""
        missing_settings = [
            setting
            for setting in self.REQUIRED_SETTINGS
            if not getattr(settings, setting, None)
        ]
        if missing_settings:
            message = ",".join(missing_settings)
            raise ValueError(message)

    @abstractmethod
    def get_api_kwargs(self, **kwargs) -> dict:
        """Get the api kwargs required to connect to the proxy."""

    @abstractmethod
    def get_additional_kwargs(self, service: BaseChatbot) -> dict:
        """Get any additional kwargs that should be sent to the proxy"""

    @abstractmethod
    def create_proxy_user(self, endpoint: str) -> None:
        """Create a proxy user."""


class LiteLLMProxy(AIProxy):
    """Helper class for the Lite LLM proxy."""

    REQUIRED_SETTINGS = ("AI_PROXY_URL", "AI_PROXY_AUTH_TOKEN")
    PROXY_MODEL_PREFIX = "litellm_proxy/"

    def get_api_kwargs(
        self, base_url_key: str = "api_base", api_key_key: str = "openai_api_key"
    ) -> dict:
        """
        Get the required API kwargs to connect to the Lite LLM proxy.
        When using the ChatLiteLLM class, these kwargs should be
        "api_base" and "openai_api_key".

        Args:
            base_url_key (str): The key to pass in the proxy API URL.
            api_key_key (str): The key to pass in the proxy authentication token.

        Returns:
            dict: The proxy API kwargs.
        """

        return {
            f"{base_url_key}": settings.AI_PROXY_URL,
            f"{api_key_key}": settings.AI_PROXY_AUTH_TOKEN,
        }

    def get_additional_kwargs(self, service: BaseChatbot) -> dict:
        """
        Get additional  kwargs to send to the Lite LLM proxy, such
        as user_id and job/task identification.
        """
        return {
            "user": service.user_id,
            "store": True,
            "guardrails": ["aporia-post-guard"],
            "extra_body": {
                "metadata": {
                    "tags": [
                        f"jobID:{service.JOB_ID}",
                        f"taskName:{service.TASK_NAME}",
                    ]
                },
                "guardrails": ["aporia-post-guard"],
            },
        }

    def create_proxy_user(self, user_id: str, endpoint: str = "new") -> None:
        """
        Set the rate limit for the user by creating a LiteLLM customer account.
        Anonymous users will share the same rate limit.
        """
        if settings.AI_PROXY_URL and settings.AI_PROXY_AUTH_TOKEN:
            # Unauthenticated users will share a common budget/rate limit,
            # so multiply for some extra capacity
            multiplier = (
                settings.AI_ANON_LIMIT_MULTIPLIER if user_id == AI_ANONYMOUS_USER else 1
            )
            request_json = {
                "user_id": user_id,
                "max_parallel_requests": settings.AI_MAX_PARALLEL_REQUESTS * multiplier,
                "tpm_limit": settings.AI_TPM_LIMIT * multiplier,
                "rpm_limit": settings.AI_RPM_LIMIT * multiplier,
                "max_budget": settings.AI_MAX_BUDGET * multiplier,
                "budget_duration": settings.AI_BUDGET_DURATION,
            }
            headers = {"Authorization": f"Bearer {settings.AI_PROXY_AUTH_TOKEN}"}
            try:
                response = requests.post(
                    urljoin(settings.AI_PROXY_URL, f"/customer/{endpoint}"),
                    json=request_json,
                    timeout=settings.REQUESTS_TIMEOUT,
                    headers=headers,
                )
                response.raise_for_status()
            except Exception as error:
                if "duplicate key value violates unique constraint" in str(error):
                    """
                    Try updating the LiteLLM customer account if it already exists.
                    Unfortunately, LiteLLM seems to have a bug that prevents
                    updates to the customer's max_budget:
                    https://github.com/BerriAI/litellm/issues/6920

                    We could create LiteLLM internal user accounts instead, but that
                    would require saving and using the LiteLLM keys generated per user.
                    """
                    self.create_proxy_user(user_id=user_id, endpoint="update")
                else:
                    log.exception("Error creating/updating proxy customer account")


class PortkeyProxy(AIProxy):
    """Helper class for the Portkey proxy."""

    REQUIRED_SETTINGS = ("AI_PORTKEY_API_KEY", "AI_PORTKEY_VIRTUAL_KEY_OPENAI")
    PROXY_MODEL_PREFIX = ""

    def get_api_kwargs(
        self, base_url_key: str = "api_base", api_key_key: str = "api_key"
    ) -> dict:
        """
        Get the required API kwargs to connect to the Lite LLM proxy.
        When using the ChatLiteLLM class, these kwargs should be
        "api_base" and "openai_api_key".

        Args:
            base_url_key (str): The key to pass in the proxy API URL.
            api_key_key (str): The key to pass in the proxy authentication token.

        Returns:
            dict: The proxy API kwargs.
        """

        return {
            f"{base_url_key}": PORTKEY_GATEWAY_URL,
            f"{api_key_key}": "dummy_value",  # has to be sent, but not used
        }

    def get_additional_kwargs(self, service: BaseChatbot) -> dict:  # noqa: ARG002
        """
        Get additional  kwargs to send to the Lite LLM proxy, such
        as user_id and job/task identification.
        """
        portkey_headers = createHeaders(
            api_key=settings.AI_PORTKEY_API_KEY,
            virtual_key=settings.AI_PORTKEY_VIRTUAL_KEY_OPENAI,
        )
        return {
            "headers": portkey_headers,
        }

    def create_proxy_user(self, endpoint: str) -> None:
        """Do not use, not necessary"""
