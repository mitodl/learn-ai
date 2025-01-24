import json
import os
from pathlib import Path

import pytest

from ai_chatbots.factories import HumanMessageFactory, SystemMessageFactory


@pytest.fixture(autouse=True)
def ai_settings(settings):
    """Assign default AI settings"""
    settings.AI_PROXY = None
    settings.AI_PROXY_URL = None
    settings.OPENAI_API_KEY = "test_key"
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
    return settings


@pytest.fixture
def chat_history():
    """Return a 2-round trip chat history for testing."""
    return [
        HumanMessageFactory.create(),
        SystemMessageFactory.create(),
        HumanMessageFactory.create(),
        SystemMessageFactory.create(),
    ]


class MockAsyncIterator:
    """An async iterator for testing purposes."""

    def __init__(self, seq):
        self.iter = iter(seq)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.iter)
        except StopIteration:
            raise StopAsyncIteration from StopIteration


@pytest.fixture
def search_results():
    """Return search results for testing."""
    with Path.open("./test_json/search_results.json") as f:
        yield json.loads(f.read())
