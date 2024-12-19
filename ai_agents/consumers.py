import json
import logging

from channels.generic.http import AsyncHttpConsumer
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.layers import get_channel_layer
from django.utils.text import slugify
from llama_index.core.base.llms.types import ChatMessage

from ai_agents.agents import RecommendationAgent
from ai_agents.serializers import ChatRequestSerializer

log = logging.getLogger(__name__)


def process_message(message_json, agent) -> str:
    """
    Validate the message, update the agent if necessary
    """
    text_data_json = json.loads(message_json)
    serializer = ChatRequestSerializer(data=text_data_json)
    serializer.is_valid(raise_exception=True)
    message_text = serializer.validated_data.pop("message", "")
    clear_history = serializer.validated_data.pop("clear_history", False)
    temperature = serializer.validated_data.pop("temperature", None)
    instructions = serializer.validated_data.pop("instructions", None)
    model = serializer.validated_data.pop("model", None)

    if clear_history:
        agent.clear_chat_history()
    if model:
        agent.agent.agent_worker._llm.model = model  # noqa: SLF001
    if temperature:
        agent.agent.agent_worker._llm.temperature = temperature  # noqa: SLF001
    if instructions:
        agent.agent.agent_worker.prefix_messages = [
            ChatMessage(content=instructions, role="system")
        ]
    return message_text


class RecommendationAgentWSConsumer(AsyncWebsocketConsumer):
    """
    Async websocket consumer for the recommendation agent.
    """

    async def connect(self):
        """Connect to the websocket and initialize the AI agent."""
        user = self.scope.get("user", None)
        session = self.scope.get("session", None)

        if user and user.username:
            self.user_id = user.username
        elif session:
            if not session.session_key:
                session.save()
            self.user_id = session.session_key
        else:
            self.user_id = None

        self.agent = RecommendationAgent(self.user_id)
        await super().connect()

    async def receive(self, text_data: str) -> str:
        """Send the message to the AI agent and return its response."""

        try:
            message_text = process_message(text_data, self.agent)

            for chunk in self.agent.get_completion(message_text):
                await self.send(text_data=chunk)
        except:  # noqa: E722
            log.exception("Error in RecommendationAgentConsumer")
        finally:
            # This is a bit hacky, but it works for now
            await self.send(text_data="!endResponse")


class RecommendationAgentSSEConsumer(AsyncHttpConsumer):
    async def handle(self, message: str):
        user = self.scope.get("user", None)
        session = self.scope.get("session", None)

        if user and user.username and user.username != "AnonymousUser":
            self.user_id = user.username
        elif session:
            if not session.session_key:
                session.save()
            self.user_id = slugify(session.session_key)[:100]
        else:
            log.info("Anon user, no session")
            self.user_id = "Anonymous"

        agent = RecommendationAgent(self.user_id)

        self.channel_layer = get_channel_layer()
        self.room_name = "recommendation_bot"
        self.room_group_name = f"recommendation_bot_{self.user_id}"
        await self.channel_layer.group_add(
            f"recommendation_bot_{self.user_id}", self.channel_name
        )

        await self.send_headers(
            headers=[
                (b"Cache-Control", b"no-cache"),
                (
                    b"Content-Type",
                    b"text/event-stream",
                ),
                (
                    b"Transfer-Encoding",
                    b"chunked",
                ),
                (b"Connection", b"keep-alive"),
            ]
        )
        # Headers are only sent after the first body event.
        # Set "more_body" to tell the interface server to not
        # finish the response yet:
        payload = "\nevent: ping", "data: null\n\n\n"
        await self.send_body(payload.encode("utf-8"), more_body=True)

        try:
            message_text = process_message(message, agent)

            for chunk in agent.get_completion(message_text):
                await self.send_event(event=chunk)
        except:  # noqa: E722
            log.exception("Error in RecommendationAgentConsumer")
        finally:
            self.disconnect()

    async def disconnect(self):
        await self.channel_layer.group_discard(f"sse_{self.user_id}", self.channel_name)

    async def send_event(self, event: str):
        # Send response event
        log.info(event)
        data = f"event: agent_response\ndata: {event}\n\n"
        await self.send_body(data.encode("utf-8"), more_body=True)

    async def http_request(self, message):
        """
        Receives an SSE request and holds the connection open
        until the client or server chooses to disconnect.
        """
        try:
            await self.handle(message.get("body"))
        finally:
            pass
