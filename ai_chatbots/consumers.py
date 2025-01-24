import json
import logging
from uuid import uuid4

from channels.generic.http import AsyncHttpConsumer
from channels.layers import get_channel_layer
from django.utils.text import slugify

from ai_chatbots.chatbots import ResourceRecommendationBot
from ai_chatbots.constants import AI_THREAD_COOKIE_KEY
from ai_chatbots.serializers import ChatRequestSerializer

log = logging.getLogger(__name__)


def process_message(message_json) -> str:
    """
    Validate the message, update the agent if necessary
    """
    text_data_json = json.loads(message_json)
    serializer = ChatRequestSerializer(data=text_data_json)
    serializer.is_valid(raise_exception=True)
    return serializer


def create_chatbot(user_id, thread_id, serializer):
    temperature = serializer.validated_data.pop("temperature", None)
    instructions = serializer.validated_data.pop("instructions", None)
    model = serializer.validated_data.pop("model", None)

    return ResourceRecommendationBot(
        user_id,
        temperature=temperature,
        instructions=instructions,
        model=model,
        thread_id=thread_id,
    )


class RecommendationBotHttpConsumer(AsyncHttpConsumer):
    """
    Async HTTP consumer for the recommendation agent.
    """

    async def handle(self, message: str):
        try:
            user = self.scope.get("user", None)
            session = self.scope.get("session", None)
            serializer = process_message(message)
            clear_history = serializer.validated_data.pop("clear_history", False)
            thread_id = self.scope["cookies"].get(AI_THREAD_COOKIE_KEY)

            if clear_history or not thread_id:
                thread_id = str(uuid4())

            cookie_thread = (
                f"{AI_THREAD_COOKIE_KEY}={thread_id}; Path=/; HttpOnly".encode()
            )

            if user and user.username and user.username != "AnonymousUser":
                self.user_id = user.username
            elif session:
                if not session.session_key:
                    session.save()
                self.user_id = slugify(session.session_key)[:100]
            else:
                log.info("Anon user, no session")
                self.user_id = "Anonymous"

            bot = create_chatbot(self.user_id, thread_id, serializer)

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
                    (
                        b"Set-Cookie",
                        cookie_thread,
                    ),
                ]
            )
            # Headers are only sent after the first body event.
            # Set "more_body" to tell the interface server to not
            # finish the response yet:
            await self.send_chunk("")

            message_text = serializer.validated_data["message"]

            for chunk in bot.get_completion(message_text):
                await self.send_chunk(chunk)
        except:  # noqa: E722
            log.exception("Error in RecommendationAgentConsumer")
        finally:
            await self.send_chunk("", more_body=False)
            await self.disconnect()

    async def disconnect(self):
        await self.channel_layer.group_discard(
            f"recommendation_bot_{self.user_id}", self.channel_name
        )

    async def send_chunk(self, chunk: str, *, more_body: bool = True):
        await self.send_body(body=chunk.encode("utf-8"), more_body=more_body)

    async def http_request(self, message):
        """
        Receives a request and holds the connection open
        until the client or server chooses to disconnect.
        """
        try:
            await self.handle(message.get("body"))
        finally:
            pass
