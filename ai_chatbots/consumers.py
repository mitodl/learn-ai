import json
import logging
from abc import ABC, abstractmethod
from uuid import uuid4

from channels.generic.http import AsyncHttpConsumer
from channels.layers import get_channel_layer
from django.utils.text import slugify

from ai_chatbots.chatbots import ResourceRecommendationBot, SyllabusBot
from ai_chatbots.constants import AI_THREAD_COOKIE_KEY
from ai_chatbots.serializers import ChatRequestSerializer, SyllabusChatRequestSerializer

log = logging.getLogger(__name__)


class BaseBotHttpConsumer(ABC, AsyncHttpConsumer):
    """Base HttpConsumer for chatbots"""

    ROOM_NAME = "base_bot"

    serializer_class = ChatRequestSerializer

    @abstractmethod
    def create_chatbot(self, serializer):
        """Return a bot instance"""
        raise NotImplementedError

    def process_message(
        self, message_json: str, serializer_class: type[ChatRequestSerializer]
    ) -> ChatRequestSerializer:
        """
        Validate the message and return the serializer.
        """
        text_data_json = json.loads(message_json)
        serializer = serializer_class(data=text_data_json)
        serializer.is_valid(raise_exception=True)
        return serializer

    async def prepare_response(self, serializer) -> ChatRequestSerializer:
        """Prepare the response"""
        user = self.scope.get("user", None)
        session = self.scope.get("session", None)

        clear_history = serializer.validated_data.pop("clear_history", False)
        thread_id = self.scope["cookies"].get(AI_THREAD_COOKIE_KEY)

        if clear_history or not thread_id:
            thread_id = str(uuid4())

        self.thread_id = thread_id

        cookie_thread = (
            f"{AI_THREAD_COOKIE_KEY}={self.thread_id}; Path=/; HttpOnly".encode()
        )

        if user and user.username and user.username != "AnonymousUser":
            self.user_id = user.global_id.replace("-", "_")
        elif session:
            if not session.session_key:
                session.save()
            self.user_id = slugify(session.session_key).replace("-", "_")
        else:
            log.info("Anon user, no session")
            self.user_id = "Anonymous"

        self.channel_layer = get_channel_layer()
        self.room_name = self.ROOM_NAME
        self.room_group_name = f"{self.ROOM_NAME}_{self.user_id}"[:50]
        await self.channel_layer.group_add(self.room_group_name, self.channel_name)

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
        return serializer

    def process_extra_state(self, data: dict) -> dict:  # noqa: ARG002
        """Process extra state if any"""
        return None

    async def handle(self, message: str):
        """Handle the incoming message and send the response."""
        try:
            serializer = self.process_message(message, self.serializer_class)
            await self.prepare_response(serializer)
            self.bot = self.create_chatbot(serializer)
            message_text = serializer.validated_data["message"]
            extra_state = self.process_extra_state(serializer.validated_data)
            async for chunk in self.bot.get_completion(
                message_text, extra_state=extra_state
            ):
                await self.send_chunk(chunk)
        except:  # noqa: E722
            log.exception("Error in consumer handle")
        finally:
            await self.send_chunk("", more_body=False)
            await self.disconnect()

    async def disconnect(self):
        """Discard the group when the connection is closed."""
        if hasattr(self, "channel_layer"):
            await self.channel_layer.group_discard(
                self.room_group_name, self.channel_name
            )

    async def send_chunk(self, chunk: str, *, more_body: bool = True):
        """send_chunk should call send_body with the chunk and more_body kwarg"""
        await self.send_body(body=chunk.encode("utf-8"), more_body=more_body)

    async def http_request(self, message):
        """
        Receives a request and holds the connection open
        until the client or server chooses to disconnect.
        """
        try:
            await self.handle(message.get("body"))
        finally:
            await self.disconnect()


class RecommendationBotHttpConsumer(BaseBotHttpConsumer):
    """
    Async HTTP consumer for the recommendation agent.
    """

    def create_chatbot(self, serializer: ChatRequestSerializer):
        """Return a ResourceRecommendationBot instance"""
        temperature = serializer.validated_data.pop("temperature", None)
        instructions = serializer.validated_data.pop("instructions", None)
        model = serializer.validated_data.pop("model", None)

        return ResourceRecommendationBot(
            self.user_id,
            temperature=temperature,
            instructions=instructions,
            model=model,
            thread_id=self.thread_id,
        )


class SyllabusBotHttpConsumer(BaseBotHttpConsumer):
    """
    Async HTTP consumer for the syllabus bot.
    """

    serializer_class = SyllabusChatRequestSerializer

    def create_chatbot(self, serializer: SyllabusChatRequestSerializer):
        """Return a SyllabusBot instance"""
        temperature = serializer.validated_data.pop("temperature", None)
        instructions = serializer.validated_data.pop("instructions", None)
        model = serializer.validated_data.pop("model", None)

        return SyllabusBot(
            self.user_id,
            temperature=temperature,
            instructions=instructions,
            model=model,
            thread_id=self.thread_id,
        )

    def process_extra_state(self, data: dict) -> dict:
        """Process extra state parameters if any"""
        return {
            "course_id": [data.get("course_id")],
            "collection_name": [data.get("collection_name")],
        }
