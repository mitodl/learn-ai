import json
import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from uuid import uuid4

from channels.generic.http import AsyncHttpConsumer
from channels.layers import get_channel_layer
from django.utils.text import slugify
from langgraph.checkpoint.base import BaseCheckpointSaver
from rest_framework.exceptions import ValidationError
from rest_framework.status import HTTP_200_OK
from ai_chatbots.chatbots import ResourceRecommendationBot, SyllabusBot
from ai_chatbots.checkpointers import AsyncDjangoSaver
from ai_chatbots.constants import AI_THREAD_COOKIE_KEY, AI_THREADS_ANONYMOUS_COOKIE_KEY
from ai_chatbots.models import UserChatSession
from ai_chatbots.serializers import ChatRequestSerializer, SyllabusChatRequestSerializer
from users.models import User

log = logging.getLogger(__name__)


class BaseBotHttpConsumer(ABC, AsyncHttpConsumer):
    """Base HttpConsumer for chatbots"""

    # Each bot consumer should define a unique ROOM_NAME
    ROOM_NAME = None

    serializer_class = ChatRequestSerializer
    headers_sent = False

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
        serializer = serializer_class(
            data=text_data_json,
            context={"user": self.scope.get("user", None)}
        )
        serializer.is_valid(raise_exception=True)
        return serializer

    async def assign_thread_cookies(
        self,
        user: User,
        *,
        clear_history: Optional[bool] = False,
        thread_id: Optional[str] = None,
    ) -> tuple[str, list[str]]:
        """
        Extract and update separate cookie values for logged in vs anonymous users.
        Each chatbot should have its own thread_id cookies.
        Assign a new thread_id if clear_history is True or no thread_id is found.
        """
        latest_cookie_key = f"{self.ROOM_NAME}_{AI_THREAD_COOKIE_KEY}"
        anon_cookie_key = f"{self.ROOM_NAME}_{AI_THREADS_ANONYMOUS_COOKIE_KEY}"

        threads_ids_str = self.scope["cookies"].get(anon_cookie_key) or ""
        thread_ids = [tid for tid in (threads_ids_str).split(",") if tid]

        if thread_ids:
            # Anonymous users may have multiple thread ids, ordered by most recent last.
            current_thread_id = thread_ids[-1] if thread_ids else None
        else:
            # Logged in users have a single thread id (except after first login).
            current_thread_id = self.scope["cookies"].get(latest_cookie_key) or ""

        if (
            thread_id
            and user
            and not user.is_anonymous
            and await UserChatSession.objects.filter(
                user=user, thread_id=thread_id
            ).aexists()
        ):
            current_thread_id = thread_id

        if clear_history or not current_thread_id:
            current_thread_id = uuid4().hex

        if not user.is_anonymous:
            if thread_ids:
                # Assign old thread ids to this user
                await UserChatSession.objects.filter(
                    user_id=None, thread_id__in=thread_ids
                ).aupdate(user=user)
            cookies = [
                f"{latest_cookie_key}={current_thread_id}; Path=/; HttpOnly",
                f"{anon_cookie_key}=; Path=/; HttpOnly",
            ]
        else:
            # Append current thread_id to cookie for anonymous users
            if current_thread_id and current_thread_id not in thread_ids:
                thread_ids.append(current_thread_id)
            thread_ids_str = f"{','.join(thread_ids)}," if thread_ids else ""
            cookies = [
                f"{latest_cookie_key}=; Path=/; HttpOnly",
                f"{anon_cookie_key}={thread_ids_str}; Path=/; HttpOnly",
            ]
        return current_thread_id, cookies

    async def prepare_response(self, serializer: ChatRequestSerializer) -> Tuple[str, list[str]]:
        """Prepare the response"""
        user = self.scope.get("user", None)
        session = self.scope.get("session", None)

        current_thread_id, cookies = await self.assign_thread_cookies(
            user,
            clear_history=serializer.validated_data.pop("clear_history", False),
            thread_id=serializer.validated_data.pop("thread_id", None),
        )

        self.thread_id = current_thread_id

        if user and user.username and user.username != "AnonymousUser":
            self.user_id = user.global_id
        elif session:
            if not session.session_key:
                session.save()
            self.user_id = slugify(session.session_key)
        else:
            log.info("Anon user, no session")
            self.user_id = "Anonymous"

        self.channel_layer = get_channel_layer()
        self.room_name = self.ROOM_NAME
        self.room_group_name = f"{self.ROOM_NAME}_{self.user_id.replace('-', '_')}"[:90]
        await self.channel_layer.group_add(self.room_group_name, self.channel_name)
        return current_thread_id, cookies


    def process_extra_state(self, data: dict) -> dict:  # noqa: ARG002
        """Process extra state if any"""
        return None

    async def start_response(
        self,
            thread_id: Optional[str] = None,
            status: Optional[int] = HTTP_200_OK,
            cookies: Optional[list[str]] = None
    ):
        headers = (
            [
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
            if status == HTTP_200_OK
            else [
                (b"Cache-Control", b"no-cache"),
                (
                    b"Content-Type",
                    b"application/json",
                ),
                (b"Connection", b"close"),
            ]
        )

        if thread_id and cookies:
            headers.extend(
                [(b"Set-Cookie", cookie_thread.encode()) for cookie_thread in cookies]
            )

        await self.send_headers(status=status, headers=headers)
        self.headers_sent = True
        # Headers are only sent after the first body event.
        await self.send_chunk("")

    async def send_error_response(self, status: int, error: Exception, cookies: list[str]):
        """
        Send the appropriate error response. Send error status code if the
        headers have not yet been sent; otherwise it is too late.
        """
        log.exception("Error in consumer handle")
        error_msg = {"error": {"message": str(error)}}
        if not self.headers_sent:
            await self.start_response(status=status, cookies=cookies)
            await self.send_chunk(json.dumps(error_msg))
        else:
            error_msg = json.dumps(error_msg)
            await self.send_chunk(f"<!-- {error_msg} -->")

    async def handle(self, message: str):
        """Handle the incoming message and send the response."""
        cookies = None
        try:
            serializer = self.process_message(message, self.serializer_class)
            thread_id, cookies = await self.prepare_response(serializer)
            message_text = serializer.validated_data["message"]
            checkpointer = await AsyncDjangoSaver.create_with_session(
                thread_id=thread_id,
                message=message_text,
                user=self.scope.get("user", None),
                agent=self.ROOM_NAME,
            )
            self.bot = self.create_chatbot(serializer, checkpointer)
            extra_state = self.process_extra_state(serializer.validated_data)
            # Start to send the response, including the headers
            await self.start_response(thread_id=thread_id, status=200, cookies=cookies)
            # Stream an LLM
            async for chunk in self.bot.get_completion(
                message_text, extra_state=extra_state
            ):
                await self.send_chunk(chunk)
        except (ValidationError, json.JSONDecodeError) as err:
            log.exception("Bad request")
            await self.send_error_response(400, err, cookies)
        except Exception as err:
            log.exception("An error occured in consumer handle")
            await self.send_error_response(500, err, cookies)
        finally:
            await self.send_chunk("", more_body=False)
            await self.disconnect()

    async def disconnect(self):
        """Discard the group when the connection is closed."""
        if hasattr(self, "channel_layer") and hasattr(self, "room_group_name"):
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

    ROOM_NAME = ResourceRecommendationBot.__name__

    def create_chatbot(
        self, serializer: ChatRequestSerializer, checkpointer: BaseCheckpointSaver
    ):
        """Return a ResourceRecommendationBot instance"""
        temperature = serializer.validated_data.pop("temperature", None)
        instructions = serializer.validated_data.pop("instructions", None)
        model = serializer.validated_data.pop("model", None)

        return ResourceRecommendationBot(
            self.user_id,
            checkpointer,
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
    ROOM_NAME = SyllabusBot.__name__

    def create_chatbot(
        self,
        serializer: SyllabusChatRequestSerializer,
        checkpointer: BaseCheckpointSaver,
    ):
        """Return a SyllabusBot instance"""
        temperature = serializer.validated_data.pop("temperature", None)
        instructions = serializer.validated_data.pop("instructions", None)
        model = serializer.validated_data.pop("model", None)

        return SyllabusBot(
            self.user_id,
            checkpointer,
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
