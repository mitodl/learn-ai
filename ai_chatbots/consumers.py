import json
import logging
from abc import ABC, abstractmethod
from typing import Optional
from uuid import uuid4

from channels.generic.http import AsyncHttpConsumer
from channels.layers import get_channel_layer
from django.conf import settings
from django.utils.encoding import force_bytes
from django.utils.http import urlsafe_base64_encode
from langgraph.checkpoint.base import BaseCheckpointSaver
from langsmith import trace
from rest_framework.exceptions import ValidationError
from rest_framework.status import HTTP_200_OK

from ai_chatbots.chatbots import (
    CanvasSyllabusBot,
    ResourceRecommendationBot,
    SyllabusBot,
    TutorBot,
    VideoGPTBot,
)
from ai_chatbots.checkpointers import AsyncDjangoSaver
from ai_chatbots.constants import (
    AI_THREAD_COOKIE_KEY,
    AI_THREADS_ANONYMOUS_COOKIE_KEY,
    ChatbotCookie,
)
from ai_chatbots.models import UserChatSession
from ai_chatbots.serializers import (
    CanvasTutorChatRequestSerializer,
    ChatRequestSerializer,
    RecommendationChatRequestSerializer,
    SyllabusChatRequestSerializer,
    TutorChatRequestSerializer,
    VideoGPTRequestSerializer,
)
from main.consumers import BaseThrottledAsyncConsumer
from main.exceptions import AsyncThrottled
from main.utils import decode_value, format_seconds
from users.models import User

log = logging.getLogger(__name__)


class BaseBotHttpConsumer(ABC, AsyncHttpConsumer, BaseThrottledAsyncConsumer):
    """Base HttpConsumer for chatbots"""

    # Each bot consumer should define a unique ROOM_NAME
    ROOM_NAME = None

    serializer_class = RecommendationChatRequestSerializer
    headers_sent = False
    session_key = ""

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
            data=text_data_json, context={"user": self.scope.get("user", None)}
        )
        serializer.is_valid(raise_exception=True)
        return serializer

    async def assign_thread_cookies(
        self,
        user: User,
        *,
        clear_history: Optional[bool] = False,
        thread_id: Optional[str] = None,
        object_id: Optional[str] = None,
    ) -> tuple[str, list[str]]:
        """
        Extract and update separate cookie values for logged in vs anonymous users.
        Each chatbot should have its own thread_id cookies.
        Each course/video/object-specific chatbot should have its own thread id.
        Assign a new thread_id if clear_history is True or no thread_id is found.
        """
        if not object_id:
            object_id = ""
        latest_cookie_key = f"{self.ROOM_NAME}_{AI_THREAD_COOKIE_KEY}"
        anon_cookie_key = f"{self.ROOM_NAME}_{AI_THREADS_ANONYMOUS_COOKIE_KEY}"

        current_thread_id = None
        anon_cookie = False
        if clear_history:
            # Create a new random thread id
            current_thread_id = uuid4().hex
        elif (
            # If an explicit thread_id is passed in the serializer,
            # use it if not related to another user
            thread_id
            and (
                # is a new thread?
                not await UserChatSession.objects.filter(thread_id=thread_id).aexists()
                or (
                    # existing thread belonging to same user?
                    user
                    and not user.is_anonymous
                    and (
                        await UserChatSession.objects.filter(
                            user=user, thread_id=thread_id
                        ).aexists()
                    )
                )
                or (
                    # existing chat thread, anon user, same session key
                    user.is_anonymous
                    and await UserChatSession.objects.filter(
                        user=None, thread_id=thread_id, dj_session_key=self.session_key
                    ).aexists()
                )
            )
        ):
            current_thread_id = thread_id
        else:
            # Use cookie thread id if any, check anon cookie first
            anon_cookie = True
            current_thread_id = decode_value(
                self.scope["cookies"].get(anon_cookie_key) or ""
            )
            if not current_thread_id:
                # no anon cookie thread id, check authenticated cookie
                anon_cookie = False
                current_thread_id = decode_value(
                    self.scope["cookies"].get(latest_cookie_key) or ""
                )
            if current_thread_id and "|" in current_thread_id:
                # Object-specific thread, ensure it's the same object
                current_thread_id, cookie_object_id = current_thread_id.split("|")
                if object_id and object_id != cookie_object_id:
                    current_thread_id = None
        if user and not user.is_anonymous:
            if anon_cookie:
                # Anon user has logged in, so associate any existing anon threads
                # with the same session key
                await UserChatSession.objects.filter(
                    user_id=None, dj_session_key=self.session_key
                ).aupdate(user=user)
            if not clear_history and current_thread_id:
                # User may have logged in under a different account
                session = (
                    await UserChatSession.objects.filter(thread_id=current_thread_id)
                    .select_related("user")
                    .afirst()
                )
                current_thread_id = (
                    None if (session and session.user != user) else current_thread_id
                )
        current_thread_id = current_thread_id or uuid4().hex

        # Incorporate object id into cookie value if present
        cookie_value = urlsafe_base64_encode(
            force_bytes(f"{current_thread_id}{f'|{object_id}' if object_id else ''}")
        )
        # assign the cookie values
        max_age = settings.AI_CHATBOTS_COOKIE_MAX_AGE
        cookies = [
            ChatbotCookie(
                name=latest_cookie_key,
                value=("" if user.is_anonymous else cookie_value),
                max_age=max_age,
            ),
            ChatbotCookie(
                name=anon_cookie_key,
                value=(cookie_value if user.is_anonymous else ""),
                max_age=max_age,
            ),
        ]
        return current_thread_id, cookies

    async def prepare_response(
        self, serializer: ChatRequestSerializer, object_id_field: Optional[str] = None
    ) -> tuple[str, list[str]]:
        """Prepare consumer for the API response"""
        if object_id_field:
            object_id = f"{serializer.validated_data.get(object_id_field, '')}"
        else:
            object_id = ""

        current_thread_id, cookies = await self.assign_thread_cookies(
            self.scope.get("user", None),
            clear_history=serializer.validated_data.pop("clear_history", False),
            thread_id=serializer.validated_data.pop("thread_id", None),
            object_id=object_id,
        )

        self.thread_id = current_thread_id
        self.user_id = self.get_ident()

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
        cookies: Optional[list[str]] = None,
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
                [
                    (
                        b"Set-Cookie",
                        str(cookie).encode(),
                    )
                    for cookie in cookies
                ]
            )

        await self.send_headers(status=status, headers=headers)
        self.headers_sent = True
        # Headers are only sent after the first body event.
        await self.send_chunk("")

    async def send_error_response(
        self, status: int, error: Exception, cookies: list[str]
    ):
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

    async def create_checkpointer(
        self,
        thread_id: str,
        message: str,
        serializer: ChatRequestSerializer,  # noqa: ARG002
    ):
        """Create a checkpointer instance"""
        return await AsyncDjangoSaver.create_with_session(
            thread_id=thread_id,
            message=message,
            user=self.scope.get("user", None),
            dj_session_key=self.session_key,
            agent=self.ROOM_NAME,
        )

    async def handle(self, message: str):
        """Handle the incoming message and send the response."""
        cookies = None
        try:
            await self.check_throttles()
            serializer = self.process_message(message, self.serializer_class)
            thread_id, cookies = await self.prepare_response(serializer)
            message_text = serializer.validated_data["message"]
            checkpointer = await self.create_checkpointer(
                thread_id, message_text, serializer
            )
            self.bot = self.create_chatbot(serializer, checkpointer)
            extra_state = self.process_extra_state(serializer.validated_data)
            # Start to send the response, including the headers
            await self.start_response(thread_id=thread_id, status=200, cookies=cookies)
            # Stream an LLM
            with trace(
                name=self.bot.JOB_ID,
                tags=[self.bot.JOB_ID],
                inputs={"messages": message_text},
                metadata={
                    "thread_id": thread_id,
                    "user_id": self.user_id,
                    "model": self.bot.model,
                },
            ) as langsmith_trace:
                output = []
                async for chunk in self.bot.get_completion(
                    message_text, extra_state=extra_state
                ):
                    await self.send_chunk(chunk)
                    output.append(chunk)
                langsmith_trace.end(outputs={"output": "".join(output)})
        except (ValidationError, json.JSONDecodeError) as err:
            log.exception("Bad request")
            await self.send_error_response(400, err, cookies)
        except AsyncThrottled as err:
            log_msg = "User %s throttled on %s for %d seconds" % (
                self.get_ident(),
                self.__class__.__name__,
                err.wait,
            )
            log.info(log_msg)
            await self.start_response(thread_id=None, status=200, cookies=cookies)
            await self.send_chunk(
                f"You have reached the maximum number of chat requests.\
                \nPlease try again in {format_seconds(err.wait)}."
            )
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
        except:  # noqa: E722
            log.exception("Error in handling consumer http_request")
        finally:
            await self.disconnect()


class RecommendationBotHttpConsumer(BaseBotHttpConsumer):
    """
    Async HTTP consumer for the recommendation agent.
    """

    ROOM_NAME = ResourceRecommendationBot.__name__
    throttle_scope = "recommendation_bot"

    def process_extra_state(self, data: dict) -> dict:
        """Process extra state parameters if any"""
        return {
            "search_url": [data.get("search_url") or settings.AI_MIT_SEARCH_URL],
        }

    def create_chatbot(
        self,
        serializer: RecommendationChatRequestSerializer,
        checkpointer: BaseCheckpointSaver,
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
    throttle_scope = "syllabus_bot"

    def create_chatbot(
        self,
        serializer: SyllabusChatRequestSerializer,
        checkpointer: BaseCheckpointSaver,
    ):
        """Return a SyllabusBot instance"""
        temperature = serializer.validated_data.pop("temperature", None)
        instructions = serializer.validated_data.pop("instructions", None)
        model = serializer.validated_data.pop("model", None)
        enable_related_courses = bool(serializer.validated_data.get("related_courses"))

        return SyllabusBot(
            self.user_id,
            checkpointer,
            temperature=temperature,
            instructions=instructions,
            model=model,
            thread_id=self.thread_id,
            enable_related_courses=enable_related_courses,
        )

    def process_extra_state(self, data: dict) -> dict:
        """Process extra state parameters if any"""
        user = self.scope.get("user", None)
        related_courses = data.get("related_courses", [])
        params = {
            "course_id": [data.get("course_id")],
            "collection_name": [data.get("collection_name")],
            "exclude_canvas": [str(not user or user.is_anonymous or not user.is_staff)],
        }
        if related_courses:
            params["related_courses"] = related_courses
        return params

    def prepare_response(
        self,
        serializer: SyllabusChatRequestSerializer,
        object_id_field: Optional[str] = None,
    ) -> tuple[str, list[str]]:
        """Set the course id as the default object id field"""
        object_id_field = object_id_field or "course_id"
        return super().prepare_response(serializer, object_id_field=object_id_field)

    async def create_checkpointer(
        self, thread_id: str, message: str, serializer: SyllabusChatRequestSerializer
    ):
        """Create a checkpointer instance"""
        return await AsyncDjangoSaver.create_with_session(
            thread_id=thread_id,
            message=message,
            user=self.scope.get("user", None),
            dj_session_key=self.session_key,
            agent=self.ROOM_NAME,
            object_id=serializer.validated_data.get("course_id"),
        )


class CanvasSyllabusBotHttpConsumer(SyllabusBotHttpConsumer):
    """
    Async HTTP consumer for the Canvas syllabus bot.
    Inherits from SyllabusBotHttpConsumer to reuse the logic.
    """

    ROOM_NAME = "CanvasSyllabusBot"
    throttle_scope = "canvas_syllabus_bot"

    def process_extra_state(self, data: dict) -> dict:
        """Process extra state parameters if any"""
        return {
            **super().process_extra_state(data),
            "exclude_canvas": [str(False)],
        }

    def create_chatbot(
        self,
        serializer: SyllabusChatRequestSerializer,
        checkpointer: BaseCheckpointSaver,
    ):
        """Return a SyllabusBot instance"""
        temperature = serializer.validated_data.pop("temperature", None)
        instructions = serializer.validated_data.pop("instructions", None)
        model = serializer.validated_data.pop("model", None)
        enable_related_courses = bool(serializer.validated_data.get("related_courses"))

        return CanvasSyllabusBot(
            self.user_id,
            checkpointer,
            temperature=temperature,
            instructions=instructions,
            model=model,
            thread_id=self.thread_id,
            enable_related_courses=enable_related_courses,
        )


class DemoCanvasSyllabusBotHttpConsumer(CanvasSyllabusBotHttpConsumer):
    """
    Async HTTP consumer for the Canvas syllabus bot. This is a demo version that
    does not require authentication but is limited to demo runs.
    """

    def create_chatbot(
        self,
        serializer: CanvasTutorChatRequestSerializer,
        checkpointer: BaseCheckpointSaver,
    ):
        """Return a TutorBot instance"""
        course_id = serializer.validated_data.get("course_id", None)

        if course_id not in settings.CANVAS_SYLLABUS_DEMO_READABLE_IDS:
            error = f"Invalid canvas readable_id: {course_id}. "
            raise ValidationError(error)

        else:
            return super().create_chatbot(serializer, checkpointer)


class TutorBotHttpConsumer(BaseBotHttpConsumer):
    """
    Async HTTP consumer for the tutor bot.
    """

    serializer_class = TutorChatRequestSerializer
    ROOM_NAME = TutorBot.__name__
    throttle_scope = "tutor_bot"

    def create_chatbot(
        self,
        serializer: TutorChatRequestSerializer,
        checkpointer: BaseCheckpointSaver,
    ):
        """Return a TutorBot instance"""
        temperature = serializer.validated_data.pop("temperature", None)
        model = serializer.validated_data.pop("model", None)
        block_siblings = serializer.validated_data.pop("block_siblings", None)
        edx_module_id = serializer.validated_data.pop("edx_module_id", None)

        return TutorBot(
            self.user_id,
            checkpointer,
            temperature=temperature,
            model=model,
            thread_id=self.thread_id,
            block_siblings=block_siblings,
            edx_module_id=edx_module_id,
        )

    def prepare_response(
        self,
        serializer: TutorChatRequestSerializer,
        object_id_field: Optional[str] = None,
    ) -> tuple[str, list[str]]:
        """Set the edx_module_id as the default object id field"""
        object_id_field = object_id_field or "edx_module_id"
        return super().prepare_response(serializer, object_id_field=object_id_field)

    async def create_checkpointer(
        self, thread_id: str, message: str, serializer: TutorChatRequestSerializer
    ):
        """Create a checkpointer instance"""
        return await AsyncDjangoSaver.create_with_session(
            thread_id=thread_id,
            message=message,
            user=self.scope.get("user", None),
            dj_session_key=self.session_key,
            agent=self.ROOM_NAME,
            object_id=serializer.validated_data.get("edx_module_id"),
        )


class CanvasTutorBotHttpConsumer(BaseBotHttpConsumer):
    """
    Async HTTP consumer for the canvas tutor bot.
    """

    serializer_class = CanvasTutorChatRequestSerializer
    ROOM_NAME = TutorBot.__name__
    throttle_scope = "tutor_bot"

    def create_chatbot(
        self,
        serializer: CanvasTutorChatRequestSerializer,
        checkpointer: BaseCheckpointSaver,
    ):
        """Return a TutorBot instance"""
        temperature = serializer.validated_data.pop("temperature", None)
        model = serializer.validated_data.pop("model", None)
        problem_set_title = serializer.validated_data.pop("problem_set_title", None)
        run_readable_id = serializer.validated_data.pop("run_readable_id", None)

        return TutorBot(
            self.user_id,
            checkpointer,
            temperature=temperature,
            model=model,
            thread_id=self.thread_id,
            problem_set_title=problem_set_title,
            run_readable_id=run_readable_id,
        )

    def prepare_response(
        self,
        serializer: TutorChatRequestSerializer,
        object_id_field: Optional[str] = None,
    ) -> tuple[str, list[str]]:
        """Set the edx_module_id as the default object id field"""
        object_id_field = "object_id_field"
        return super().prepare_response(serializer, object_id_field=object_id_field)

    async def create_checkpointer(
        self, thread_id: str, message: str, serializer: TutorChatRequestSerializer
    ):
        """Create a checkpointer instance"""
        return await AsyncDjangoSaver.create_with_session(
            thread_id=thread_id,
            message=message,
            user=self.scope.get("user", None),
            dj_session_key=self.session_key,
            agent=self.ROOM_NAME,
            object_id=serializer.validated_data.get("object_id_field"),
        )


class DemoCanvasTutorBotHttpConsumer(CanvasTutorBotHttpConsumer):
    """
    Async HTTP consumer for the tutor bot. This is a demo version that
    does not require authentication but is limited to demo runs.
    """

    def create_chatbot(
        self,
        serializer: CanvasTutorChatRequestSerializer,
        checkpointer: BaseCheckpointSaver,
    ):
        """Return a TutorBot instance"""
        run_readable_id = serializer.validated_data.get("run_readable_id", None)

        if run_readable_id not in settings.CANVAS_TUTOR_DEMO_RUN_READABLE_IDS:
            error = f"Invalid run_readable_id: {run_readable_id}. "
            raise ValidationError(error)

        else:
            return super().create_chatbot(serializer, checkpointer)


class VideoGPTBotHttpConsumer(BaseBotHttpConsumer):
    """
    Async HTTP consumer for the video GPT bot.
    """

    serializer_class = VideoGPTRequestSerializer
    ROOM_NAME = VideoGPTBot.__name__
    throttle_scope = "video_gpt_bot"

    def create_chatbot(
        self,
        serializer: VideoGPTRequestSerializer,
        checkpointer: BaseCheckpointSaver,
    ):
        """Return a VideoGPTBot instance"""
        temperature = serializer.validated_data.pop("temperature", None)
        instructions = serializer.validated_data.pop("instructions", None)
        model = serializer.validated_data.pop("model", None)

        return VideoGPTBot(
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
            "transcript_asset_id": [data.get("transcript_asset_id")],
        }

    def prepare_response(
        self,
        serializer: VideoGPTRequestSerializer,
        object_id_field: Optional[str] = None,
    ) -> tuple[str, list[str]]:
        """Set the problem code as the default object id field"""
        object_id_field = object_id_field or "transcript_asset_id"
        return super().prepare_response(serializer, object_id_field=object_id_field)

    async def create_checkpointer(
        self, thread_id: str, message: str, serializer: VideoGPTRequestSerializer
    ):
        """Create a checkpointer instance"""
        return await AsyncDjangoSaver.create_with_session(
            thread_id=thread_id,
            message=message,
            user=self.scope.get("user", None),
            dj_session_key=self.session_key,
            agent=self.ROOM_NAME,
            object_id=serializer.validated_data.get("transcript_asset_id"),
        )
