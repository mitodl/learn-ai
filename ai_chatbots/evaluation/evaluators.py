"""Bot-specific evaluators for RAG evaluation."""

import json
from contextvars import ContextVar
from types import SimpleNamespace
from typing import Any

from asgiref.sync import sync_to_async
from langchain_core.messages import AIMessage
from open_learning_ai_tutor import prompts as tutor_prompts

from ai_chatbots import chatbots, learner_api
from ai_chatbots.api import get_resource_facts
from ai_chatbots.evaluation.base import BaseBotEvaluator, TestCaseSpec
from ai_chatbots.evaluation.checks import HTML_COMMENT
from main.test_utils import load_json_with_settings

# Outside an eval case the account tools read the live APIs as usual
_NOT_EVALUATING = object()

# The case's learner records (None: it has none). Per context, so concurrent
# cases each see their own.
_learner_records: ContextVar[Any] = ContextVar(
    "learner_records", default=_NOT_EVALUATING
)
# Account lookups the case tried without records. A list, so tool calls run in
# a copied context still record into it.
_blocked_calls: ContextVar[list | None] = ContextVar("blocked_calls", default=None)

# What TutorBot.get_completion streams when it fails
TUTOR_ERROR_MARKER = '<!-- {"error":'

# The tutor_problem template of the case being evaluated (None: the package's).
# Per context: asyncio.to_thread copies it into the thread that builds the prompt.
_tutor_problem_template: ContextVar[str | None] = ContextVar(
    "tutor_problem_template", default=None
)


def _serve_tutor_problem_template():
    """Make the tutor package use the case's tutor_problem template, if any."""
    original = tutor_prompts.get_system_prompt
    if getattr(original, "serves_eval_template", False):
        return

    def get_system_prompt(prompt_name, mapping, cache_func):
        template = _tutor_problem_template.get()
        if prompt_name == "tutor_problem" and template is not None:
            return template
        return original(prompt_name, mapping, cache_func)

    get_system_prompt.serves_eval_template = True
    tutor_prompts.get_system_prompt = get_system_prompt


class LiveAccountCallBlockedError(RuntimeError):
    """An eval case without learner records tried to read the live account APIs."""


# learner_api function: the key of a case's learner_records that replaces it
_RECORD_KEYS = {
    "get_enrollments": "enrollments",
    "get_program_enrollments": "program_enrollments",
    "get_program_certificates": "program_certificates",
    "get_course_certificate": "course_certificates",
    "get_program_certificate": "program_certificate_details",
    "get_course_dates": "course_dates",
}


def _serve_learner_records():
    """
    Make the account tools read the eval case's learner_records instead of the
    MIT Learn and MITx Online APIs; a case without records is blocked and
    recorded. Lists default to empty, a certificate not in the records has no
    digital credential, and {"error": true} makes every lookup fail.
    """
    for name, key in _RECORD_KEYS.items():
        live = getattr(learner_api, name)
        if getattr(live, "serves_learner_records", False) is True:
            continue

        # async like the originals, so patching one later still gives an AsyncMock
        async def serve(*args, _live=live, _key=key, _name=name):
            records = _learner_records.get()
            if records is _NOT_EVALUATING:
                return await _live(*args)
            if records is None:
                blocked = _blocked_calls.get()
                if blocked is not None:
                    blocked.append(_name)
                raise LiveAccountCallBlockedError(_name)
            return await _from_records(records, _key, *args)

        serve.serves_learner_records = True
        setattr(learner_api, name, serve)


async def _from_records(records: dict, key: str, lookup_id=None):
    """
    Look up one learner_api result in a case's learner_records. lookup_id is a
    certificate's uuid or a course run's key.
    """
    if records.get("error"):
        raise learner_api.LearnerAPIError(key)
    if lookup_id is None:
        return records.get(key, [])
    return records.get(key, {}).get(lookup_id, {})


class RecommendationBotEvaluator(BaseBotEvaluator):
    """Evaluator for ResourceRecommendationBot."""

    def load_test_cases(self) -> list[TestCaseSpec]:
        """Load recommendation bot test cases."""
        test_data = load_json_with_settings(self.data_file)
        return [
            TestCaseSpec(
                question=case_data["question"],
                expected_output=case_data.get("expected_output"),
                expected_tools=case_data.get("expected_tools"),
                metadata={
                    "extra_state": case_data.get("extra_state", {}),
                    "extra_init": case_data.get("extra_init", {}),
                },
            )
            for case_data in test_data.get("recommendation", [])
        ]

    def validate_test_case(self, test_case: TestCaseSpec) -> bool:  # noqa: ARG002
        """Recommendation bot test cases need no extra state."""
        return True

    def create_bot_instance(
        self, model: str, test_case: TestCaseSpec, instructions: str | None = None
    ):
        """Create recommendation bot instance."""
        metadata = test_case.metadata or {}
        extra_init = metadata.get("extra_init", {})

        return self.bot_class(
            "eval",
            checkpointer=None,
            model=model,
            instructions=instructions,
            **extra_init,
        )

    async def collect_response(
        self, chatbot, test_case: TestCaseSpec
    ) -> dict[str, Any]:
        """Collect response from recommendation bot."""
        metadata = test_case.metadata or {}
        extra_state = metadata.get("extra_state", {})

        return await chatbot.agent.ainvoke(
            {
                "messages": [{"content": test_case.question, "role": "user"}],
                **extra_state,
            },
            config=chatbot.config["configurable"],
        )


class SyllabusBotEvaluator(BaseBotEvaluator):
    """Evaluator for SyllabusBot."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _serve_learner_records()  # so no case can reach the live account APIs

    def load_test_cases(self) -> list[TestCaseSpec]:
        """Load syllabus bot test cases."""
        _serve_learner_records()  # again, in case they were replaced since __init__
        test_data = load_json_with_settings(self.data_file)
        # Records shared by the file's cases; a case can bring its own
        default_records = test_data.get("learner_records")
        return [
            TestCaseSpec(
                question=case_data["question"],
                expected_output=case_data.get("expected_output"),
                expected_tools=case_data.get("expected_tools"),
                metadata={
                    "extra_state": case_data.get("extra_state", {}),
                    "extra_init": case_data.get("extra_init", {}),
                    "history": case_data.get("history", []),
                    "learner_records": case_data.get(
                        "learner_records", default_records
                    ),
                },
            )
            # The Canvas bot answers the same questions about one resource
            for case_data in test_data.get("syllabus", [])
        ]

    def validate_test_case(self, test_case: TestCaseSpec) -> bool:
        """Validate syllabus bot test case."""
        metadata = test_case.metadata or {}
        extra_state = metadata.get("extra_state", {})

        # Syllabus bot requires course_id in extra_state
        return "course_id" in extra_state

    def create_bot_instance(
        self, model: str, test_case: TestCaseSpec, instructions: str | None = None
    ):
        """Create syllabus bot instance."""
        metadata = test_case.metadata or {}
        extra_init = metadata.get("extra_init", {})
        extra_state = metadata.get("extra_state", {})

        return self.bot_class(
            "eval",
            checkpointer=None,
            model=model,
            instructions=instructions,
            # the consumer passes these to the constructor as well as to the
            # state, since the resource facts go into the system prompt
            course_id=(extra_state.get("course_id") or [None])[-1],
            platform=(extra_state.get("platform") or [None])[-1],
            **extra_init,
        )

    async def collect_response(
        self, chatbot, test_case: TestCaseSpec
    ) -> dict[str, Any]:
        """Collect response from syllabus bot."""
        metadata = test_case.metadata or {}
        extra_state = metadata.get("extra_state", {})
        token = _learner_records.set(metadata.get("learner_records"))
        blocked: list[str] = []
        blocked_token = _blocked_calls.set(blocked)
        try:
            response = await chatbot.agent.ainvoke(
                {
                    # Earlier turns of the conversation, for follow-up questions
                    "messages": [
                        *metadata.get("history", []),
                        {"content": test_case.question, "role": "user"},
                    ],
                    **extra_state,
                },
                config=chatbot.config["configurable"],
            )
        finally:
            _learner_records.reset(token)
            _blocked_calls.reset(blocked_token)
        response["blocked_account_calls"] = blocked
        # The resource facts reach the bot through its system prompt, not a tool.
        # The lookup is cached, but a failure only briefly, so use the facts only
        # if they made it into the prompt this bot was built with.
        facts = await sync_to_async(get_resource_facts)(
            (extra_state.get("course_id") or [None])[-1],
            (extra_state.get("platform") or [None])[-1],
        )
        in_prompt = facts and facts in (chatbot.instructions or "")
        response["extra_context"] = [facts] if in_prompt else []
        return response


class VideoGPTBotEvaluator(BaseBotEvaluator):
    """Evaluator for VideoGPTBot."""

    def load_test_cases(self) -> list[TestCaseSpec]:
        """Load video GPT bot test cases."""
        test_data = load_json_with_settings(self.data_file)
        return [
            TestCaseSpec(
                question=case_data["question"],
                expected_output=case_data.get("expected_output"),
                expected_tools=case_data.get("expected_tools"),
                metadata={
                    "extra_state": case_data.get("extra_state", {}),
                    "extra_init": case_data.get("extra_init", {}),
                },
            )
            for case_data in test_data.get("video_gpt", [])
        ]

    def validate_test_case(self, test_case: TestCaseSpec) -> bool:
        """Validate video GPT bot test case."""
        metadata = test_case.metadata or {}
        extra_state = metadata.get("extra_state", {})

        # Video GPT bot requires transcript_asset_id in extra_state
        return "transcript_asset_id" in extra_state

    def create_bot_instance(
        self, model: str, test_case: TestCaseSpec, instructions: str | None = None
    ):
        """Create video GPT bot instance."""
        metadata = test_case.metadata or {}
        extra_init = metadata.get("extra_init", {})

        return self.bot_class(
            "eval",
            checkpointer=None,
            model=model,
            instructions=instructions,
            **extra_init,
        )

    async def collect_response(
        self, chatbot, test_case: TestCaseSpec
    ) -> dict[str, Any]:
        """Collect response from video GPT bot."""
        metadata = test_case.metadata or {}
        extra_state = metadata.get("extra_state", {})

        return await chatbot.agent.ainvoke(
            {
                "messages": [{"content": test_case.question, "role": "user"}],
                **extra_state,
            },
            config=chatbot.config["configurable"],
        )


class TutorBotEvaluator(BaseBotEvaluator):
    """Evaluator for TutorBot."""

    def load_test_cases(self) -> list[TestCaseSpec]:
        """Load tutor bot test cases."""
        test_data = load_json_with_settings(self.data_file)
        return [
            TestCaseSpec(
                question=case_data["question"],
                expected_output=case_data.get("expected_output"),
                expected_tools=case_data.get("expected_tools"),
                metadata={
                    "extra_state": case_data.get("extra_state", {}),
                    "extra_init": case_data.get("extra_init", {}),
                    # Saved tutor state before this turn, to replay mid-conversation
                    "history": case_data.get("history"),
                },
            )
            for case_data in test_data.get("tutor", [])
        ]

    def validate_test_case(self, test_case: TestCaseSpec) -> bool:
        """EdX problems need edx_module_id; Canvas ones a run and problem set."""
        extra_init = (test_case.metadata or {}).get("extra_init", {})
        return (
            "edx_module_id" in extra_init
            or {
                "run_readable_id",
                "problem_set_title",
            }
            <= extra_init.keys()
        )

    def create_bot_instance(
        self, model: str, test_case: TestCaseSpec, instructions: str | None = None
    ):
        """Create tutor bot instance; instructions replace its tutor_problem prompt."""
        extra_init = (test_case.metadata or {}).get("extra_init", {})
        bot = self.bot_class("eval", checkpointer=None, model=model, **extra_init)
        bot.eval_problem_template = instructions
        return bot

    async def collect_response(
        self, chatbot, test_case: TestCaseSpec
    ) -> dict[str, Any]:
        """Collect a tutor answer, with the problem and conversation as evidence."""
        history = (test_case.metadata or {}).get("history")
        if history:
            saved = SimpleNamespace(chat_json=json.dumps(history))

            async def latest_history():
                return saved

            chatbot.get_latest_history = latest_history

        _serve_tutor_problem_template()
        token = _tutor_problem_template.set(chatbot.eval_problem_template)
        try:
            chunks = [
                chunk async for chunk in chatbot.get_completion(test_case.question)
            ]
        finally:
            _tutor_problem_template.reset(token)
        if any(TUTOR_ERROR_MARKER in chunk for chunk in chunks):
            msg = "the tutor streamed an error instead of an answer"
            raise RuntimeError(msg)
        # Drop the trailing <!-- metadata --> comment the tutor appends
        answer = HTML_COMMENT.sub("", "".join(chunks)).strip()

        problem_set = chatbot.problem_set
        if isinstance(problem_set, dict):
            problem_set = "\n\n".join(
                f"[{kind}] {f.get('file_name', '')}\n{f.get('content', '')}"
                for kind, files in problem_set.items()
                for f in files
            )
        evidence = [f"Problem:\n{chatbot.problem}"] if chatbot.problem else []
        if problem_set and problem_set != chatbot.problem:
            evidence.append(f"Problem set and solutions:\n{problem_set}")
        turns = [
            f"{'Student' if m['type'] == 'HumanMessage' else 'Tutor'}: {m['content']}"
            for m in (history or {}).get("chat_history", [])
        ]
        if turns:
            evidence.append("Conversation so far:\n" + "\n\n".join(turns))
        return {"messages": [AIMessage(content=answer)], "extra_context": evidence}


# Bot evaluator registry
BOT_EVALUATORS = {
    "recommendation": (chatbots.ResourceRecommendationBot, RecommendationBotEvaluator),
    "syllabus": (chatbots.SyllabusBot, SyllabusBotEvaluator),
    "syllabus_canvas": (chatbots.CanvasSyllabusBot, SyllabusBotEvaluator),
    "video_gpt": (chatbots.VideoGPTBot, VideoGPTBotEvaluator),
    "tutor": (chatbots.TutorBot, TutorBotEvaluator),
}
