"""Bot-specific evaluators for RAG evaluation."""

from contextvars import ContextVar
from typing import Any

from asgiref.sync import sync_to_async
from langchain_core.messages import AIMessage

from ai_chatbots import chatbots, learner_api
from ai_chatbots.api import get_resource_facts
from ai_chatbots.evaluation.base import BaseBotEvaluator, TestCaseSpec
from main.test_utils import load_json_with_settings

# The learner records of the case being evaluated. A context variable, so cases
# run concurrently each see their own.
_learner_records: ContextVar[dict | None] = ContextVar("learner_records", default=None)


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
    Make the account tools read the learner_records of the case being
    evaluated instead of the MIT Learn and MITx Online APIs. A case without
    records still reads the live APIs. Lists default to empty, a certificate
    not in the records has no digital credential, and {"error": true} makes
    every lookup fail.
    """
    for name, key in _RECORD_KEYS.items():
        live = getattr(learner_api, name)
        if getattr(live, "serves_learner_records", False) is True:
            continue

        def serve(*args, _live=live, _key=key):
            records = _learner_records.get()
            if records is None:
                return _live(*args)
            return _from_records(records, _key, *args)

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

    def load_test_cases(self) -> list[TestCaseSpec]:
        """Load syllabus bot test cases."""
        test_data = load_json_with_settings(self.data_file)
        # Records shared by the file's cases; a case can bring its own
        default_records = test_data.get("learner_records")
        cases = [
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
            for case_data in test_data.get("syllabus", [])
        ]
        if any(case.metadata["learner_records"] is not None for case in cases):
            _serve_learner_records()
        return cases

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
        # The resource facts reach the bot through its system prompt, not a tool
        facts = await sync_to_async(get_resource_facts)(
            (extra_state.get("course_id") or [None])[-1],
            (extra_state.get("platform") or [None])[-1],
        )
        response["extra_context"] = [facts] if facts else []
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
                },
            )
            for case_data in test_data.get("tutor", [])
        ]

    def validate_test_case(self, test_case: TestCaseSpec) -> bool:
        """Validate tutor bot test case."""
        metadata = test_case.metadata or {}
        extra_init = metadata.get("extra_init", {})

        # Tutor bot requires edx_module_id in extra_init
        return "edx_module_id" in extra_init

    def create_bot_instance(
        self,
        model: str,
        test_case: TestCaseSpec,
        instructions: str | None = None,  # noqa: ARG002
    ):
        """Create tutor bot instance."""
        metadata = test_case.metadata or {}
        extra_init = metadata.get("extra_init", {})

        # TutorBot doesn't accept instructions parameter, so we ignore it
        return self.bot_class(
            "eval",
            checkpointer=None,
            model=model,
            **extra_init,
        )

    async def collect_response(
        self, chatbot, test_case: TestCaseSpec
    ) -> dict[str, Any]:
        """Collect response from tutor bot (special async handling)."""
        chunks = [chunk async for chunk in chatbot.get_completion(test_case.question)]

        return {"messages": [AIMessage(content="".join(chunks))]}


# Bot evaluator registry
BOT_EVALUATORS = {
    "recommendation": (chatbots.ResourceRecommendationBot, RecommendationBotEvaluator),
    "syllabus": (chatbots.SyllabusBot, SyllabusBotEvaluator),
    "video_gpt": (chatbots.VideoGPTBot, VideoGPTBotEvaluator),
    "tutor": (chatbots.TutorBot, TutorBotEvaluator),
}
