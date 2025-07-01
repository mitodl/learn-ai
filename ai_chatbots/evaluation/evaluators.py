"""Bot-specific evaluators for RAG evaluation."""

from typing import Any, Optional

from langchain_core.messages import AIMessage

from ai_chatbots import chatbots
from main.test_utils import load_json_with_settings

from .base import BaseBotEvaluator, TestCaseSpec


class RecommendationBotEvaluator(BaseBotEvaluator):
    """Evaluator for ResourceRecommendationBot."""

    def load_test_cases(self) -> list[TestCaseSpec]:
        """Load recommendation bot test cases."""
        test_data = load_json_with_settings("test_json/rag_evaluation.json")
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

    def validate_test_case(self, test_case: TestCaseSpec) -> bool:
        """Validate recommendation bot test case."""
        metadata = test_case.metadata or {}
        extra_state = metadata.get("extra_state", {})

        # Recommendation bot requires search_url in extra_state
        return "search_url" in extra_state

    def create_bot_instance(
        self, model: str, test_case: TestCaseSpec, instructions: Optional[str] = None
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

        return chatbot.agent.invoke(
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
        test_data = load_json_with_settings("test_json/rag_evaluation.json")
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
            for case_data in test_data.get("syllabus", [])
        ]

    def validate_test_case(self, test_case: TestCaseSpec) -> bool:
        """Validate syllabus bot test case."""
        metadata = test_case.metadata or {}
        extra_state = metadata.get("extra_state", {})

        # Syllabus bot requires course_id in extra_state
        return "course_id" in extra_state

    def create_bot_instance(
        self, model: str, test_case: TestCaseSpec, instructions: Optional[str] = None
    ):
        """Create syllabus bot instance."""
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
        """Collect response from syllabus bot."""
        metadata = test_case.metadata or {}
        extra_state = metadata.get("extra_state", {})

        return chatbot.agent.invoke(
            {
                "messages": [{"content": test_case.question, "role": "user"}],
                **extra_state,
            },
            config=chatbot.config["configurable"],
        )


class VideoGPTBotEvaluator(BaseBotEvaluator):
    """Evaluator for VideoGPTBot."""

    def load_test_cases(self) -> list[TestCaseSpec]:
        """Load video GPT bot test cases."""
        test_data = load_json_with_settings("test_json/rag_evaluation.json")
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
        self, model: str, test_case: TestCaseSpec, instructions: Optional[str] = None
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

        return chatbot.agent.invoke(
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
        test_data = load_json_with_settings("test_json/rag_evaluation.json")
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
        instructions: Optional[str] = None,  # noqa: ARG002
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
