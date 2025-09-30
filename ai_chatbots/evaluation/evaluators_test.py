"""Unit tests for bot-specific evaluators."""

import inspect

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from ai_chatbots.evaluation.base import TestCaseSpec
from ai_chatbots.evaluation.evaluators import (
    BOT_EVALUATORS,
    RecommendationBotEvaluator,
    SyllabusBotEvaluator,
    TutorBotEvaluator,
    VideoGPTBotEvaluator,
)


class TestRecommendationBotEvaluator:
    """Test cases for RecommendationBotEvaluator."""

    @pytest.fixture
    def evaluator(self, mocker):
        """Create evaluator for testing."""
        mock_bot_class = mocker.Mock()
        return RecommendationBotEvaluator(mock_bot_class, "recommendation")

    def test_load_test_cases(self, evaluator, mocker):
        """Test loading recommendation bot test cases."""
        mock_load_json = mocker.patch(
            "ai_chatbots.evaluation.evaluators.load_json_with_settings"
        )
        mock_load_json.return_value = {
            "recommendation": [
                {
                    "question": "What courses are available?",
                    "expected_output": "Here are the courses...",
                    "expected_tools": ["search_courses"],
                    "extra_state": {"search_url": ["http://test.com"]},
                    "extra_init": {"param": "value"},
                }
            ]
        }

        test_cases = evaluator.load_test_cases()

        assert len(test_cases) == 1
        assert isinstance(test_cases[0], TestCaseSpec)
        assert test_cases[0].question == "What courses are available?"
        assert test_cases[0].expected_output == "Here are the courses..."
        assert test_cases[0].expected_tools == ["search_courses"]
        assert test_cases[0].metadata["extra_state"]["search_url"] == [
            "http://test.com"
        ]
        assert test_cases[0].metadata["extra_init"]["param"] == "value"

    def test_validate_test_case_valid(self, evaluator):
        """Test validation of valid recommendation bot test case."""
        test_case = TestCaseSpec(
            question="Test question",
            metadata={
                "extra_state": {"search_url": ["http://test.com"]},
                "extra_init": {},
            },
        )

        assert evaluator.validate_test_case(test_case) is True

    def test_validate_test_case_invalid(self, evaluator):
        """Test validation of invalid recommendation bot test case."""
        test_case = TestCaseSpec(
            question="Test question", metadata={"extra_state": {}, "extra_init": {}}
        )

        assert evaluator.validate_test_case(test_case) is False

    def test_validate_test_case_missing_metadata(self, evaluator):
        """Test validation with missing metadata."""
        test_case = TestCaseSpec(question="Test question")

        assert evaluator.validate_test_case(test_case) is False

    def test_create_bot_instance(self, evaluator):
        """Test bot instance creation."""
        test_case = TestCaseSpec(
            question="Test", metadata={"extra_init": {"param": "value"}}
        )

        bot_instance = evaluator.create_bot_instance("gpt-4", test_case)

        evaluator.bot_class.assert_called_once_with(
            "eval", checkpointer=None, model="gpt-4", instructions=None, param="value"
        )
        # Test that the method returns the created instance
        assert bot_instance == evaluator.bot_class.return_value

    @pytest.mark.asyncio
    async def test_collect_response(self, evaluator, mocker):
        """Test response collection."""
        mock_bot = mocker.Mock()
        response_message = HumanMessage(content="Response")
        mock_bot.agent.invoke.return_value = {"messages": [response_message]}
        mock_bot.config = {"configurable": {"test": "config"}}

        test_case = TestCaseSpec(
            question="Test question",
            metadata={
                "extra_state": {"search_url": ["http://test.com"]},
                "extra_init": {},
            },
        )

        response = await evaluator.collect_response(mock_bot, test_case)

        mock_bot.agent.invoke.assert_called_once_with(
            {
                "messages": [{"content": "Test question", "role": "user"}],
                "search_url": ["http://test.com"],
            },
            config={"test": "config"},
        )

        assert "messages" in response
        assert len(response["messages"]) == 1
        assert response["messages"][0].content == "Response"


class TestSyllabusBotEvaluator:
    """Test cases for SyllabusBotEvaluator."""

    @pytest.fixture
    def evaluator(self, mocker):
        """Create evaluator for testing."""
        mock_bot_class = mocker.Mock()
        return SyllabusBotEvaluator(mock_bot_class, "syllabus")

    def test_load_test_cases(self, evaluator, mocker):
        """Test loading syllabus bot test cases."""
        mock_load_json = mocker.patch(
            "ai_chatbots.evaluation.evaluators.load_json_with_settings"
        )
        mock_load_json.return_value = {
            "syllabus": [
                {
                    "question": "Who teaches this course?",
                    "expected_output": "The instructor is...",
                    "expected_tools": ["search_content_files"],
                    "extra_state": {"course_id": ["8.01SC+fall_2016"]},
                }
            ]
        }

        test_cases = evaluator.load_test_cases()

        assert len(test_cases) == 1
        assert isinstance(test_cases[0], TestCaseSpec)
        assert test_cases[0].question == "Who teaches this course?"
        assert test_cases[0].metadata["extra_state"]["course_id"] == [
            "8.01SC+fall_2016"
        ]

    def test_validate_test_case_valid(self, evaluator):
        """Test validation of valid syllabus bot test case."""
        # Create valid TestCaseSpec
        test_case = TestCaseSpec(
            question="Test question",
            metadata={
                "extra_state": {"course_id": ["8.01SC+fall_2016"]},
                "extra_init": {},
            },
        )

        assert evaluator.validate_test_case(test_case) is True

    def test_validate_test_case_invalid(self, evaluator):
        """Test validation of invalid syllabus bot test case."""
        # Create invalid TestCaseSpec
        test_case = TestCaseSpec(
            question="Test question", metadata={"extra_state": {}, "extra_init": {}}
        )

        assert evaluator.validate_test_case(test_case) is False


class TestVideoGPTBotEvaluator:
    """Test cases for VideoGPTBotEvaluator."""

    @pytest.fixture
    def evaluator(self, mocker):
        """Create evaluator for testing."""
        mock_bot_class = mocker.Mock()
        return VideoGPTBotEvaluator(mock_bot_class, "video_gpt")

    def test_load_test_cases(self, evaluator, mocker):
        """Test loading video GPT bot test cases."""
        mock_load_json = mocker.patch(
            "ai_chatbots.evaluation.evaluators.load_json_with_settings"
        )
        mock_load_json.return_value = {
            "video_gpt": [
                {
                    "question": "What is this video about?",
                    "expected_output": "The video discusses...",
                    "expected_tools": ["get_video_transcript_chunk"],
                    "extra_state": {
                        "transcript_asset_id": ["asset-123"],
                        "edx_module_id": ["module-456"],
                    },
                }
            ]
        }

        test_cases = evaluator.load_test_cases()

        assert len(test_cases) == 1
        assert isinstance(test_cases[0], TestCaseSpec)
        assert test_cases[0].question == "What is this video about?"
        assert test_cases[0].metadata["extra_state"]["transcript_asset_id"] == [
            "asset-123"
        ]

    def test_validate_test_case_valid(self, evaluator):
        """Test validation of valid video GPT bot test case."""
        test_case = TestCaseSpec(
            question="Test question",
            metadata={
                "extra_state": {"transcript_asset_id": ["asset-123"]},
                "extra_init": {},
            },
        )

        assert evaluator.validate_test_case(test_case) is True

    def test_validate_test_case_invalid(self, evaluator):
        """Test validation of invalid video GPT bot test case."""
        test_case = TestCaseSpec(
            question="Test question", metadata={"extra_state": {}, "extra_init": {}}
        )

        assert evaluator.validate_test_case(test_case) is False


class TestTutorBotEvaluator:
    """Test cases for TutorBotEvaluator."""

    @pytest.fixture
    def evaluator(self, mocker):
        """Create evaluator for testing."""
        mock_bot_class = mocker.Mock()
        return TutorBotEvaluator(mock_bot_class, "tutor")

    def test_load_test_cases(self, evaluator, mocker):
        """Test loading tutor bot test cases."""
        mock_load_json = mocker.patch(
            "ai_chatbots.evaluation.evaluators.load_json_with_settings"
        )
        mock_load_json.return_value = {
            "tutor": [
                {
                    "question": "Can you help me?",
                    "expected_output": "Think about...",
                    "expected_tools": [],
                    "extra_init": {
                        "edx_module_id": "block-v1:MITxT+3.012Sx+3T2024+type@problem+block@123"
                    },
                }
            ]
        }

        test_cases = evaluator.load_test_cases()

        assert len(test_cases) == 1
        assert isinstance(test_cases[0], TestCaseSpec)
        assert test_cases[0].question == "Can you help me?"
        assert "edx_module_id" in test_cases[0].metadata["extra_init"]

    def test_validate_test_case_valid(self, evaluator):
        """Test validation of valid tutor bot test case."""
        test_case = TestCaseSpec(
            question="Test question",
            metadata={"extra_state": {}, "extra_init": {"edx_module_id": "block-123"}},
        )

        assert evaluator.validate_test_case(test_case) is True

    def test_validate_test_case_invalid(self, evaluator):
        """Test validation of invalid tutor bot test case."""
        test_case = TestCaseSpec(
            question="Test question", metadata={"extra_state": {}, "extra_init": {}}
        )

        assert evaluator.validate_test_case(test_case) is False

    @pytest.mark.asyncio
    async def test_collect_response(self, evaluator, mocker):
        """Test tutor bot response collection."""
        mock_bot = mocker.Mock()

        async def mock_get_completion(question):
            yield "Think "
            yield "about "
            yield "the problem."

        mock_bot.get_completion = mock_get_completion

        test_case = TestCaseSpec(question="Can you help me?")

        response = await evaluator.collect_response(mock_bot, test_case)

        assert "messages" in response
        assert len(response["messages"]) == 1
        assert isinstance(response["messages"][0], AIMessage)
        assert response["messages"][0].content == "Think about the problem."


class TestBotEvaluatorRegistry:
    """Test cases for bot evaluator registry."""

    def test_bot_evaluators_registry(self):
        """Test that all expected bot evaluators are registered."""
        expected_bots = ["recommendation", "syllabus", "video_gpt", "tutor"]

        # Test BOT_EVALUATORS registry
        assert set(BOT_EVALUATORS.keys()) == set(expected_bots)

        # Test that each entry has bot class and evaluator class
        for bot_class, evaluator_class in BOT_EVALUATORS.values():
            assert bot_class is not None
            assert evaluator_class is not None
            assert inspect.isclass(evaluator_class)

    def test_evaluator_class_names(self):
        """Test that evaluator classes have expected names."""
        expected_classes = {
            "recommendation": RecommendationBotEvaluator,
            "syllabus": SyllabusBotEvaluator,
            "video_gpt": VideoGPTBotEvaluator,
            "tutor": TutorBotEvaluator,
        }

        for bot_name, (_, evaluator_class) in BOT_EVALUATORS.items():
            assert evaluator_class == expected_classes[bot_name]


class TestEvaluatorIntegration:
    """Integration tests for evaluators."""

    @pytest.mark.asyncio
    async def test_evaluator_workflow(self, mocker):
        """Test complete evaluator workflow."""
        mock_bot_class = mocker.Mock()
        evaluator = RecommendationBotEvaluator(mock_bot_class, "recommendation")

        mock_load = mocker.patch(
            "ai_chatbots.evaluation.evaluators.load_json_with_settings"
        )
        mock_load.return_value = {
            "recommendation": [
                {
                    "question": "Test question",
                    "expected_output": "Test output",
                    "expected_tools": ["search_courses"],
                    "extra_state": {"search_url": ["http://test.com"]},
                    "extra_init": {},
                }
            ]
        }

        test_cases = evaluator.load_test_cases()
        assert len(test_cases) == 1
        assert isinstance(test_cases[0], TestCaseSpec)
        assert evaluator.validate_test_case(test_cases[0]) is True

        bot = evaluator.create_bot_instance("gpt-4", test_cases[0])
        assert bot is not None

        mock_bot = mocker.Mock()
        mock_bot.agent.invoke.return_value = {
            "messages": [HumanMessage(content="Bot response")]
        }
        mock_bot.config = {"configurable": {}}

        response = await evaluator.collect_response(mock_bot, test_cases[0])
        assert "messages" in response
        assert isinstance(response["messages"][0], HumanMessage)

        llm_test_case = evaluator.create_llm_test_case(
            test_cases[0], response, "gpt-4", "default"
        )
        assert llm_test_case.name == "recommendation-gpt-4-default"
        assert llm_test_case.input == "Test question"
        assert llm_test_case.actual_output == "Bot response"
