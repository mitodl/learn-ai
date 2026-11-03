"""Unit tests for bot-specific evaluators."""

import asyncio
import inspect

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from ai_chatbots import learner_api
from ai_chatbots.evaluation.base import TestCaseSpec
from ai_chatbots.evaluation.evaluators import (
    BOT_EVALUATORS,
    RecommendationBotEvaluator,
    SupportBotEvaluator,
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
                    "extra_state": {"search_filters": [{"topic": ["Physics"]}]},
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
        assert test_cases[0].metadata["extra_state"]["search_filters"] == [
            {"topic": ["Physics"]}
        ]
        assert test_cases[0].metadata["extra_init"]["param"] == "value"

    @pytest.mark.parametrize(
        "metadata",
        [
            {"extra_state": {"search_filters": [{"topic": ["Physics"]}]}},
            {"extra_state": {}, "extra_init": {}},
            None,
        ],
    )
    def test_validate_test_case(self, evaluator, metadata):
        """Recommendation bot test cases need no extra state."""
        test_case = TestCaseSpec(question="Test question", metadata=metadata)

        assert evaluator.validate_test_case(test_case) is True

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
        mock_bot.agent.ainvoke = mocker.AsyncMock(
            return_value={"messages": [response_message]}
        )
        mock_bot.config = {"configurable": {"test": "config"}}

        test_case = TestCaseSpec(
            question="Test question",
            metadata={
                "extra_state": {"search_filters": [{"topic": ["Physics"]}]},
                "extra_init": {},
            },
        )

        response = await evaluator.collect_response(mock_bot, test_case)

        mock_bot.agent.ainvoke.assert_called_once_with(
            {
                "messages": [{"content": "Test question", "role": "user"}],
                "search_filters": [{"topic": ["Physics"]}],
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

    @pytest.fixture(autouse=True)
    def no_resource_facts(self, mocker):
        """Keep collect_response from looking up resource facts."""
        mocker.patch(
            "ai_chatbots.evaluation.evaluators.get_resource_facts", return_value=""
        )

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

    @pytest.mark.parametrize(
        ("shared", "expected"),
        [
            # no shared records: a case without its own reads the live API
            (None, ["live", [{"id": "own"}], "error", []]),
            (
                {"enrollments": [{"id": "shared"}]},
                [[{"id": "shared"}], [{"id": "own"}], "error", []],
            ),
        ],
    )
    async def test_learner_records_stay_with_their_case(
        self, evaluator, mocker, shared, expected
    ):
        """Cases run concurrently each read their own records, or their error."""
        for name in (
            "get_program_enrollments",
            "get_program_certificates",
            "get_course_certificate",
            "get_program_certificate",
            "get_course_dates",
        ):
            mocker.patch.object(learner_api, name, getattr(learner_api, name))
        mocker.patch.object(
            learner_api, "get_enrollments", mocker.AsyncMock(return_value="live")
        )
        own = [
            {"enrollments": [{"id": "own"}]},
            {"error": True},
            {},
        ]
        mocker.patch(
            "ai_chatbots.evaluation.evaluators.load_json_with_settings",
            return_value={
                "learner_records": shared,
                "syllabus": [
                    {"question": "a", "extra_state": {"course_id": ["c"]}},
                    *(
                        {
                            "question": "b",
                            "extra_state": {"course_id": ["c"]},
                            "learner_records": records,
                        }
                        for records in own
                    ),
                ],
            },
        )
        cases = evaluator.load_test_cases()
        evaluator.load_test_cases()  # loading again doesn't wrap the API twice

        async def ainvoke(state, config):
            await asyncio.sleep(0)  # let the other cases set their records
            try:
                return {"records": await learner_api.get_enrollments()}
            except learner_api.LearnerAPIError:
                return {"records": "error"}

        chatbot = mocker.Mock(config={"configurable": {}})
        chatbot.agent.ainvoke = ainvoke
        results = await asyncio.gather(
            *(evaluator.collect_response(chatbot, case) for case in cases)
        )

        assert [result["records"] for result in results] == expected

    async def test_history_comes_before_the_question(self, evaluator, mocker):
        """A follow-up case sends the earlier turns, then the question."""
        history = [
            {"role": "user", "content": "Did I earn a certificate?"},
            {"role": "assistant", "content": "Yes."},
        ]
        chatbot = mocker.Mock(config={"configurable": {}})
        chatbot.agent.ainvoke = mocker.AsyncMock()
        case = TestCaseSpec(
            question="And a digital credential?",
            metadata={"extra_state": {"course_id": ["c"]}, "history": history},
        )

        await evaluator.collect_response(chatbot, case)

        sent = chatbot.agent.ainvoke.call_args.args[0]
        assert sent["messages"] == [
            *history,
            {"role": "user", "content": "And a digital credential?"},
        ]

    @pytest.mark.parametrize(
        ("facts", "instructions", "expected"),
        [
            ("Facts", "Prompt\n\nFacts", ["Facts"]),
            ("", "Prompt", []),
            # Lookup failed when the bot was built, then succeeded here
            ("Facts", "Prompt", []),
        ],
    )
    @pytest.mark.asyncio
    async def test_collect_response_adds_resource_facts(
        self, evaluator, mocker, facts, instructions, expected
    ):
        """The resource facts in the system prompt are returned as extra context."""
        mock_facts = mocker.patch(
            "ai_chatbots.evaluation.evaluators.get_resource_facts",
            return_value=facts,
        )
        mock_bot = mocker.Mock()
        mock_bot.agent.ainvoke = mocker.AsyncMock(
            return_value={"messages": [AIMessage(content="Response")]}
        )
        mock_bot.config = {"configurable": {}}
        mock_bot.instructions = instructions
        test_case = TestCaseSpec(
            question="Test question",
            metadata={
                "extra_state": {"course_id": ["8.01SC+fall_2016"], "platform": ["ocw"]},
                "extra_init": {},
            },
        )

        response = await evaluator.collect_response(mock_bot, test_case)

        mock_facts.assert_called_once_with("8.01SC+fall_2016", "ocw")
        assert response["extra_context"] == expected
        assert response["messages"][0].content == "Response"


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


class TestSupportBotEvaluator:
    """Test cases for SupportBotEvaluator."""

    @pytest.fixture
    def evaluator(self, mocker):
        """Create evaluator for testing."""
        return SupportBotEvaluator(mocker.Mock(), "support")

    def test_load_test_cases(self, evaluator, mocker):
        """Cases come from the file's "support" key, with its shared records."""
        mocker.patch(
            "ai_chatbots.evaluation.evaluators.load_json_with_settings",
            return_value={
                "learner_records": {"enrollments": []},
                "syllabus": [{"question": "not this one"}],
                "support": [
                    {
                        "question": "what's my grade?",
                        "expected_output": "0%",
                        "expected_tools": ["get_course_progress"],
                        "extra_state": {"page_url": ["https://courses/c"]},
                        "extra_init": {"enable_account_tools": True},
                    }
                ],
            },
        )

        cases = evaluator.load_test_cases()

        assert [case.question for case in cases] == ["what's my grade?"]
        assert cases[0].expected_tools == ["get_course_progress"]
        assert cases[0].metadata["extra_state"]["page_url"] == ["https://courses/c"]
        assert cases[0].metadata["learner_records"] == {"enrollments": []}

    async def test_the_evaluator_drives_the_answering_graph(self, evaluator, mocker):
        """
        The eval must run the graph that calls the model.

        SupportBot has two. `agent` is the intake graph, which never reaches a
        model and emits the scripted reply, so running it would let every
        "calls no tool" case pass without proving anything and make every
        "calls this tool" case fail for the wrong reason.
        """
        chatbot = mocker.Mock(config={"configurable": {}})
        chatbot.agent.ainvoke = mocker.AsyncMock(return_value="intake")
        chatbot._answering_agent.ainvoke = mocker.AsyncMock(return_value="answered")  # noqa: SLF001
        case = TestCaseSpec(
            question="what's my grade?",
            metadata={
                "extra_state": {"page_url": ["https://courses/c"]},
                "extra_init": {},
                "history": [],
                "learner_records": None,
            },
        )

        result = await evaluator.collect_response(chatbot, case)

        assert result == "answered"
        chatbot.agent.ainvoke.assert_not_called()
        state = chatbot._answering_agent.ainvoke.call_args[0][0]  # noqa: SLF001
        assert state["page_url"] == ["https://courses/c"]

    @pytest.mark.parametrize(
        ("extra_init", "valid"),
        [({"enable_account_tools": True}, True), ({}, False)],
    )
    def test_a_case_without_tools_has_nothing_to_measure(
        self, evaluator, extra_init, valid
    ):
        """
        No tools means no answering graph at all, so the case would be scored
        against a bot that cannot answer - a failure that says nothing about
        the prompt.
        """
        case = TestCaseSpec(
            question="q", metadata={"extra_state": {}, "extra_init": extra_init}
        )

        assert evaluator.validate_test_case(case) is valid

    def test_no_course_id_is_needed(self, evaluator):
        """
        Unlike the syllabus bot, the support bot is never handed a course: it
        reads one off page_url, so requiring course_id would reject every case
        that tests exactly that.
        """
        case = TestCaseSpec(
            question="q",
            metadata={
                "extra_state": {"page_url": ["https://courses/c"]},
                "extra_init": {"enable_account_tools": True},
            },
        )

        assert evaluator.validate_test_case(case) is True


class TestBotEvaluatorRegistry:
    """Test cases for bot evaluator registry."""

    def test_bot_evaluators_registry(self):
        """Test that all expected bot evaluators are registered."""
        expected_bots = ["recommendation", "syllabus", "video_gpt", "tutor", "support"]

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
            "support": SupportBotEvaluator,
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
                    "extra_state": {"search_filters": [{"topic": ["Physics"]}]},
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
        mock_bot.agent.ainvoke = mocker.AsyncMock(
            return_value={"messages": [HumanMessage(content="Bot response")]}
        )
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
