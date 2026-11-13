"""Unit tests for evaluation orchestrator."""

import os
from unittest.mock import AsyncMock, patch

import pytest
from deepeval.test_case import LLMTestCaseParams

from ai_chatbots.chatbots import ResourceRecommendationBot
from ai_chatbots.evaluation.base import EvaluationConfig
from ai_chatbots.evaluation.orchestrator import EvaluationOrchestrator
from ai_chatbots.evaluation.timeout_wrapper import TimeoutMetricWrapper

NUM_METRICS = 5


def _answer(model, question="Q1", case_id=None):
    """Build an answer shaped like the ones evaluate_model returns."""
    from deepeval.test_case import LLMTestCase

    case_id = case_id or f"f.json#{question}"
    return LLMTestCase(
        input=question,
        actual_output=f"{model} answer",
        retrieval_context=["evidence"],
        metadata={
            "bot_name": "test_bot",
            "model": model,
            "prompt_label": "default",
            "case_id": case_id,
            "metadata": {"case_id": case_id},
        },
    )


@pytest.fixture(autouse=True)
def mock_prepare(mocker):
    """Keep TimeoutMetricWrapper.prepare from asking the judge for GEval steps."""
    return mocker.patch.object(TimeoutMetricWrapper, "prepare")


@pytest.fixture
def mock_evaluator(mocker):
    """Create a mock evaluator with standard behavior."""
    mock_evaluator_class = mocker.Mock()
    mock_evaluator = mocker.Mock()
    mock_evaluator.load_cases.return_value = [mocker.Mock()]
    mock_evaluator.evaluate_model = AsyncMock(
        side_effect=lambda model, *_args, **_kwargs: [_answer(model)]
    )
    mock_evaluator_class.return_value = mock_evaluator
    return mock_evaluator_class, mock_evaluator


class TestEvaluationOrchestrator:
    """Test cases for EvaluationOrchestrator."""

    @pytest.fixture
    def orchestrator(self, mock_stdout):
        """Create orchestrator for testing."""
        return EvaluationOrchestrator(mock_stdout)

    def test_orchestrator_initialization(self, orchestrator, mock_stdout):
        """Test orchestrator initialization."""
        assert orchestrator.stdout == mock_stdout
        assert orchestrator.reporter is not None

    def test_create_evaluation_config_default_thresholds(self, orchestrator):
        """Test evaluation config creation with default thresholds."""
        models = ["gpt-4o", "gpt-3.5-turbo"]
        evaluation_model = "judge-model"

        config = orchestrator.create_evaluation_config(models, evaluation_model)

        assert isinstance(config, EvaluationConfig)
        assert config.models == models
        assert config.evaluation_model == evaluation_model
        assert len(config.metrics) == NUM_METRICS

        for metric in config.metrics:
            assert isinstance(metric, TimeoutMetricWrapper)

        assert config.confident_api_key is not None or config.confident_api_key is None

    def test_create_evaluation_config_custom_thresholds(self, orchestrator):
        """Test evaluation config creation with custom thresholds."""
        models = ["gpt-4o"]
        evaluation_model = "judge-model"
        custom_thresholds = {
            "Faithfulness": 0.9,
            "ContextualRelevancy": 0.8,
            "Hallucination": 0.9,
            "AnswerRelevancy": 0.8,
            "Helpfulness": 0.8,
        }

        config = orchestrator.create_evaluation_config(
            models, evaluation_model, custom_thresholds
        )
        assert isinstance(config, EvaluationConfig)
        assert config.models == models
        assert config.evaluation_model == evaluation_model
        assert len(config.metrics) == NUM_METRICS

    def test_create_evaluation_config_require_expected(self, orchestrator):
        """Test evaluation config creation with require_expected=True."""
        models = ["gpt-4o"]
        evaluation_model = "judge-model"

        config = orchestrator.create_evaluation_config(
            models, evaluation_model, require_expected=True
        )
        assert isinstance(config, EvaluationConfig)
        assert len(config.metrics) == NUM_METRICS

        underlying_metric_names = [
            m.base_metric.__class__.__name__ for m in config.metrics
        ]
        assert "ContextualPrecisionMetric" in underlying_metric_names
        assert "ContextualRecallMetric" in underlying_metric_names

    def test_create_evaluation_config_correctness_only(self, orchestrator):
        """correctness_only should score against the expected answer alone."""
        config = orchestrator.create_evaluation_config(
            ["gpt-4o"], "judge-model", correctness_only=True
        )
        assert len(config.metrics) == 1
        metric = config.metrics[0]
        assert isinstance(metric, TimeoutMetricWrapper)
        assert metric.base_metric.name == "Correctness"
        assert metric.base_metric.threshold == 0.7
        assert metric.base_metric.evaluation_params == [
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ]

    @patch.dict(os.environ, {"CONFIDENT_AI_API_KEY": "test-api-key"})
    def test_create_evaluation_config_with_api_key(self, orchestrator):
        """Test evaluation config creation with API key from environment."""
        config = orchestrator.create_evaluation_config(["gpt-4o"], "judge-model")

        assert config.confident_api_key == "test-api-key"

    def test_get_available_bots(self, orchestrator):
        """Test getting available bot names."""
        available_bots = orchestrator.get_available_bots()

        expected_bots = ["recommendation", "syllabus", "video_gpt", "tutor"]
        assert set(available_bots) == set(expected_bots)

    def test_validate_bot_names_valid(self, orchestrator):
        """Test validation of valid bot names."""
        bot_names = ["recommendation", "syllabus"]

        valid_bots = orchestrator.validate_bot_names(bot_names)

        assert valid_bots == bot_names

    def test_validate_bot_names_invalid(self, orchestrator, mock_stdout):
        """Test validation of invalid bot names."""
        bot_names = ["recommendation", "invalid_bot", "syllabus"]

        valid_bots = orchestrator.validate_bot_names(bot_names)

        assert valid_bots == ["recommendation", "syllabus"]
        # Check that warning was written to stdout
        mock_stdout.write.assert_called()

    def test_validate_bot_names_all_invalid(self, orchestrator):
        """Test validation when all bot names are invalid."""
        bot_names = ["invalid_bot1", "invalid_bot2"]

        valid_bots = orchestrator.validate_bot_names(bot_names)

        assert valid_bots == []

    @pytest.mark.asyncio
    async def test_run_evaluation_success(self, orchestrator, mock_evaluator, mocker):
        """Test successful evaluation run."""
        config = orchestrator.create_evaluation_config(
            models=["gpt-4"], evaluation_model="gpt-4o"
        )
        config.confident_api_key = "test-key"

        mock_evaluator_class, _ = mock_evaluator

        mock_deepeval = mocker.patch("ai_chatbots.evaluation.orchestrator.deepeval")
        mocker.patch(
            "ai_chatbots.evaluation.orchestrator.BOT_EVALUATORS",
            {"test_bot": (mocker.Mock(), mock_evaluator_class)},
        )

        # Create a mock EvaluationResult
        from deepeval.evaluate.types import EvaluationResult

        mock_results = mocker.Mock(spec=EvaluationResult)
        mock_results.test_results = [
            mocker.Mock(
                metadata={"model": "gpt-4", "bot_name": "test_bot"},
                metrics_data=[mocker.Mock()],
            )
        ]
        mock_results.confident_link = None
        mock_results.test_run_id = None
        mock_deepeval.evaluate.return_value = mock_results
        orchestrator.reporter.generate_report = mocker.Mock()

        result = await orchestrator.run_evaluation(config, bot_names=["test_bot"])

        mock_deepeval.evaluate.assert_called_once()
        orchestrator.reporter.generate_report.assert_called_once_with(
            mock_results,
            ["gpt-4"],
            ["test_bot"],
            use_prompts=True,
            metric_thresholds=config.metric_thresholds,
            evaluation_model="gpt-4o",
            metric_names=mocker.ANY,
            failures=orchestrator.failures,
            answer_records=list(orchestrator.answer_records.values()),
        )
        assert result == mock_results

    @pytest.mark.asyncio
    async def test_run_evaluation_default_bots(
        self, orchestrator, mock_evaluator, mocker
    ):
        """Test evaluation run with default bot names."""
        config = orchestrator.create_evaluation_config(
            models=["gpt-4"], evaluation_model="gpt-4o"
        )
        config.confident_api_key = None

        mock_evaluator_class, mock_evaluator_instance = mock_evaluator
        mock_evaluator_instance.load_cases.return_value = []
        mock_evaluator_instance.evaluate_model = AsyncMock(return_value=[])

        mock_evaluators = {}
        for bot_name in ["recommendation", "syllabus", "video_gpt", "tutor"]:
            mock_evaluators[bot_name] = (mocker.Mock(), mock_evaluator_class)

        mock_deepeval = mocker.patch("ai_chatbots.evaluation.orchestrator.deepeval")
        mocker.patch(
            "ai_chatbots.evaluation.orchestrator.BOT_EVALUATORS", mock_evaluators
        )

        # Create a mock EvaluationResult
        from deepeval.evaluate.types import EvaluationResult

        mock_deepeval.evaluate.return_value = EvaluationResult(
            test_results=[], confident_link=None, test_run_id=None
        )
        orchestrator.reporter.generate_report = mocker.Mock()

        await orchestrator.run_evaluation(config, bot_names=None)
        assert len(mock_evaluators) == 4

    @pytest.mark.asyncio
    async def test_run_evaluation_unknown_bot(self, orchestrator, mock_stdout, mocker):
        """Test evaluation run with unknown bot name."""
        config = orchestrator.create_evaluation_config(
            models=["gpt-4"], evaluation_model="gpt-4o"
        )
        config.confident_api_key = None

        # Mock external dependencies
        mock_deepeval = mocker.patch("ai_chatbots.evaluation.orchestrator.deepeval")
        mocker.patch("ai_chatbots.evaluation.orchestrator.BOT_EVALUATORS", {})

        from deepeval.evaluate.types import EvaluationResult

        mock_deepeval.evaluate.return_value = EvaluationResult(
            test_results=[], confident_link=None, test_run_id=None
        )
        orchestrator.reporter.generate_report = mocker.Mock()

        await orchestrator.run_evaluation(config, bot_names=["unknown_bot"])
        mock_stdout.write.assert_called()

    @pytest.mark.asyncio
    async def test_run_evaluation_evaluator_error(
        self, orchestrator, mock_evaluator, mocker
    ):
        """Test evaluation run with evaluator error raises immediately."""
        config = orchestrator.create_evaluation_config(
            models=["gpt-4"], evaluation_model="gpt-4o"
        )
        config.confident_api_key = None

        mock_evaluator_class, mock_evaluator_instance = mock_evaluator
        mock_evaluator_instance.evaluate_model = AsyncMock(
            side_effect=Exception("Test error")
        )

        mocker.patch(
            "ai_chatbots.evaluation.orchestrator.BOT_EVALUATORS",
            {"test_bot": (mocker.Mock(), mock_evaluator_class)},
        )

        with pytest.raises(RuntimeError, match="Test error"):
            await orchestrator.run_evaluation(config, bot_names=["test_bot"])

    @pytest.mark.asyncio
    async def test_run_evaluation_multiple_models(
        self, orchestrator, mock_evaluator, mocker
    ):
        """Test evaluation processes multiple models correctly."""
        config = orchestrator.create_evaluation_config(
            models=["gpt-4", "gpt-3.5", "claude-3"], evaluation_model="gpt-4o"
        )
        config.confident_api_key = None

        mock_evaluator_class, mock_evaluator_instance = mock_evaluator

        # Track which models were called and return unique test cases for each
        model_calls = []

        async def track_evaluate_model(  # noqa: PLR0913
            model,
            test_cases,
            instructions=None,
            prompt_label="default",
            max_concurrent=10,
            failures=None,
        ):
            model_calls.append(model)
            # Return unique test case for each model to verify they're all collected
            # Tagged with its model, to check they're all collected
            return [_answer(model)]

        mock_evaluator_instance.evaluate_model = track_evaluate_model

        # Mock external dependencies
        mock_deepeval = mocker.patch("ai_chatbots.evaluation.orchestrator.deepeval")
        mocker.patch(
            "ai_chatbots.evaluation.orchestrator.BOT_EVALUATORS",
            {"test_bot": (mocker.Mock(), mock_evaluator_class)},
        )

        from deepeval.evaluate.types import EvaluationResult

        mock_deepeval.evaluate.return_value = EvaluationResult(
            test_results=[], confident_link=None, test_run_id=None
        )
        orchestrator.reporter.generate_report = mocker.Mock()

        await orchestrator.run_evaluation(config, bot_names=["test_bot"])
        assert set(model_calls) == {"gpt-4", "gpt-3.5", "claude-3"}
        mock_deepeval.evaluate.assert_called_once()
        call_args = mock_deepeval.evaluate.call_args
        test_cases_passed = call_args.kwargs["test_cases"]
        assert len(test_cases_passed) == 3
        models_in_test_cases = {tc.metadata["model"] for tc in test_cases_passed}
        assert models_in_test_cases == {"gpt-4", "gpt-3.5", "claude-3"}

    @pytest.fixture
    def mock_prompts_data(self):
        """Mock prompts data for testing."""
        return [
            {"name": "alt_prompt_1", "text": "my prompt"},
            {"name": "langsmith_prompt_1"},
        ]

    @pytest.mark.asyncio
    async def test_prompt_with_name_and_text(
        self, orchestrator, mock_prompts_data, mocker
    ):
        """Test prompt extraction when entry has both name and text."""
        # Mock external dependencies only
        mock_load_json = mocker.patch(
            "ai_chatbots.evaluation.orchestrator.load_json_with_settings"
        )
        mock_load_json.return_value = {"test_bot": mock_prompts_data}

        mock_evaluator = mocker.Mock()
        mock_evaluator.load_cases.return_value = []
        mock_evaluator.evaluate_model = AsyncMock(return_value=[])

        orchestrator.reporter.generate_report = mocker.Mock()

        mocker.patch("ai_chatbots.evaluation.orchestrator.deepeval")
        mocker.patch(
            "ai_chatbots.evaluation.orchestrator.BOT_EVALUATORS",
            {"test_bot": (mocker.Mock(), mocker.Mock(return_value=mock_evaluator))},
        )

        config = orchestrator.create_evaluation_config(
            models=["gpt-4"], evaluation_model="gpt-4o"
        )
        config.confident_api_key = None

        await orchestrator.run_evaluation(
            config, bot_names=["test_bot"], use_prompts=True
        )
        calls = mock_evaluator.evaluate_model.call_args_list
        call = next(c for c in calls if c.kwargs["prompt_label"] == "alt_prompt_1")
        assert call.kwargs["instructions"] == "my prompt"

    @pytest.mark.asyncio
    async def test_prompt_with_name_only(self, orchestrator, mock_prompts_data, mocker):
        """Test prompt extraction when entry has only name, reqeuiring a call to langsmith for text."""
        # Mock external dependencies only
        mock_load_json = mocker.patch(
            "ai_chatbots.evaluation.orchestrator.load_json_with_settings"
        )
        mock_get_langsmith = mocker.patch(
            "ai_chatbots.evaluation.orchestrator.get_langsmith_prompt"
        )

        mock_load_json.return_value = {"test_bot": mock_prompts_data}
        mock_get_langsmith.return_value = "mocked langsmith text"

        mock_evaluator = mocker.Mock()
        mock_evaluator.load_cases.return_value = []
        mock_evaluator.evaluate_model = AsyncMock(return_value=[])

        orchestrator.reporter.generate_report = mocker.Mock()

        mocker.patch("ai_chatbots.evaluation.orchestrator.deepeval")
        mocker.patch(
            "ai_chatbots.evaluation.orchestrator.BOT_EVALUATORS",
            {"test_bot": (mocker.Mock(), mocker.Mock(return_value=mock_evaluator))},
        )

        config = orchestrator.create_evaluation_config(
            models=["gpt-4"], evaluation_model="gpt-4o"
        )
        config.confident_api_key = None

        await orchestrator.run_evaluation(
            config, bot_names=["test_bot"], use_prompts=True
        )

        mock_get_langsmith.assert_called_with("langsmith_prompt_1")
        calls = mock_evaluator.evaluate_model.call_args_list
        call = next(
            c for c in calls if c.kwargs["prompt_label"] == mock_prompts_data[1]["name"]
        )
        assert call.kwargs["instructions"] == mock_get_langsmith.return_value

    @pytest.mark.asyncio
    async def test_run_evaluation_with_data_file(self, orchestrator, mocker):
        """Test evaluation run with data_file parameter."""
        config = orchestrator.create_evaluation_config(
            models=["gpt-4"], evaluation_model="gpt-4o"
        )
        config.confident_api_key = None

        mock_evaluator_class = mocker.Mock()
        mock_evaluator = mocker.Mock()
        mock_evaluator.load_cases.return_value = []
        mock_evaluator.evaluate_model = AsyncMock(return_value=[])
        mock_evaluator_class.return_value = mock_evaluator
        mock_deepeval = mocker.patch("ai_chatbots.evaluation.orchestrator.deepeval")
        mocker.patch(
            "ai_chatbots.evaluation.orchestrator.BOT_EVALUATORS",
            {"recommendation": (ResourceRecommendationBot, mock_evaluator_class)},
        )
        from deepeval.evaluate.types import EvaluationResult

        mock_deepeval.evaluate.return_value = EvaluationResult(
            test_results=[], confident_link=None, test_run_id=None
        )
        orchestrator.reporter.generate_report = mocker.Mock()

        await orchestrator.run_evaluation(
            config,
            bot_names=["recommendation"],
            data_file="/path/to/custom_data.json",
        )

        mock_evaluator_class.assert_called_once_with(
            ResourceRecommendationBot,
            "recommendation",
            data_file="/path/to/custom_data.json",
            stdout=mocker.ANY,
            error_log_file=mocker.ANY,
        )


class TestEvaluationConfigIntegration:
    """Integration tests for evaluation configuration."""

    @pytest.fixture
    def mock_stdout(self, mocker):
        """Create mock stdout for testing."""
        return mocker.Mock()

    @pytest.fixture
    def orchestrator(self, mock_stdout):
        """Create orchestrator for testing."""
        return EvaluationOrchestrator(mock_stdout)

    def test_config_creation(self, mocker):
        """Test config creation with default reference-free metrics."""
        mock_stdout = mocker.Mock()
        orchestrator = EvaluationOrchestrator(mock_stdout)
        config = orchestrator.create_evaluation_config(
            models=["gpt-4o"], evaluation_model="judge-model"
        )

        assert isinstance(config, EvaluationConfig)
        assert len(config.metrics) == NUM_METRICS

        for metric in config.metrics:
            assert isinstance(metric, TimeoutMetricWrapper)
            # Verify the underlying metric is the expected reference-free type
            underlying_metric_names = [
                "FaithfulnessMetric",
                "ContextualRelevancyMetric",
                "HallucinationMetric",
                "AnswerRelevancyMetric",
                "GEval",
            ]
            assert metric.base_metric.__class__.__name__ in underlying_metric_names

    def test_create_evaluation_config_with_timeout(self, orchestrator):
        """Test evaluation config creation with timeout parameter."""
        models = ["gpt-4o"]
        evaluation_model = "judge-model"
        timeout_seconds = 120

        config = orchestrator.create_evaluation_config(
            models, evaluation_model, timeout_seconds=timeout_seconds
        )

        # Verify that metrics are wrapped with timeout functionality
        for metric in config.metrics:
            assert isinstance(metric, TimeoutMetricWrapper)
            assert metric.timeout_seconds == timeout_seconds

    def test_create_evaluation_config_default_timeout(self, orchestrator):
        """Test evaluation config creation uses default timeout."""
        models = ["gpt-4o"]
        evaluation_model = "judge-model"

        config = orchestrator.create_evaluation_config(models, evaluation_model)

        # Verify that metrics are wrapped with default timeout
        for metric in config.metrics:
            assert isinstance(metric, TimeoutMetricWrapper)
            assert metric.timeout_seconds == 360  # Default timeout

    @pytest.mark.asyncio
    async def test_run_evaluation_with_max_concurrent(
        self, orchestrator, mock_stdout, mock_evaluator, mocker
    ):
        """Test evaluation run with max_concurrent parameter."""
        config = orchestrator.create_evaluation_config(
            models=["gpt-4"], evaluation_model="gpt-4o"
        )
        config.confident_api_key = None

        mock_evaluator_class, _ = mock_evaluator
        mock_deepeval = mocker.patch("ai_chatbots.evaluation.orchestrator.deepeval")
        mocker.patch(
            "ai_chatbots.evaluation.orchestrator.BOT_EVALUATORS",
            {"test_bot": (mocker.Mock(), mock_evaluator_class)},
        )

        from deepeval.evaluate.types import EvaluationResult

        mock_results = mocker.Mock(spec=EvaluationResult)
        mock_results.test_results = [
            mocker.Mock(
                metadata={"model": "gpt-4", "bot_name": "test_bot"},
                metrics_data=[mocker.Mock()],
            )
        ]
        mock_results.confident_link = None
        mock_results.test_run_id = None
        mock_deepeval.evaluate.return_value = mock_results
        orchestrator.reporter.generate_report = mocker.Mock()
        await orchestrator.run_evaluation(
            config, bot_names=["test_bot"], max_concurrent=5
        )
        mock_deepeval.evaluate.assert_called_once()
        call_args = mock_deepeval.evaluate.call_args
        assert "async_config" in call_args.kwargs
        async_config = call_args.kwargs["async_config"]
        assert async_config.max_concurrent == 5
        mock_stdout.write.assert_called()
        calls = [call[0][0] for call in mock_stdout.write.call_args_list]
        output_text = " ".join(str(call) for call in calls)
        assert "max_concurrent=5" in output_text

    @pytest.mark.asyncio
    async def test_run_evaluation_with_error_config(
        self, orchestrator, mock_evaluator, mocker
    ):
        """Test evaluation run includes ErrorConfig with ignore_errors=True."""
        config = orchestrator.create_evaluation_config(
            models=["gpt-4"], evaluation_model="gpt-4o"
        )
        config.confident_api_key = None

        mock_evaluator_class, _ = mock_evaluator
        mock_deepeval = mocker.patch("ai_chatbots.evaluation.orchestrator.deepeval")
        mocker.patch(
            "ai_chatbots.evaluation.orchestrator.BOT_EVALUATORS",
            {"test_bot": (mocker.Mock(), mock_evaluator_class)},
        )
        from deepeval.evaluate.types import EvaluationResult

        mock_results = mocker.Mock(spec=EvaluationResult)
        mock_results.test_results = [
            mocker.Mock(
                metadata={"model": "gpt-4", "bot_name": "test_bot"},
                metrics_data=[mocker.Mock()],
            )
        ]
        mock_results.confident_link = None
        mock_results.test_run_id = None
        mock_deepeval.evaluate.return_value = mock_results
        orchestrator.reporter.generate_report = mocker.Mock()

        await orchestrator.run_evaluation(config, bot_names=["test_bot"])
        mock_deepeval.evaluate.assert_called_once()
        call_args = mock_deepeval.evaluate.call_args
        assert "error_config" in call_args.kwargs
        error_config = call_args.kwargs["error_config"]
        assert error_config.ignore_errors is True

    @pytest.mark.asyncio
    async def test_run_evaluation_empty_test_cases(
        self, orchestrator, mock_stdout, mock_evaluator, mocker
    ):
        """Test evaluation handles empty test cases gracefully."""
        config = orchestrator.create_evaluation_config(
            models=["gpt-4"], evaluation_model="gpt-4o"
        )
        config.confident_api_key = None

        mock_evaluator_class, mock_evaluator_instance = mock_evaluator
        mock_evaluator_instance.load_cases.return_value = []
        mock_evaluator_instance.evaluate_model = AsyncMock(return_value=[])
        mock_deepeval = mocker.patch("ai_chatbots.evaluation.orchestrator.deepeval")
        mocker.patch(
            "ai_chatbots.evaluation.orchestrator.BOT_EVALUATORS",
            {"test_bot": (mocker.Mock(), mock_evaluator_class)},
        )

        from deepeval.evaluate.types import EvaluationResult

        # This should not be called, but set it just in case
        mock_deepeval.evaluate.return_value = EvaluationResult(
            test_results=[], confident_link=None, test_run_id=None
        )
        orchestrator.reporter.generate_report = mocker.Mock()

        result = await orchestrator.run_evaluation(config, bot_names=["test_bot"])
        mock_deepeval.evaluate.assert_not_called()
        assert result.test_results == []
        assert result.confident_link is None
        mock_stdout.write.assert_called()
        # Note: With the new implementation, empty test cases don't trigger "No test cases available"
        # since the bot itself returns empty results


class TestPairwise:
    """Test cases for the --pairwise judge."""

    @pytest.fixture
    def orchestrator(self, mock_stdout):
        """Create orchestrator for testing."""
        return EvaluationOrchestrator(mock_stdout)

    @pytest.fixture
    def config(self, orchestrator):
        """Config comparing two models."""
        return orchestrator.create_evaluation_config(
            models=["model-a", "model-b"],
            evaluation_model="judge",
            timeout_seconds=30,
            max_retries=1,
            retry_delay=0.5,
        )

    @pytest.fixture
    def arena(self, mocker):
        """Patch the judge; it picks whichever winner the test sets."""
        arena_class = mocker.patch("ai_chatbots.evaluation.orchestrator.ArenaGEval")
        metric = mocker.Mock(reason="because", error=None)
        wrapper_class = mocker.patch(
            "ai_chatbots.evaluation.orchestrator.TimeoutMetricWrapper",
            return_value=metric,
        )
        return arena_class, metric, wrapper_class

    def test_run_pairwise_counts_wins(self, orchestrator, config, arena, mock_stdout):
        """Answers to the same question are grouped, and wins are tallied."""
        arena_class, metric, wrapper_class = arena
        metric.measure.side_effect = ["model-b", "model-a", "model-b"]
        cases = [
            _answer("model-a", "Q1"),
            _answer("model-b", "Q1"),
            _answer("model-a", "Q2"),
            _answer("model-b", "Q2"),
            # Same question for a different course is judged separately
            _answer("model-a", "Q1", case_id="Q1@c2"),
            _answer("model-b", "Q1", case_id="Q1@c2"),
        ]

        orchestrator._run_pairwise("syllabus", cases, config)  # noqa: SLF001

        assert metric.measure.call_count == 3
        arenas = [call.args[0] for call in metric.measure.call_args_list]
        assert [sorted(c.name for c in arena.contestants) for arena in arenas] == [
            ["model-a", "model-b"]
        ] * 3
        assert orchestrator.pairwise_wins == {"syllabus": {"model-a": 1, "model-b": 2}}
        assert arena_class.call_args.kwargs["model"] == "judge"
        # The judge gets the run's timeout and retries, and its steps once
        assert (config.timeout_seconds, config.max_retries, config.retry_delay) == (
            30,
            1,
            0.5,
        )
        wrapper_class.assert_called_once_with(
            arena_class.return_value,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
            retry_base_delay=config.retry_delay,
        )
        metric.prepare.assert_called_once_with()
        assert arena_class.call_args.kwargs["evaluation_params"] == [
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ]
        output = " ".join(str(c.args[0]) for c in mock_stdout.write.call_args_list)
        assert "Pairwise wins for syllabus: {'model-a': 1, 'model-b': 2}" in output

    def test_run_pairwise_by_side(self, orchestrator, config, arena):
        """A baseline run's contestants and winners are sides: model plus prompt."""
        metric = arena[1]
        metric.measure.return_value = "model-b [short]"
        cases = [_answer("model-a"), _answer("model-b")]
        cases[1].metadata["prompt_label"] = "short"

        orchestrator._run_pairwise("syllabus", cases, config, by_side=True)  # noqa: SLF001

        contestants = metric.measure.call_args.args[0].contestants
        assert [c.name for c in contestants] == ["model-a", "model-b [short]"]
        assert orchestrator.pairwise_results == {
            ("syllabus", "f.json#Q1", None): "model-b [short]"
        }

    def test_run_pairwise_skips_unmatched_and_judge_errors(
        self, orchestrator, config, arena, mock_stdout
    ):
        """A question only one model answered is skipped; judge errors don't stop
        the run.
        """
        _, metric, _ = arena
        outcomes = iter([None, "model-a", "model-b"])

        def measure(_arena):
            # The wrapper returns 0.0 and sets error once its retries run out
            winner = next(outcomes)
            metric.error = None if winner else "judge down"
            return winner or 0.0

        metric.measure.side_effect = measure
        reprompted = [_answer(m, "Q2") for m in ("model-a", "model-b")]
        for case in reprompted:
            case.metadata["prompt_label"] = "#1"
        cases = [
            _answer("model-a", "Q1"),
            _answer("model-b", "Q1"),
            _answer("model-a", "Q2"),
            _answer("model-b", "Q2"),
            *reprompted,
            _answer("model-a", "Only one"),
        ]

        orchestrator._run_pairwise("syllabus", cases, config)  # noqa: SLF001

        assert orchestrator.pairwise_wins == {"syllabus": {"model-a": 1, "model-b": 1}}
        assert orchestrator.pairwise_results == {
            ("syllabus", "f.json#Q1", "default"): None,
            ("syllabus", "f.json#Q2", "default"): "model-a",
            ("syllabus", "f.json#Q2", "#1"): "model-b",
        }
        assert [(f.kind, f.case_id) for f in orchestrator.failures.items] == [
            ("pairwise judge error", "f.json#Q1"),
            ("pairwise skipped: an answer is missing", "f.json#Only one"),
        ]
        output = " ".join(str(c.args[0]) for c in mock_stdout.write.call_args_list)
        assert "judge error: judge down" in output

    def test_run_pairwise_skipped(self, orchestrator, arena, mock_stdout):
        """Single-model runs are skipped."""
        _, metric, _ = arena
        config = orchestrator.create_evaluation_config(
            models=["model-a"], evaluation_model="judge"
        )

        orchestrator._run_pairwise("syllabus", [_answer("model-a", "Q1")], config)  # noqa: SLF001

        metric.measure.assert_not_called()
        assert orchestrator.pairwise_wins == {}
        output = " ".join(str(c.args[0]) for c in mock_stdout.write.call_args_list)
        assert "--pairwise needs two or more models" in output

    @pytest.mark.parametrize("pairwise", [True, False])
    @pytest.mark.asyncio
    async def test_run_evaluation_pairwise_flag(
        self, orchestrator, config, mock_stdout, mocker, pairwise
    ):
        """run_evaluation passes every model's cases to the judge only when
        asked.
        """
        cases = {m: _answer(m, "Q1") for m in config.models}
        evaluator = mocker.Mock()
        evaluator.load_cases.return_value = [mocker.Mock()]
        evaluator.evaluate_model = AsyncMock(
            side_effect=lambda model, *_args, **_kwargs: [cases[model]]
        )
        mocker.patch(
            "ai_chatbots.evaluation.orchestrator.BOT_EVALUATORS",
            {"test_bot": (mocker.Mock(), mocker.Mock(return_value=evaluator))},
        )
        mock_deepeval = mocker.patch("ai_chatbots.evaluation.orchestrator.deepeval")
        from deepeval.evaluate.types import EvaluationResult

        mock_deepeval.evaluate.return_value = EvaluationResult(
            test_results=[], confident_link=None, test_run_id=None
        )
        orchestrator.reporter.generate_report = mocker.Mock()

        def fake_pairwise(bot_name, _test_cases, _config, **_kwargs):
            orchestrator.pairwise_wins[bot_name] = {"model-a": 1, "model-b": 0}

        run_pairwise = mocker.patch.object(
            orchestrator, "_run_pairwise", side_effect=fake_pairwise
        )

        await orchestrator.run_evaluation(
            config, bot_names=["test_bot"], use_prompts=False, pairwise=pairwise
        )

        output = " ".join(str(c.args[0]) for c in mock_stdout.write.call_args_list)
        if pairwise:
            run_pairwise.assert_called_once_with(
                "test_bot", list(cases.values()), config, by_side=False
            )
            assert "PAIRWISE WINS" in output
        else:
            run_pairwise.assert_not_called()
            assert "PAIRWISE WINS" not in output

    def test_evaluate_prepares_metrics(
        self, orchestrator, config, mock_prepare, mocker
    ):
        """Every metric is prepared (GEval steps generated) before scoring."""
        mock_deepeval = mocker.patch("ai_chatbots.evaluation.orchestrator.deepeval")
        mock_deepeval.evaluate.return_value = mocker.Mock(test_results=[1])

        orchestrator._run_deepeval_evaluation(  # noqa: SLF001
            [_answer("model-a", "Q1")], config, 2
        )

        assert mock_prepare.call_count == len(config.metrics)

    def test_evaluate_warns_when_deepeval_drops_cases(
        self, orchestrator, config, mock_stdout, mocker
    ):
        """A warning is printed when deepeval returns fewer results than cases."""
        mock_deepeval = mocker.patch("ai_chatbots.evaluation.orchestrator.deepeval")
        mock_deepeval.evaluate.return_value = mocker.Mock(test_results=[1])

        orchestrator._run_deepeval_evaluation(  # noqa: SLF001
            [_answer("model-a", "Q1"), _answer("model-a", "Q2")], config, 2
        )

        output = " ".join(str(c.args[0]) for c in mock_stdout.write.call_args_list)
        assert "deepeval returned 1 of 2 test cases" in output


class TestFailuresAndBaseline:
    """Failures are recorded in every run; a baseline adds a verdict."""

    MODELS = ["model-a", "model-b"]

    @pytest.fixture
    def orchestrator(self, mock_stdout):
        """Create orchestrator for testing."""
        return EvaluationOrchestrator(mock_stdout)

    @pytest.fixture
    def bot(self, mocker):
        """Register "test_bot"; each model answers the cases in bot.answers."""
        evaluator = mocker.Mock()
        evaluator.answers = {"Q1": "f.json#Q1"}
        evaluator.load_cases.side_effect = lambda **_: (
            [mocker.Mock()] * len(evaluator.answers)
        )

        def answers(model, *_args, prompt_label="default", **_kwargs):
            found = [_answer(model, q, c) for q, c in evaluator.answers.items()]
            for answer in found:
                answer.metadata["prompt_label"] = prompt_label
            return found

        evaluator.evaluate_model = AsyncMock(side_effect=answers)
        mocker.patch(
            "ai_chatbots.evaluation.orchestrator.BOT_EVALUATORS",
            {"test_bot": (mocker.Mock(), mocker.Mock(return_value=evaluator))},
        )
        return evaluator

    @pytest.fixture
    def deepeval_returns(self, mocker):
        """Make deepeval.evaluate return results for the given answers."""
        mock_deepeval = mocker.patch("ai_chatbots.evaluation.orchestrator.deepeval")

        def returns(*results):
            mock_deepeval.evaluate.return_value = mocker.Mock(
                test_results=[
                    mocker.Mock(metadata=dict(case.metadata), metrics_data=metrics)
                    for case, metrics in results
                ]
            )
            return mock_deepeval

        return returns

    def _config(self, orchestrator, **kwargs):
        return orchestrator.create_evaluation_config(self.MODELS, "judge", **kwargs)

    @staticmethod
    def _metric(mocker, score, error=None, name="Correctness [GEval]"):
        metric = mocker.Mock(score=score, error=error)
        metric.name = name
        return metric

    @pytest.mark.parametrize(
        ("judge", "models"),
        [
            ("gpt-5.4", ["openai/gpt-5.4", "openai/gpt-4o-mini"]),
            ("openai/gpt-5.4", ["gpt-5.4"]),
            ("GPT-5.4", ["openai/gpt-5.4"]),
        ],
    )
    def test_judge_under_test_is_refused(self, orchestrator, judge, models):
        """The judge can't be a model under test, with or without its prefix."""
        with pytest.raises(ValueError, match="is one of the models under test"):
            orchestrator.create_evaluation_config(models, judge)

    @pytest.mark.usefixtures("bot")
    async def test_pairwise_only(self, orchestrator, deepeval_returns, mocker):
        """--pairwise-only builds no metric and never calls deepeval.evaluate."""
        config = self._config(orchestrator, pairwise_only=True)
        mock_deepeval = deepeval_returns()
        mocker.patch.object(orchestrator, "_run_pairwise")
        orchestrator.reporter.generate_report = mocker.Mock()

        await orchestrator.run_evaluation(
            config, bot_names=["test_bot"], use_prompts=False, pairwise=True
        )

        assert (config.metrics, config.pairwise_only, config.correctness_only) == (
            [],
            True,
            False,
        )
        mock_deepeval.evaluate.assert_not_called()

    def test_record_answers(self, orchestrator):
        """Empty answers, search errors and blocked account calls are recorded."""
        empty, searched, blocked = (
            _answer("model-a", case_id=f"f#{i}") for i in range(3)
        )
        # Only a metadata comment: still empty
        empty.actual_output = '\n\n<!-- {"thread_id": "1"} -->\n\n'
        searched.metadata["search_errors"] = ["search_content_files"]
        blocked.metadata["blocked_account_calls"] = ["get_enrollments"]

        orchestrator._record_answers("syllabus", [empty, searched, blocked])  # noqa: SLF001

        assert [(f.kind, f.case_id) for f in orchestrator.failures.items] == [
            ("empty answer", "f#0"),
            ("search tool returned an error", "f#1"),
            ("account tool called without made-up records", "f#2"),
        ]
        record = next(iter(orchestrator.answer_records.values()))
        assert set(record["checks"]) == {"follow_up_offer"}
        # Each answer is saved with what the judge is shown
        saved = orchestrator.stdout.write.call_args.args[0]
        assert saved.startswith("ANSWER_JSON ")
        assert '"context": ["evidence"]' in saved

    def test_scores_recorded_under_threshold_names(
        self, orchestrator, deepeval_returns, mocker
    ):
        """Deepeval's "Answer Relevancy" is scored as the threshold's AnswerRelevancy."""
        config = self._config(orchestrator, require_expected=True)
        case = _answer("model-a")
        orchestrator._record_answers("test_bot", [case])  # noqa: SLF001
        names = [metric.__name__ for metric in config.metrics]
        deepeval_returns((case, [self._metric(mocker, 0.9, name=n) for n in names]))

        orchestrator._run_deepeval_evaluation([case], config, 2, bot_name="test_bot")  # noqa: SLF001

        (record,) = orchestrator.answer_records.values()
        assert set(record["scores"]) == set(config.metric_thresholds)

    @pytest.mark.parametrize(
        ("metric", "kinds"),
        [
            ((0.9, None), ["deepeval returned no result"]),  # model-b's is lost
            ((None, "timed out"), ["judge error", "deepeval returned no result"]),
            ((0.0, "timed out"), ["judge error", "deepeval returned no result"]),
            ((None, None), ["judge error", "deepeval returned no result"]),
            (None, ["judge returned no score", "deepeval returned no result"]),
        ],
    )
    def test_deepeval_failures_recorded(
        self, orchestrator, deepeval_returns, mocker, metric, kinds
    ):
        """A missing result or score and a judge error are failures, not scores."""
        config = self._config(orchestrator, correctness_only=True)
        cases = [_answer(model) for model in self.MODELS]
        deepeval_returns((cases[0], [self._metric(mocker, *metric)] if metric else []))

        orchestrator._run_deepeval_evaluation(cases, config, 2, bot_name="test_bot")  # noqa: SLF001

        assert [f.kind for f in orchestrator.failures.items] == kinds

    async def test_baseline_comparison(
        self, orchestrator, bot, deepeval_returns, mock_stdout, mocker
    ):
        """A baseline run answers once per side, with its prompt, then compares."""
        bot.answers = {f"Q{i}": f"f#{i}" for i in range(8)}
        sides = [("model-a", "default", 0.9), ("model-b", "short", 0.4)]
        deepeval_returns(
            *(
                (answer, [self._metric(mocker, score)])
                for model, label, score in sides
                for answer in bot.evaluate_model.side_effect(model, prompt_label=label)
            )
        )

        def baseline_wins(bot_name, _cases, _config, **_kwargs):
            for case_id in bot.answers.values():
                orchestrator.pairwise_results[(bot_name, case_id, "default")] = (
                    "model-a"
                )

        mocker.patch.object(orchestrator, "_run_pairwise", side_effect=baseline_wins)
        orchestrator.reporter.generate_report = mocker.Mock()

        await orchestrator.run_evaluation(
            self._config(orchestrator, correctness_only=True),
            bot_names=["test_bot"],
            baseline="model-a",
            candidate_prompt={"name": "short", "text": "Be short."},
            sample=0,
        )

        calls = bot.evaluate_model.call_args_list
        assert [(c.args[0], c.kwargs["instructions"]) for c in calls] == [
            ("model-a", None),
            ("model-b", "Be short."),
        ]
        output = " ".join(str(c.args[0]) for c in mock_stdout.write.call_args_list)
        for text in (
            "model-b [short] versus baseline model-a, on 8 cases",
            "pairwise: better 0, worse 8, same 0; p=0.008; 🚩 CLEARLY WORSE",
            "Correctness: better 0, worse 8, same 0",
            "↳ worse: f#0: Q0",
            "Regression found: clearly worse on pairwise, Correctness.",
        ):
            assert text in output
        assert "INCOMPLETE RUN" not in output

    @pytest.mark.parametrize(
        ("named", "recorded"), [(["test_bot"], True), (None, False)]
    )
    async def test_no_cases_loaded(self, orchestrator, bot, mocker, named, recorded):
        """A bot asked for by name that loads no cases is a failure."""
        bot.answers = {}
        mocker.patch.object(orchestrator, "_run_pairwise")
        orchestrator.reporter.generate_report = mocker.Mock()

        await orchestrator.run_evaluation(
            self._config(orchestrator, pairwise_only=True),
            bot_names=named,
            data_file="x.json",
            use_prompts=False,
        )

        found = [(f.bot, f.kind, f.detail) for f in orchestrator.failures.items]
        assert found == (
            [("test_bot", "no cases loaded", "x.json")] if recorded else []
        )
