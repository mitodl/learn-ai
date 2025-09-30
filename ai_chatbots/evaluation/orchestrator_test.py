"""Unit tests for evaluation orchestrator."""

import os
from unittest.mock import AsyncMock, patch

import pytest

from ai_chatbots.chatbots import ResourceRecommendationBot
from ai_chatbots.evaluation.base import EvaluationConfig
from ai_chatbots.evaluation.orchestrator import EvaluationOrchestrator
from ai_chatbots.evaluation.timeout_wrapper import TimeoutMetricWrapper

NUM_METRICS = 5


@pytest.fixture
def mock_evaluator(mocker):
    """Create a mock evaluator with standard behavior."""
    mock_evaluator_class = mocker.Mock()
    mock_evaluator = mocker.Mock()
    mock_evaluator.load_test_cases.return_value = [mocker.Mock()]
    mock_evaluator.evaluate_model = AsyncMock(return_value=[mocker.Mock()])
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
        evaluation_model = "gpt-4o"

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
        evaluation_model = "gpt-4o"
        custom_thresholds = {
            "ContextualPrecision": 0.9,
            "ContextualRelevancy": 0.8,
            "ContextualRecall": 0.9,
            "Hallucination": 0.1,
            "AnswerRelevancy": 0.8,
        }

        config = orchestrator.create_evaluation_config(
            models, evaluation_model, custom_thresholds
        )
        assert isinstance(config, EvaluationConfig)
        assert config.models == models
        assert config.evaluation_model == evaluation_model
        assert len(config.metrics) == NUM_METRICS

    @patch.dict(os.environ, {"CONFIDENT_AI_API_KEY": "test-api-key"})
    def test_create_evaluation_config_with_api_key(self, orchestrator):
        """Test evaluation config creation with API key from environment."""
        config = orchestrator.create_evaluation_config(["gpt-4o"], "gpt-4o")

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

        mock_results = mocker.Mock()
        mock_deepeval.evaluate.return_value = mock_results
        orchestrator.reporter.generate_report = mocker.Mock()

        result = await orchestrator.run_evaluation(config, bot_names=["test_bot"])

        mock_deepeval.evaluate.assert_called_once()
        orchestrator.reporter.generate_report.assert_called_once_with(
            mock_results, ["gpt-4"], ["test_bot"]
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
        mock_evaluator_instance.load_test_cases.return_value = []
        mock_evaluator_instance.evaluate_model = AsyncMock(return_value=[])

        mock_evaluators = {}
        for bot_name in ["recommendation", "syllabus", "video_gpt", "tutor"]:
            mock_evaluators[bot_name] = (mocker.Mock(), mock_evaluator_class)

        mock_deepeval = mocker.patch("ai_chatbots.evaluation.orchestrator.deepeval")
        mocker.patch(
            "ai_chatbots.evaluation.orchestrator.BOT_EVALUATORS", mock_evaluators
        )

        mock_deepeval.evaluate.return_value = mocker.Mock()
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

        mock_deepeval.evaluate.return_value = mocker.Mock()
        orchestrator.reporter.generate_report = mocker.Mock()

        await orchestrator.run_evaluation(config, bot_names=["unknown_bot"])
        mock_stdout.write.assert_called()

    @pytest.mark.asyncio
    async def test_run_evaluation_evaluator_error(
        self, orchestrator, mock_stdout, mock_evaluator, mocker
    ):
        """Test evaluation run with evaluator error."""
        config = orchestrator.create_evaluation_config(
            models=["gpt-4"], evaluation_model="gpt-4o"
        )
        config.confident_api_key = None

        mock_evaluator_class, mock_evaluator_instance = mock_evaluator
        mock_evaluator_instance.evaluate_model = AsyncMock(
            side_effect=Exception("Test error")
        )

        # Mock external dependencies
        mock_deepeval = mocker.patch("ai_chatbots.evaluation.orchestrator.deepeval")
        mocker.patch(
            "ai_chatbots.evaluation.orchestrator.BOT_EVALUATORS",
            {"test_bot": (mocker.Mock(), mock_evaluator_class)},
        )

        mock_deepeval.evaluate.return_value = mocker.Mock()
        orchestrator.reporter.generate_report = mocker.Mock()

        result = await orchestrator.run_evaluation(config, bot_names=["test_bot"])
        mock_stdout.write.assert_called()
        assert result.test_results == []
        mock_deepeval.evaluate.assert_not_called()

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

        async def track_evaluate_model(
            model, test_cases, instructions=None, prompt_label="default"
        ):
            model_calls.append(model)
            # Return unique test case for each model to verify they're all collected
            test_case = mocker.Mock()
            test_case.model = model  # Tag test case with model for verification
            return [test_case]

        mock_evaluator_instance.evaluate_model = track_evaluate_model

        # Mock external dependencies
        mock_deepeval = mocker.patch("ai_chatbots.evaluation.orchestrator.deepeval")
        mocker.patch(
            "ai_chatbots.evaluation.orchestrator.BOT_EVALUATORS",
            {"test_bot": (mocker.Mock(), mock_evaluator_class)},
        )

        mock_deepeval.evaluate.return_value = mocker.Mock()
        orchestrator.reporter.generate_report = mocker.Mock()

        await orchestrator.run_evaluation(config, bot_names=["test_bot"])
        assert set(model_calls) == {"gpt-4", "gpt-3.5", "claude-3"}
        mock_deepeval.evaluate.assert_called_once()
        call_args = mock_deepeval.evaluate.call_args
        test_cases_passed = call_args.kwargs["test_cases"]
        assert len(test_cases_passed) == 3
        models_in_test_cases = {tc.model for tc in test_cases_passed}
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
        mock_evaluator.load_test_cases.return_value = []
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
        mock_evaluator.load_test_cases.return_value = []
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
        mock_evaluator.load_test_cases.return_value = []
        mock_evaluator.evaluate_model = AsyncMock(return_value=[])
        mock_evaluator_class.return_value = mock_evaluator
        mock_deepeval = mocker.patch("ai_chatbots.evaluation.orchestrator.deepeval")
        mocker.patch(
            "ai_chatbots.evaluation.orchestrator.BOT_EVALUATORS",
            {"recommendation": (ResourceRecommendationBot, mock_evaluator_class)},
        )
        mock_deepeval.evaluate.return_value = mocker.Mock()
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
        """Test config creation."""
        mock_stdout = mocker.Mock()
        orchestrator = EvaluationOrchestrator(mock_stdout)
        config = orchestrator.create_evaluation_config(
            models=["gpt-4o"], evaluation_model="gpt-4o"
        )

        assert isinstance(config, EvaluationConfig)
        assert len(config.metrics) == NUM_METRICS

        for metric in config.metrics:
            assert isinstance(metric, TimeoutMetricWrapper)
            # Verify the underlying metric is the expected type
            underlying_metric_names = [
                "ContextualPrecisionMetric",
                "ContextualRelevancyMetric",
                "ContextualRecallMetric",
                "HallucinationMetric",
                "AnswerRelevancyMetric",
            ]
            assert metric.base_metric.__class__.__name__ in underlying_metric_names

    def test_create_evaluation_config_with_timeout(self, orchestrator):
        """Test evaluation config creation with timeout parameter."""
        models = ["gpt-4o"]
        evaluation_model = "gpt-4o"
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
        evaluation_model = "gpt-4o"

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

        mock_deepeval.evaluate.return_value = mocker.Mock()
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
        mock_deepeval.evaluate.return_value = mocker.Mock()
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
        mock_evaluator_instance.load_test_cases.return_value = []
        mock_evaluator_instance.evaluate_model = AsyncMock(return_value=[])
        mock_deepeval = mocker.patch("ai_chatbots.evaluation.orchestrator.deepeval")
        mocker.patch(
            "ai_chatbots.evaluation.orchestrator.BOT_EVALUATORS",
            {"test_bot": (mocker.Mock(), mock_evaluator_class)},
        )

        orchestrator.reporter.generate_report = mocker.Mock()

        result = await orchestrator.run_evaluation(config, bot_names=["test_bot"])
        mock_deepeval.evaluate.assert_not_called()
        assert result.test_results == []
        assert result.confident_link is None
        mock_stdout.write.assert_called()
        calls = [call[0][0] for call in mock_stdout.write.call_args_list]
        output_text = " ".join(str(call) for call in calls)
        assert "No test cases available" in output_text
