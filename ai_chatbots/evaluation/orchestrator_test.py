"""Unit tests for evaluation orchestrator."""

import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from .base import EvaluationConfig
from .orchestrator import EvaluationOrchestrator

NUM_METRICS = 6


class TestEvaluationOrchestrator:
    """Test cases for EvaluationOrchestrator."""

    @pytest.fixture
    def mock_stdout(self):
        """Create mock stdout for testing."""
        return Mock()

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
        # API key might be set by default in the environment
        assert config.confident_api_key is not None or config.confident_api_key is None

    def test_create_evaluation_config_custom_thresholds(self, orchestrator):
        """Test evaluation config creation with custom thresholds."""
        models = ["gpt-4o"]
        evaluation_model = "gpt-4o"
        custom_thresholds = {
            "ContextualPrecision": 0.9,
            "ContextualRelevancy": 0.8,
            "ContextualRecall": 0.9,
            "Faithfulness": 0.8,
            "Hallucination": 0.1,
            "AnswerRelevancy": 0.8,
        }

        config = orchestrator.create_evaluation_config(
            models, evaluation_model, custom_thresholds
        )

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
    @patch("ai_chatbots.evaluation.orchestrator.deepeval")
    async def test_run_evaluation_success(self, mock_deepeval, orchestrator):
        """Test successful evaluation run."""
        # Mock config
        config = Mock()
        config.confident_api_key = "test-key"
        config.models = ["gpt-4"]
        config.metrics = [Mock()]

        # Mock BOT_EVALUATORS
        mock_evaluator_class = Mock()
        mock_evaluator = Mock()
        mock_evaluator.load_test_cases.return_value = [Mock()]
        mock_evaluator.evaluate_model = AsyncMock(return_value=[Mock()])
        mock_evaluator_class.return_value = mock_evaluator

        with patch(
            "ai_chatbots.evaluation.orchestrator.BOT_EVALUATORS",
            {"test_bot": (Mock(), mock_evaluator_class)},
        ):
            # Mock deepeval.evaluate
            mock_results = Mock()
            mock_deepeval.evaluate.return_value = mock_results

            # Mock reporter
            orchestrator.reporter.generate_report = Mock()

            # Run evaluation
            result = await orchestrator.run_evaluation(config, ["test_bot"])

            # Verify deepeval was configured
            mock_deepeval.login_with_confident_api_key.assert_called_once_with(
                "test-key"
            )

            # Verify evaluation was run
            mock_deepeval.evaluate.assert_called_once()

            # Verify report was generated
            orchestrator.reporter.generate_report.assert_called_once_with(
                mock_results, ["gpt-4"], ["test_bot"]
            )

            assert result == mock_results

    @pytest.mark.asyncio
    @patch("ai_chatbots.evaluation.orchestrator.deepeval")
    async def test_run_evaluation_no_api_key(self, mock_deepeval, orchestrator):
        """Test evaluation run without API key."""
        config = Mock()
        config.confident_api_key = None
        config.models = ["gpt-4"]
        config.metrics = [Mock()]

        with patch("ai_chatbots.evaluation.orchestrator.BOT_EVALUATORS", {}):
            mock_deepeval.evaluate.return_value = Mock()
            orchestrator.reporter.generate_report = Mock()

            await orchestrator.run_evaluation(config, [])

            # Verify login was not called
            mock_deepeval.login_with_confident_api_key.assert_not_called()

    @pytest.mark.asyncio
    @patch("ai_chatbots.evaluation.orchestrator.deepeval")
    async def test_run_evaluation_default_bots(self, mock_deepeval, orchestrator):
        """Test evaluation run with default bot names."""
        config = Mock()
        config.confident_api_key = None
        config.models = ["gpt-4"]
        config.metrics = [Mock()]

        # Mock all default bots
        mock_evaluators = {}
        for bot_name in ["recommendation", "syllabus", "video_gpt", "tutor"]:
            mock_evaluator_class = Mock()
            mock_evaluator = Mock()
            mock_evaluator.load_test_cases.return_value = []
            mock_evaluator.evaluate_model = AsyncMock(return_value=[])
            mock_evaluator_class.return_value = mock_evaluator
            mock_evaluators[bot_name] = (Mock(), mock_evaluator_class)

        with patch(
            "ai_chatbots.evaluation.orchestrator.BOT_EVALUATORS", mock_evaluators
        ):
            mock_deepeval.evaluate.return_value = Mock()
            orchestrator.reporter.generate_report = Mock()

            # Run evaluation with no bot_names (should use all available)
            await orchestrator.run_evaluation(config, None)

            # Verify all bots were processed
            assert len(mock_evaluators) == 4

    @pytest.mark.asyncio
    @patch("ai_chatbots.evaluation.orchestrator.deepeval")
    async def test_run_evaluation_unknown_bot(
        self, mock_deepeval, orchestrator, mock_stdout
    ):
        """Test evaluation run with unknown bot name."""
        config = Mock()
        config.confident_api_key = None
        config.models = ["gpt-4"]
        config.metrics = [Mock()]

        with patch("ai_chatbots.evaluation.orchestrator.BOT_EVALUATORS", {}):
            mock_deepeval.evaluate.return_value = Mock()
            orchestrator.reporter.generate_report = Mock()

            await orchestrator.run_evaluation(config, ["unknown_bot"])

            # Verify warning was written
            mock_stdout.write.assert_called()

    @pytest.mark.asyncio
    @patch("ai_chatbots.evaluation.orchestrator.deepeval")
    async def test_run_evaluation_evaluator_error(
        self, mock_deepeval, orchestrator, mock_stdout
    ):
        """Test evaluation run with evaluator error."""
        config = Mock()
        config.confident_api_key = None
        config.models = ["gpt-4"]
        config.metrics = [Mock()]

        # Mock evaluator that raises exception
        mock_evaluator_class = Mock()
        mock_evaluator = Mock()
        mock_evaluator.load_test_cases.return_value = [Mock()]
        mock_evaluator.evaluate_model = AsyncMock(side_effect=Exception("Test error"))
        mock_evaluator_class.return_value = mock_evaluator

        with patch(
            "ai_chatbots.evaluation.orchestrator.BOT_EVALUATORS",
            {"test_bot": (Mock(), mock_evaluator_class)},
        ):
            mock_deepeval.evaluate.return_value = Mock()
            orchestrator.reporter.generate_report = Mock()

            # Run evaluation - should handle error gracefully
            _ = await orchestrator.run_evaluation(config, ["test_bot"])

            # Verify error was logged
            mock_stdout.write.assert_called()

            # Verify evaluation still completed (with empty test cases)
            mock_deepeval.evaluate.assert_called_once()

    @pytest.mark.asyncio
    @patch("ai_chatbots.evaluation.orchestrator.deepeval")
    async def test_run_evaluation_multiple_models(self, mock_deepeval, orchestrator):
        """Test evaluation run with multiple models."""
        config = Mock()
        config.confident_api_key = None
        config.models = ["gpt-4", "gpt-3.5", "claude-3"]
        config.metrics = [Mock()]

        # Mock evaluator
        mock_evaluator_class = Mock()
        mock_evaluator = Mock()
        mock_evaluator.load_test_cases.return_value = [Mock()]

        # Track calls to evaluate_model
        evaluate_calls = []

        async def mock_evaluate_model(
            model, test_cases, instructions=None, prompt_label="default"
        ):
            evaluate_calls.append(model)
            return [Mock()]

        mock_evaluator.evaluate_model = mock_evaluate_model
        mock_evaluator_class.return_value = mock_evaluator

        with patch(
            "ai_chatbots.evaluation.orchestrator.BOT_EVALUATORS",
            {"test_bot": (Mock(), mock_evaluator_class)},
        ):
            mock_deepeval.evaluate.return_value = Mock()
            orchestrator.reporter.generate_report = Mock()

            await orchestrator.run_evaluation(config, ["test_bot"])

            # Verify all models were evaluated
            assert set(evaluate_calls) == {"gpt-4", "gpt-3.5", "claude-3"}


class TestEvaluationConfigIntegration:
    """Integration tests for evaluation configuration."""

    def test_config_with_real_metrics(self):
        """Test config creation with real DeepEval metrics."""

        mock_stdout = Mock()
        orchestrator = EvaluationOrchestrator(mock_stdout)

        config = orchestrator.create_evaluation_config(
            models=["gpt-4o"], evaluation_model="gpt-4o"
        )

        # Verify metrics are properly instantiated
        assert len(config.metrics) == NUM_METRICS
        metric_names = [metric.__class__.__name__ for metric in config.metrics]
        expected_names = [
            "ContextualPrecisionMetric",
            "ContextualRelevancyMetric",
            "ContextualRecallMetric",
            "FaithfulnessMetric",
            "HallucinationMetric",
            "AnswerRelevancyMetric",
        ]
        assert set(metric_names) == set(expected_names)
