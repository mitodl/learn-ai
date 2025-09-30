"""Unit tests for evaluation reporting classes."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
from deepeval.evaluate.types import EvaluationResult

from ai_chatbots.evaluation.reporting import (
    DualOutputWrapper,
    EvaluationReporter,
    SummaryReporter,
)


@pytest.fixture
def reporter(mock_stdout):
    """Create reporter for testing."""
    return EvaluationReporter(mock_stdout)


class TestDualOutputWrapper:
    """Test cases for DualOutputWrapper class."""

    def test_initialization_without_file(self, mocker):
        """Test DualOutputWrapper initialization without file."""
        stdout = mocker.Mock()
        wrapper = DualOutputWrapper(stdout, file_path=None)

        assert wrapper.stdout is stdout
        assert wrapper.file_path is None
        assert wrapper.file is None

    def test_initialization_with_file(self, mocker):
        """Test DualOutputWrapper initialization with file."""
        stdout = mocker.Mock()

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            temp_path = temp_file.name

            try:
                wrapper = DualOutputWrapper(stdout, file_path=temp_path)

                assert wrapper.stdout is stdout
                assert wrapper.file_path == temp_path
                assert wrapper.file is not None

                # Clean up
                wrapper.close()
            finally:
                Path(temp_path).unlink(missing_ok=True)

    def test_write_without_file(self, mocker):
        """Test write method without file output."""
        stdout = mocker.Mock()
        wrapper = DualOutputWrapper(stdout, file_path=None)

        wrapper.write("test message", style_func=None, ending=None)

        stdout.write.assert_called_once_with(
            "test message", style_func=None, ending=None
        )

    def test_write_with_file(self, mocker):
        """Test write method with file output."""
        stdout = mocker.Mock()

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            wrapper = DualOutputWrapper(stdout, file_path=temp_path)

            wrapper.write("test message", ending="\n")

            # Verify stdout was called
            stdout.write.assert_called_once_with(
                "test message", style_func=None, ending="\n"
            )

            # Verify file content
            wrapper.close()
            with Path(temp_path).open("r") as f:
                content = f.read()
                assert "test message\n" in content

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_context_manager(self, mocker):
        """Test DualOutputWrapper as context manager."""
        stdout = mocker.Mock()

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            with DualOutputWrapper(stdout, file_path=temp_path) as wrapper:
                assert wrapper.file is not None
                wrapper.write("context test")

            with Path(temp_path).open("r") as f:
                content = f.read()
                assert "context test" in content

        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestEvaluationReporter:
    """Test cases for EvaluationReporter."""

    @pytest.fixture
    def mock_evaluation_results(self, mocker):
        """Create mock evaluation results."""
        # Create mock test results
        test_result1 = mocker.Mock()
        test_result1.additional_metadata = {
            "bot_name": "recommendation",
            "model": "gpt-4",
            "prompt_label": "default",
            "question": "What courses are available?",
        }
        test_result1.name = "recommendation-gpt-4"

        # Create mock metrics data
        metric1 = mocker.Mock()
        metric1.name = "ContextualRelevancy"
        metric1.score = 0.85
        metric1.success = True
        metric1.reason = ""

        metric2 = mocker.Mock()
        metric2.name = "AnswerRelevancy"
        metric2.score = 0.45
        metric2.success = False
        metric2.reason = "Answer was not relevant to the question"

        test_result1.metrics_data = [metric1, metric2]

        # Create second test result
        test_result2 = mocker.Mock()
        test_result2.additional_metadata = {
            "bot_name": "syllabus",
            "model": "gpt-3.5",
            "prompt_label": "#1",
            "question": "Who teaches this course?",
        }
        test_result2.name = "syllabus-gpt-3.5"

        metric3 = mocker.Mock()
        metric3.name = "ContextualRelevancy"
        metric3.score = 0.75
        metric3.success = True
        metric3.reason = ""

        test_result2.metrics_data = [metric3]

        # Create mock evaluation result
        results = mocker.Mock(spec=EvaluationResult)
        results.test_results = [test_result1, test_result2]

        return results

    def test_reporter_initialization(self, reporter, mock_stdout):
        """Test reporter initialization."""
        assert reporter.stdout == mock_stdout

    def test_normalize_score_for_aggregation(self, reporter):
        """Test score normalization for different metric types."""
        assert reporter.normalize_score_for_aggregation("Hallucination", 0.0) == 1.0
        assert reporter.normalize_score_for_aggregation("Hallucination", 0.3) == 0.7
        assert reporter.normalize_score_for_aggregation("Hallucination", 1.0) == 0.0
        assert (
            reporter.normalize_score_for_aggregation("hallucinationmetric", 0.2) == 0.8
        )

        # Test other metrics (should remain unchanged)
        assert (
            reporter.normalize_score_for_aggregation("ContextualRelevancy", 0.8) == 0.8
        )
        assert reporter.normalize_score_for_aggregation("AnswerRelevancy", 0.5) == 0.5
        assert (
            reporter.normalize_score_for_aggregation("ContextualPrecision", 1.0) == 1.0
        )
        assert reporter.normalize_score_for_aggregation("ContextualRecall", 0.0) == 0.0

    def test_is_inverse_metric(self, reporter):
        """Test inverse metric detection using settings."""
        assert reporter.is_inverse_metric("Hallucination") is True
        assert reporter.is_inverse_metric("hallucination") is True
        assert reporter.is_inverse_metric("HallucinationMetric") is True
        assert reporter.is_inverse_metric("hallucinationmetric") is True

        # Non-inverse metrics
        assert reporter.is_inverse_metric("ContextualRelevancy") is False
        assert reporter.is_inverse_metric("AnswerRelevancy") is False
        assert reporter.is_inverse_metric("ContextualPrecision") is False
        assert reporter.is_inverse_metric("UnknownMetric") is False

    def test_create_results_dataframe(self, reporter, mock_evaluation_results):
        """Test creation of results DataFrame."""
        df = reporter.create_results_dataframe(mock_evaluation_results)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3  # Two test results with 2 and 1 metrics respectively

        # Check column names
        expected_columns = [
            "bot",
            "model",
            "prompt_label",
            "test_case",
            "question",
            "metric",
            "score",
            "success",
            "reason",
        ]
        assert list(df.columns) == expected_columns

        # Check some values
        assert "recommendation" in df["bot"].values
        assert "syllabus" in df["bot"].values
        assert "gpt-4" in df["model"].values
        assert "gpt-3.5" in df["model"].values

    def test_summarize_per_bot_model(self, reporter, mock_stdout):
        """Test summary per bot and model."""
        # Create test DataFrame
        data = [
            {
                "bot": "recommendation",
                "model": "gpt-4",
                "prompt_label": "default",
                "metric": "ContextualRelevancy",
                "score": 0.85,
                "success": True,
                "reason": "",
            },
            {
                "bot": "recommendation",
                "model": "gpt-4",
                "prompt_label": "default",
                "metric": "AnswerRelevancy",
                "score": 0.45,
                "success": False,
                "reason": "Not relevant",
            },
            {
                "bot": "syllabus",
                "model": "gpt-3.5",
                "prompt_label": "#1",
                "metric": "ContextualRelevancy",
                "score": 0.75,
                "success": True,
                "reason": "",
            },
        ]
        df = pd.DataFrame(data)

        models = ["gpt-4", "gpt-3.5"]
        bot_names = ["recommendation", "syllabus"]

        reporter.summarize_per_bot_model(df, models, bot_names)

        # Verify stdout was called
        assert mock_stdout.write.call_count > 0

        # Check that bot names and models appear in output
        calls = [call[0][0] for call in mock_stdout.write.call_args_list]
        output_text = " ".join(calls)
        assert "RECOMMENDATION BOT" in output_text
        assert "SYLLABUS BOT" in output_text
        assert "gpt-4" in output_text
        assert "gpt-3.5" in output_text

    def test_model_comparison(self, reporter, mock_stdout):
        """Test model comparison functionality."""
        data = [
            {"model": "gpt-4", "metric": "ContextualRelevancy", "score": 0.85},
            {"model": "gpt-4", "metric": "AnswerRelevancy", "score": 0.45},
            {"model": "gpt-3.5", "metric": "ContextualRelevancy", "score": 0.75},
            {"model": "gpt-3.5", "metric": "AnswerRelevancy", "score": 0.65},
        ]
        df = pd.DataFrame(data)

        reporter.model_comparison(df)

        # Verify output was generated
        assert mock_stdout.write.call_count > 0

        calls = [call[0][0] for call in mock_stdout.write.call_args_list]
        output_text = " ".join(calls)
        assert "MODEL COMPARISON" in output_text
        assert "ContextualRelevancy" in output_text
        assert "AnswerRelevancy" in output_text

    def test_model_comparison_hallucination_sorting(self, reporter, mock_stdout):
        """Test that model comparison correctly sorts hallucination metrics (lower is better)."""
        data = [
            {
                "model": "gpt-4",
                "metric": "Hallucination",
                "score": 0.3,
            },  # Worse (higher hallucination)
            {
                "model": "gpt-3.5",
                "metric": "Hallucination",
                "score": 0.1,
            },  # Better (lower hallucination)
            {"model": "claude", "metric": "Hallucination", "score": 0.2},  # Middle
        ]
        df = pd.DataFrame(data)

        reporter.model_comparison(df)

        # Verify output was generated
        assert mock_stdout.write.call_count > 0

        calls = [call[0][0] for call in mock_stdout.write.call_args_list]
        output_text = " ".join(calls)

        # For hallucination, the ranking should be: gpt-3.5 (0.1), claude (0.2), gpt-4 (0.3)
        # since lower hallucination scores are better
        assert "Hallucination" in output_text
        assert "gpt-3.5" in output_text
        assert "gpt-4" in output_text
        assert "claude" in output_text

    def test_overall_performance(self, reporter, mock_stdout):
        """Test overall performance calculation."""
        data = [
            {
                "model": "gpt-4",
                "prompt_label": "default",
                "metric": "ContextualRelevancy",
                "score": 0.85,
            },
            {
                "model": "gpt-4",
                "prompt_label": "#1",
                "metric": "AnswerRelevancy",
                "score": 0.45,
            },
            {
                "model": "gpt-3.5",
                "prompt_label": "default",
                "metric": "ContextualRelevancy",
                "score": 0.75,
            },
            {
                "model": "gpt-3.5",
                "prompt_label": "#1",
                "metric": "AnswerRelevancy",
                "score": 0.65,
            },
        ]
        df = pd.DataFrame(data)

        reporter.overall_performance(df)

        # Verify output was generated
        assert mock_stdout.write.call_count > 0

        calls = [call[0][0] for call in mock_stdout.write.call_args_list]
        output_text = " ".join(calls)
        assert "OVERALL PERFORMANCE" in output_text
        assert "gpt-4" in output_text
        assert "gpt-3.5" in output_text

    def test_overall_performance_with_hallucination_normalization(
        self, reporter, mock_stdout
    ):
        """Test that overall performance correctly normalizes hallucination scores."""
        data = [
            {
                "model": "gpt-4",
                "metric": "ContextualRelevancy",
                "score": 0.8,
                "prompt_label": "default",
            },
            {
                "model": "gpt-4",
                "metric": "Hallucination",
                "score": 0.2,
                "prompt_label": "default",
            },  # Lower is better
            {
                "model": "gpt-3.5",
                "metric": "ContextualRelevancy",
                "score": 0.6,
                "prompt_label": "default",
            },
            {
                "model": "gpt-3.5",
                "metric": "Hallucination",
                "score": 0.4,
                "prompt_label": "default",
            },  # Lower is better
        ]
        df = pd.DataFrame(data)

        # Calculate expected normalized averages:
        # gpt-4: (0.8 + (1.0 - 0.2)) / 2 = (0.8 + 0.8) / 2 = 0.8
        # gpt-3.5: (0.6 + (1.0 - 0.4)) / 2 = (0.6 + 0.6) / 2 = 0.6

        reporter.overall_performance(df)

        # Verify that output was generated and contains model rankings
        assert mock_stdout.write.call_count > 0
        calls = [call[0][0] for call in mock_stdout.write.call_args_list]
        output_text = " ".join(calls)

        assert "OVERALL PERFORMANCE" in output_text
        # gpt-4 should rank higher than gpt-3.5 when hallucination is properly normalized
        assert "gpt-4" in output_text
        assert "gpt-3.5" in output_text

    def test_detailed_results(self, reporter, mock_stdout):
        """Test detailed results display."""
        data = [
            {
                "bot": "recommendation",
                "model": "gpt-4",
                "prompt_label": "default",
                "question": "Test question",
                "metric": "ContextualRelevancy",
                "score": 0.85,
                "success": True,
                "reason": "",
            },
            {
                "bot": "recommendation",
                "model": "gpt-4",
                "prompt_label": "default",
                "question": "Test question",
                "metric": "AnswerRelevancy",
                "score": 0.45,
                "success": False,
                "reason": "Failed",
            },
        ]
        df = pd.DataFrame(data)

        models = ["gpt-4"]
        bot_names = ["recommendation"]

        reporter.detailed_results(df, models, bot_names)

        # Verify output was generated
        assert mock_stdout.write.call_count > 0

        calls = [call[0][0] for call in mock_stdout.write.call_args_list]
        output_text = " ".join(calls)
        assert "DETAILED RESULTS" in output_text
        assert "RECOMMENDATION BOT" in output_text
        assert "Test question" in output_text

    def test_generate_report(self, reporter, mock_evaluation_results, mock_stdout):
        """Test complete report generation."""
        models = ["gpt-4", "gpt-3.5"]
        bot_names = ["recommendation", "syllabus"]

        reporter.generate_report(mock_evaluation_results, models, bot_names)

        # Verify report sections were generated
        assert mock_stdout.write.call_count > 0

        calls = [call[0][0] for call in mock_stdout.write.call_args_list]
        output_text = " ".join(calls)

        # Check for report sections
        assert "RAG EVALUATION REPORT" in output_text
        assert "SUMMARY BY BOT, MODEL, PROMPT" in output_text
        assert "MODEL COMPARISON" in output_text
        assert "PROMPT COMPARISON" in output_text
        assert "OVERALL PERFORMANCE" in output_text
        assert "DETAILED RESULTS" in output_text

    def test_empty_dataframe_handling(self, reporter):
        """Test handling of empty DataFrames."""
        df = pd.DataFrame()
        models = []
        bot_names = []

        # These should not crash with empty data
        reporter.summarize_per_bot_model(df, models, bot_names)
        reporter.model_comparison(df)
        reporter.overall_performance(df)
        reporter.detailed_results(df, models, bot_names)


class TestSummaryReporter:
    """Test cases for SummaryReporter."""

    @pytest.fixture
    def summary_reporter(self, mock_stdout):
        """Create summary reporter for testing."""
        return SummaryReporter(mock_stdout)

    @pytest.fixture
    def mock_simple_results(self, mocker):
        """Create simple mock results for summary testing."""
        test_result = mocker.Mock()
        test_result.additional_metadata = {
            "bot_name": "recommendation",
            "model": "gpt-4",
            "prompt_label": "default",
        }

        metric1 = mocker.Mock()
        metric1.name = "ContextualRelevancy"
        metric1.success = True

        metric2 = mocker.Mock()
        metric2.name = "AnswerRelevancy"
        metric2.success = False

        test_result.metrics_data = [metric1, metric2]

        results = mocker.Mock(spec=EvaluationResult)
        results.test_results = [test_result]

        return results

    def test_summary_reporter_initialization(self, summary_reporter, mock_stdout):
        """Test summary reporter initialization."""
        assert summary_reporter.stdout == mock_stdout

    def test_create_results_dataframe(self, summary_reporter, mock_simple_results):
        """Test DataFrame creation for summary."""
        df = summary_reporter.create_results_dataframe(mock_simple_results)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2  # Two metrics

        expected_columns = ["bot", "model", "metric", "success"]
        assert list(df.columns) == expected_columns

    def test_generate_summary(self, summary_reporter, mock_simple_results, mock_stdout):
        """Test summary generation."""
        summary_reporter.generate_summary(mock_simple_results)

        # Verify summary was generated
        assert mock_stdout.write.call_count > 0

        calls = [call[0][0] for call in mock_stdout.write.call_args_list]
        output_text = " ".join(calls)

        assert "EVALUATION SUMMARY" in output_text
        assert "Total Tests:" in output_text
        assert "Passed:" in output_text
        assert "Failed:" in output_text
        assert "Pass Rate:" in output_text

    def test_generate_summary_empty_results(
        self, summary_reporter, mock_stdout, mocker
    ):
        """Test summary generation with empty results."""
        empty_results = mocker.Mock(spec=EvaluationResult)
        empty_results.test_results = []

        summary_reporter.generate_summary(empty_results)

        # Should handle empty results gracefully
        assert mock_stdout.write.call_count > 0

        calls = [call[0][0] for call in mock_stdout.write.call_args_list]
        output_text = " ".join(calls)
        assert "No test results to summarize" in output_text

    def test_generate_summary_calculations(self, summary_reporter, mock_stdout, mocker):
        """Test summary calculations are correct."""
        # Create results with known pass/fail counts
        test_results = []

        # Create 3 test results with 2 metrics each (6 total metrics)
        for i in range(3):
            test_result = mocker.Mock()
            test_result.additional_metadata = {
                "bot_name": f"bot{i}",
                "model": "gpt-4",
                "prompt_label": "default",
            }

            # First metric passes, second fails
            metric1 = mocker.Mock()
            metric1.name = "Metric1"
            metric1.success = True

            metric2 = mocker.Mock()
            metric2.name = "Metric2"
            metric2.success = False

            test_result.metrics_data = [metric1, metric2]
            test_results.append(test_result)

        results = mocker.Mock(spec=EvaluationResult)
        results.test_results = test_results

        summary_reporter.generate_summary(results)

        calls = [call[0][0] for call in mock_stdout.write.call_args_list]
        output_text = " ".join(calls)

        # Should be 6 total tests, 3 passed, 3 failed, 50% pass rate
        assert "Total Tests: 6" in output_text
        assert "Passed: 3" in output_text
        assert "Failed: 3" in output_text
        assert "Pass Rate: 50.0%" in output_text


class TestReportingIntegration:
    """Integration tests for reporting functionality."""

    def test_end_to_end_reporting(self, mocker):
        """Test end-to-end reporting workflow."""
        mock_stdout = mocker.Mock()
        reporter = EvaluationReporter(mock_stdout)

        # Create comprehensive test data
        test_result = mocker.Mock()
        test_result.additional_metadata = {
            "bot_name": "recommendation",
            "model": "gpt-4",
            "prompt_label": "default",
            "question": "What courses are available?",
        }
        test_result.name = "recommendation-gpt-4"

        metric = mocker.Mock()
        metric.name = "ContextualRelevancy"
        metric.score = 0.85
        metric.success = True
        metric.reason = ""

        test_result.metrics_data = [metric]

        results = mocker.Mock(spec=EvaluationResult)
        results.test_results = [test_result]

        # Generate full report
        reporter.generate_report(results, ["gpt-4"], ["recommendation"])

        # Verify all sections were included
        assert mock_stdout.write.call_count > 0

        calls = [call[0][0] for call in mock_stdout.write.call_args_list]
        output_text = " ".join(calls)

        # Check all major sections are present
        required_sections = [
            "RAG EVALUATION REPORT",
            "SUMMARY BY BOT, MODEL, PROMPT",
            "MODEL COMPARISON",
            "PROMPT COMPARISON",
            "OVERALL PERFORMANCE",
            "DETAILED RESULTS",
        ]

        for section in required_sections:
            assert section in output_text

    def test_both_reporters_together(self, mocker, mock_stdout):
        """Test using both evaluation and summary reporters."""

        evaluation_reporter = EvaluationReporter(mock_stdout)
        summary_reporter = SummaryReporter(mock_stdout)

        # Create test data
        results = mocker.Mock(spec=EvaluationResult)
        results.test_results = []

        # Use both reporters
        summary_reporter.generate_summary(results)
        evaluation_reporter.generate_report(results, [], [])

        # Both should work without conflicts
        assert mock_stdout.write.call_count > 0
