"""Unit tests for evaluation reporting classes."""

from unittest.mock import Mock

import pandas as pd
import pytest
from deepeval.evaluate.types import EvaluationResult

from .reporting import EvaluationReporter, SummaryReporter


class TestEvaluationReporter:
    """Test cases for EvaluationReporter."""

    @pytest.fixture
    def mock_stdout(self):
        """Create mock stdout for testing."""
        return Mock()

    @pytest.fixture
    def reporter(self, mock_stdout):
        """Create reporter for testing."""
        return EvaluationReporter(mock_stdout)

    @pytest.fixture
    def mock_evaluation_results(self):
        """Create mock evaluation results."""
        # Create mock test results
        test_result1 = Mock()
        test_result1.additional_metadata = {
            "bot_name": "recommendation",
            "model": "gpt-4",
            "prompt_label": "default",
            "question": "What courses are available?",
        }
        test_result1.name = "recommendation-gpt-4"

        # Create mock metrics data
        metric1 = Mock()
        metric1.name = "ContextualRelevancy"
        metric1.score = 0.85
        metric1.success = True
        metric1.reason = ""

        metric2 = Mock()
        metric2.name = "AnswerRelevancy"
        metric2.score = 0.45
        metric2.success = False
        metric2.reason = "Answer was not relevant to the question"

        test_result1.metrics_data = [metric1, metric2]

        # Create second test result
        test_result2 = Mock()
        test_result2.additional_metadata = {
            "bot_name": "syllabus",
            "model": "gpt-3.5",
            "prompt_label": "#1",
            "question": "Who teaches this course?",
        }
        test_result2.name = "syllabus-gpt-3.5"

        metric3 = Mock()
        metric3.name = "ContextualRelevancy"
        metric3.score = 0.75
        metric3.success = True
        metric3.reason = ""

        test_result2.metrics_data = [metric3]

        # Create mock evaluation result
        results = Mock(spec=EvaluationResult)
        results.test_results = [test_result1, test_result2]

        return results

    def test_reporter_initialization(self, reporter, mock_stdout):
        """Test reporter initialization."""
        assert reporter.stdout == mock_stdout

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

    def test_overall_performance(self, reporter, mock_stdout):
        """Test overall performance calculation."""
        data = [
            {"model": "gpt-4", "prompt_label": "default", "score": 0.85},
            {"model": "gpt-4", "prompt_label": "#1", "score": 0.45},
            {"model": "gpt-3.5", "prompt_label": "default", "score": 0.75},
            {"model": "gpt-3.5", "prompt_label": "#1", "score": 0.65},
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
    def mock_stdout(self):
        """Create mock stdout for testing."""
        return Mock()

    @pytest.fixture
    def summary_reporter(self, mock_stdout):
        """Create summary reporter for testing."""
        return SummaryReporter(mock_stdout)

    @pytest.fixture
    def mock_simple_results(self):
        """Create simple mock results for summary testing."""
        test_result = Mock()
        test_result.additional_metadata = {
            "bot_name": "recommendation",
            "model": "gpt-4",
            "prompt_label": "default",
        }

        metric1 = Mock()
        metric1.name = "ContextualRelevancy"
        metric1.success = True

        metric2 = Mock()
        metric2.name = "AnswerRelevancy"
        metric2.success = False

        test_result.metrics_data = [metric1, metric2]

        results = Mock(spec=EvaluationResult)
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

    def test_generate_summary_empty_results(self, summary_reporter, mock_stdout):
        """Test summary generation with empty results."""
        empty_results = Mock(spec=EvaluationResult)
        empty_results.test_results = []

        summary_reporter.generate_summary(empty_results)

        # Should handle empty results gracefully
        assert mock_stdout.write.call_count > 0

        calls = [call[0][0] for call in mock_stdout.write.call_args_list]
        output_text = " ".join(calls)
        assert "No test results to summarize" in output_text

    def test_generate_summary_calculations(self, summary_reporter, mock_stdout):
        """Test summary calculations are correct."""
        # Create results with known pass/fail counts
        test_results = []

        # Create 3 test results with 2 metrics each (6 total metrics)
        for i in range(3):
            test_result = Mock()
            test_result.additional_metadata = {
                "bot_name": f"bot{i}",
                "model": "gpt-4",
                "prompt_label": "default",
            }

            # First metric passes, second fails
            metric1 = Mock()
            metric1.name = "Metric1"
            metric1.success = True

            metric2 = Mock()
            metric2.name = "Metric2"
            metric2.success = False

            test_result.metrics_data = [metric1, metric2]
            test_results.append(test_result)

        results = Mock(spec=EvaluationResult)
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

    def test_end_to_end_reporting(self):
        """Test end-to-end reporting workflow."""
        mock_stdout = Mock()
        reporter = EvaluationReporter(mock_stdout)

        # Create comprehensive test data
        test_result = Mock()
        test_result.additional_metadata = {
            "bot_name": "recommendation",
            "model": "gpt-4",
            "prompt_label": "default",
            "question": "What courses are available?",
        }
        test_result.name = "recommendation-gpt-4"

        metric = Mock()
        metric.name = "ContextualRelevancy"
        metric.score = 0.85
        metric.success = True
        metric.reason = ""

        test_result.metrics_data = [metric]

        results = Mock(spec=EvaluationResult)
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

    def test_both_reporters_together(self):
        """Test using both evaluation and summary reporters."""
        mock_stdout = Mock()

        evaluation_reporter = EvaluationReporter(mock_stdout)
        summary_reporter = SummaryReporter(mock_stdout)

        # Create test data
        results = Mock(spec=EvaluationResult)
        results.test_results = []

        # Use both reporters
        summary_reporter.generate_summary(results)
        evaluation_reporter.generate_report(results, [], [])

        # Both should work without conflicts
        assert mock_stdout.write.call_count > 0
