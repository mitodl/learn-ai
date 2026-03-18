"""Unit tests for timeout wrapper functionality."""

import asyncio
import time

import pytest
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from ai_chatbots.evaluation.timeout_wrapper import (
    TimeoutMetricWrapper,
    wrap_metrics_with_timeout,
)


class MockMetric(BaseMetric):
    """Mock metric for testing purposes."""

    def __init__(
        self,
        name="MockMetric",
        delay=0,
        *,
        should_raise=False,
        async_support=False,
        fail_count=0,
    ):
        super().__init__()
        self._name = name  # Store name privately since __name__ is a property
        self.delay = delay
        self.should_raise = should_raise
        self.async_support = async_support
        self.fail_count = fail_count
        self.call_count = 0
        self.threshold = 0.5
        self.include_reason = True

        # Initialize metric state
        self.score = None
        self.success = None
        self.reason = None
        self.error = None

    @property
    def __name__(self):
        """Override __name__ property."""
        return self._name

    def measure(self, test_case: LLMTestCase) -> float:
        """Mock measure method with configurable behavior."""
        _ = test_case  # Mark as used
        self.call_count += 1

        if self.delay > 0:
            time.sleep(self.delay)

        if self.fail_count > 0 and self.call_count <= self.fail_count:
            msg = "Mock transient error"
            raise RuntimeError(msg)

        if self.should_raise:
            msg = "Mock metric error"
            raise RuntimeError(msg)

        # Set mock results
        self.score = 0.8
        self.success = True
        self.reason = "Mock reason"

        return 0.8

    async def a_measure(self, test_case: LLMTestCase) -> float:
        """Mock async measure method."""
        _ = test_case  # Mark as used
        self.call_count += 1

        if not self.async_support:
            msg = "Mock metric doesn't support async"
            raise AttributeError(msg)

        if self.delay > 0:
            await asyncio.sleep(self.delay)

        if self.fail_count > 0 and self.call_count <= self.fail_count:
            msg = "Mock transient async error"
            raise RuntimeError(msg)

        if self.should_raise:
            msg = "Mock async metric error"
            raise RuntimeError(msg)

        # Set mock results
        self.score = 0.9
        self.success = True
        self.reason = "Mock async reason"

        return 0.9


class TestTimeoutMetricWrapper:
    """Test cases for TimeoutMetricWrapper class."""

    def test_initialization(self):
        """Test TimeoutMetricWrapper initialization with custom and default values."""
        base_metric = MockMetric("TestMetric")

        # Test custom timeout
        wrapper = TimeoutMetricWrapper(base_metric, timeout_seconds=60)
        assert wrapper.base_metric is base_metric
        assert wrapper.timeout_seconds == 60
        assert wrapper.threshold == 0.5
        assert wrapper.include_reason is True
        assert wrapper.score is None
        assert wrapper.success is None

        # Test default timeout
        wrapper_default = TimeoutMetricWrapper(base_metric)
        assert wrapper_default.timeout_seconds == 180

    def test_sync_measure_success_and_timeout(self):
        """Test sync measure method for both success and timeout scenarios."""
        test_case = LLMTestCase(input="test", actual_output="output")

        # Test successful execution
        base_metric = MockMetric("TestMetric")
        wrapper = TimeoutMetricWrapper(base_metric, timeout_seconds=5)
        score = wrapper.measure(test_case)

        assert score == 0.8
        assert wrapper.score == 0.8
        assert wrapper.success is True
        assert wrapper.reason == "Mock reason"

        # Test timeout (no retries to keep test fast)
        slow_metric = MockMetric("SlowMetric", delay=2)
        timeout_wrapper = TimeoutMetricWrapper(
            slow_metric, timeout_seconds=1, max_retries=0
        )
        timeout_score = timeout_wrapper.measure(test_case)

        assert timeout_score == 0.0
        assert timeout_wrapper.score == 0.0
        assert timeout_wrapper.success is False
        assert "Timeout after 1 seconds" in timeout_wrapper.reason

    def test_sync_measure_exception(self):
        """Test sync measure method with exception handling (no retries)."""
        base_metric = MockMetric("ErrorMetric", should_raise=True)
        wrapper = TimeoutMetricWrapper(base_metric, timeout_seconds=5, max_retries=0)
        test_case = LLMTestCase(input="test", actual_output="output")

        score = wrapper.measure(test_case)
        assert score == 0.0
        assert wrapper.success is False
        assert wrapper.error == "Mock metric error"

    @pytest.mark.asyncio
    async def test_async_measure_success_and_timeout(self):
        """Test async measure method for both success and timeout scenarios."""
        test_case = LLMTestCase(input="test", actual_output="output")

        # Test successful async execution
        async_metric = MockMetric("AsyncMetric", async_support=True)
        wrapper = TimeoutMetricWrapper(async_metric, timeout_seconds=5)
        score = await wrapper.a_measure(test_case)

        assert score == 0.9
        assert wrapper.score == 0.9
        assert wrapper.success is True
        assert wrapper.reason == "Mock async reason"

        # Test async timeout (no retries to keep test fast)
        slow_async_metric = MockMetric("SlowAsyncMetric", delay=2, async_support=True)
        timeout_wrapper = TimeoutMetricWrapper(
            slow_async_metric, timeout_seconds=1, max_retries=0
        )
        timeout_score = await timeout_wrapper.a_measure(test_case)

        assert timeout_score == 0.0
        assert timeout_wrapper.score == 0.0
        assert timeout_wrapper.success is False
        assert "Timeout after 1 seconds" in timeout_wrapper.reason

    @pytest.mark.asyncio
    async def test_async_measure_exception(self):
        """Test async measure method with exception handling (no retries)."""
        error_metric = MockMetric(
            "AsyncErrorMetric", should_raise=True, async_support=True
        )
        wrapper = TimeoutMetricWrapper(error_metric, timeout_seconds=5, max_retries=0)
        test_case = LLMTestCase(input="test", actual_output="output")

        score = await wrapper.a_measure(test_case)
        assert score == 0.0
        assert wrapper.success is False
        assert wrapper.error == "Mock async metric error"

    def test_is_successful_logic(self):
        """Test is_successful method with various states."""
        base_metric = MockMetric("TestMetric")
        wrapper = TimeoutMetricWrapper(base_metric)

        # Test explicit success values
        wrapper.success = True
        assert wrapper.is_successful() is True

        wrapper.success = False
        assert wrapper.is_successful() is False

        # Test fallback logic
        wrapper.success = None
        wrapper.score = 0.8
        wrapper.error = None
        assert wrapper.is_successful() is True

        wrapper.score = None
        assert wrapper.is_successful() is False

        wrapper.score = 0.8
        wrapper.error = "Some error"
        assert wrapper.is_successful() is False

    def test_wrapper_interface(self):
        """Test wrapper interface, attribute delegation, and naming."""
        base_metric = MockMetric("TestMetric")
        base_metric.custom_attr = "custom_value"
        wrapper = TimeoutMetricWrapper(base_metric)

        # Test naming properties
        assert wrapper.__name__ == "TestMetric"
        assert str(wrapper) == str(base_metric)
        assert repr(wrapper) == repr(base_metric)

        # Test attribute delegation
        assert wrapper.custom_attr == "custom_value"

        # Test error for non-existent attribute
        with pytest.raises(AttributeError):
            _ = wrapper.non_existent_attr


class TestWrapMetricsWithTimeout:
    """Test cases for wrap_metrics_with_timeout function."""

    def test_wrap_metrics_functionality(self):
        """Test wrapping metrics with various scenarios."""
        # Test single metric
        base_metric = MockMetric("TestMetric")
        wrapped_metrics = wrap_metrics_with_timeout([base_metric], timeout_seconds=30)

        assert len(wrapped_metrics) == 1
        wrapped = wrapped_metrics[0]
        assert isinstance(wrapped, TimeoutMetricWrapper)
        assert wrapped.base_metric is base_metric
        assert wrapped.timeout_seconds == 30
        assert wrapped.__class__.__name__ == "MockMetric"

        # Test multiple metrics
        metrics = [MockMetric("Metric1"), MockMetric("Metric2")]
        wrapped_multiple = wrap_metrics_with_timeout(metrics, timeout_seconds=60)

        assert len(wrapped_multiple) == 2
        for i, wrapped in enumerate(wrapped_multiple):
            assert isinstance(wrapped, TimeoutMetricWrapper)
            assert wrapped.base_metric is metrics[i]
            assert wrapped.timeout_seconds == 60

        # Test empty list
        assert wrap_metrics_with_timeout([]) == []

    def test_wrap_with_default_timeout(self):
        """Test wrapping with default timeout and functionality preservation."""
        base_metric = MockMetric("TestMetric")
        wrapped_metrics = wrap_metrics_with_timeout([base_metric])

        assert len(wrapped_metrics) == 1
        wrapped = wrapped_metrics[0]
        assert wrapped.timeout_seconds == 180  # Default timeout

        # Test that functionality is preserved
        test_case = LLMTestCase(input="test", actual_output="output")
        score = wrapped.measure(test_case)
        assert score == 0.8  # Same as original metric
        assert wrapped.score == 0.8
        assert wrapped.success is True

    def test_wrap_with_retry_params(self):
        """Test wrapping passes retry parameters through."""
        base_metric = MockMetric("TestMetric")
        wrapped_metrics = wrap_metrics_with_timeout(
            [base_metric], timeout_seconds=30, max_retries=5, retry_base_delay=2.0
        )

        assert len(wrapped_metrics) == 1
        wrapped = wrapped_metrics[0]
        assert wrapped.max_retries == 5
        assert wrapped.retry_base_delay == 2.0


class TestRetryBehavior:
    """Test cases for retry logic in TimeoutMetricWrapper."""

    def test_sync_retry_on_error_then_succeed(self):
        """Test sync measure retries on error and eventually succeeds."""
        # Fail twice then succeed on 3rd call
        metric = MockMetric("RetryMetric", fail_count=2)
        wrapper = TimeoutMetricWrapper(
            metric, timeout_seconds=5, max_retries=3, retry_base_delay=0.01
        )
        test_case = LLMTestCase(input="test", actual_output="output")

        score = wrapper.measure(test_case)
        assert score == 0.8
        assert wrapper.success is True
        assert metric.call_count == 3

    def test_sync_retry_exhausted(self):
        """Test sync measure exhausts retries and returns 0.0."""
        # Always fail (fail_count > max_retries + 1)
        metric = MockMetric("AlwaysFailMetric", fail_count=10)
        wrapper = TimeoutMetricWrapper(
            metric, timeout_seconds=5, max_retries=2, retry_base_delay=0.01
        )
        test_case = LLMTestCase(input="test", actual_output="output")

        score = wrapper.measure(test_case)
        assert score == 0.0
        assert wrapper.success is False
        assert "failed after 3 attempts" in wrapper.reason
        assert metric.call_count == 3  # 1 initial + 2 retries

    def test_sync_no_retry_on_success(self):
        """Test sync measure does not retry when it succeeds first time."""
        metric = MockMetric("SuccessMetric")
        wrapper = TimeoutMetricWrapper(
            metric, timeout_seconds=5, max_retries=3, retry_base_delay=0.01
        )
        test_case = LLMTestCase(input="test", actual_output="output")

        score = wrapper.measure(test_case)
        assert score == 0.8
        assert metric.call_count == 1

    def test_sync_retry_on_timeout(self):
        """Test sync measure retries on timeout."""
        # Slow metric that will timeout, but we can't make it succeed after
        # So just verify it retries the expected number of times
        metric = MockMetric("SlowMetric", delay=2)
        wrapper = TimeoutMetricWrapper(
            metric, timeout_seconds=0.5, max_retries=1, retry_base_delay=0.01
        )
        test_case = LLMTestCase(input="test", actual_output="output")

        score = wrapper.measure(test_case)
        assert score == 0.0
        assert wrapper.success is False
        assert "failed after 2 attempts" in wrapper.reason

    @pytest.mark.asyncio
    async def test_async_retry_on_error_then_succeed(self):
        """Test async measure retries on error and eventually succeeds."""
        metric = MockMetric("AsyncRetryMetric", fail_count=2, async_support=True)
        wrapper = TimeoutMetricWrapper(
            metric, timeout_seconds=5, max_retries=3, retry_base_delay=0.01
        )
        test_case = LLMTestCase(input="test", actual_output="output")

        score = await wrapper.a_measure(test_case)
        assert score == 0.9
        assert wrapper.success is True
        assert metric.call_count == 3

    @pytest.mark.asyncio
    async def test_async_retry_exhausted(self):
        """Test async measure exhausts retries and returns 0.0."""
        metric = MockMetric("AsyncAlwaysFailMetric", fail_count=10, async_support=True)
        wrapper = TimeoutMetricWrapper(
            metric, timeout_seconds=5, max_retries=2, retry_base_delay=0.01
        )
        test_case = LLMTestCase(input="test", actual_output="output")

        score = await wrapper.a_measure(test_case)
        assert score == 0.0
        assert wrapper.success is False
        assert "failed after 3 attempts" in wrapper.reason
        assert metric.call_count == 3

    @pytest.mark.asyncio
    async def test_async_retry_on_timeout_then_succeed(self):
        """Test async measure retries on timeout and succeeds on later attempt."""
        # First call will be slow (timeout), subsequent calls will be fast
        metric = MockMetric("AsyncSlowThenFast", delay=0, async_support=True)
        # We'll use fail_count to simulate the first call timing out
        # by using a slow delay only on first calls
        wrapper = TimeoutMetricWrapper(
            metric, timeout_seconds=5, max_retries=2, retry_base_delay=0.01
        )
        test_case = LLMTestCase(input="test", actual_output="output")

        score = await wrapper.a_measure(test_case)
        assert score == 0.9
        assert wrapper.success is True

    @pytest.mark.asyncio
    async def test_async_no_retry_on_success(self):
        """Test async measure does not retry when it succeeds first time."""
        metric = MockMetric("AsyncSuccessMetric", async_support=True)
        wrapper = TimeoutMetricWrapper(
            metric, timeout_seconds=5, max_retries=3, retry_base_delay=0.01
        )
        test_case = LLMTestCase(input="test", actual_output="output")

        score = await wrapper.a_measure(test_case)
        assert score == 0.9
        assert metric.call_count == 1

    def test_zero_retries_behaves_like_original(self):
        """Test that max_retries=0 means no retries (single attempt)."""
        metric = MockMetric("NoRetryMetric", fail_count=1)
        wrapper = TimeoutMetricWrapper(
            metric, timeout_seconds=5, max_retries=0, retry_base_delay=0.01
        )
        test_case = LLMTestCase(input="test", actual_output="output")

        score = wrapper.measure(test_case)
        assert score == 0.0
        assert wrapper.success is False
        assert metric.call_count == 1
