"""Timeout wrapper for DeepEval metrics to prevent hanging test cases."""

import asyncio
import logging
import random
import threading
import time
from typing import Any

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

logger = logging.getLogger(__name__)


class TimeoutMetricWrapper(BaseMetric):
    """Wrapper that adds timeout functionality to any DeepEval metric."""

    def __init__(
        self,
        base_metric: BaseMetric,
        timeout_seconds: int = 180,
        max_retries: int = 3,
        retry_base_delay: float = 5.0,
    ):
        """Initialize the timeout wrapper.

        Args:
            base_metric: The original metric to wrap
            timeout_seconds: Maximum time allowed for metric execution
                (default: 360 seconds)
            max_retries: Maximum number of retries on timeout/error (default: 3)
            retry_base_delay: Base delay in seconds between retries, with
                exponential backoff and jitter (default: 5.0)
        """
        self.base_metric = base_metric
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay

        # Copy attributes from base metric
        super().__init__()
        self.threshold = getattr(base_metric, "threshold", 0.5)
        self.include_reason = getattr(base_metric, "include_reason", True)

        # Initialize metric state
        self.score = None
        self.success = None
        self.reason = None
        self.error = None

    def measure(self, test_case: LLMTestCase) -> float:
        """Execute the base metric with timeout protection."""
        return self._execute_with_timeout(self.base_metric.measure, test_case)

    async def a_measure(self, test_case: LLMTestCase) -> float:
        """Execute the base metric asynchronously with timeout protection."""
        if hasattr(self.base_metric, "a_measure"):
            return await self._execute_async_with_timeout(
                self.base_metric.a_measure, test_case
            )
        else:
            # Fallback to sync measure in executor
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._execute_with_timeout, self.base_metric.measure, test_case
            )

    def _compute_retry_delay(self, attempt: int) -> float:
        """Compute delay with exponential backoff and jitter."""
        delay = self.retry_base_delay * (2**attempt)
        jitter = random.uniform(0, 2)  # noqa: S311
        return delay + jitter

    def _execute_with_timeout(self, method: Any, test_case: LLMTestCase) -> float:
        """Execute a method with timeout using threading, with retries."""
        last_error = None

        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                delay = self._compute_retry_delay(attempt - 1)
                metric_name = getattr(
                    self.base_metric, "__name__", self.base_metric.__class__.__name__
                )
                logger.warning(
                    "Retry %d/%d for metric %s after %.1fs delay",
                    attempt,
                    self.max_retries,
                    metric_name,
                    delay,
                )
                time.sleep(delay)

            result = {"score": None, "error": None, "completed": False}

            def target(result=result):
                try:
                    score = method(test_case)
                    result["score"] = score
                    result["base_score"] = self.base_metric.score
                    result["base_success"] = self.base_metric.success
                    result["base_reason"] = getattr(self.base_metric, "reason", None)
                    result["base_error"] = getattr(self.base_metric, "error", None)
                    result["completed"] = True
                except Exception as e:  # noqa: BLE001
                    result["error"] = str(e)
                    result["completed"] = True

            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            thread.join(timeout=self.timeout_seconds)

            if thread.is_alive():
                last_error = f"Timeout after {self.timeout_seconds} seconds"
                continue

            if result["error"]:
                last_error = result["error"]
                continue

            if not result["completed"]:
                last_error = "Execution failed to complete"
                continue

            # Success - copy to self only from the main thread
            self.score = result["base_score"]
            self.success = result["base_success"]
            self.reason = result["base_reason"]
            self.error = result["base_error"]
            return result["score"]

        # All retries exhausted
        self.score = 0.0
        self.success = False
        self.reason = (
            f"Metric execution failed after {self.max_retries + 1} attempts. "
            f"Last error: {last_error}"
        )
        self.error = last_error
        return 0.0

    async def _execute_async_with_timeout(
        self, method: Any, test_case: LLMTestCase
    ) -> float:
        """Execute an async method with timeout, with retries."""
        last_error = None

        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                delay = self._compute_retry_delay(attempt - 1)
                metric_name = getattr(
                    self.base_metric, "__name__", self.base_metric.__class__.__name__
                )
                logger.warning(
                    "Retry %d/%d for metric %s after %.1fs delay",
                    attempt,
                    self.max_retries,
                    metric_name,
                    delay,
                )
                await asyncio.sleep(delay)

            try:
                score = await asyncio.wait_for(
                    method(test_case), timeout=self.timeout_seconds
                )
                self.score = self.base_metric.score
                self.success = self.base_metric.success
                self.reason = getattr(self.base_metric, "reason", None)
                self.error = getattr(self.base_metric, "error", None)
            except TimeoutError:
                last_error = f"Timeout after {self.timeout_seconds} seconds"
                continue
            except Exception as e:  # noqa: BLE001
                last_error = str(e)
                continue
            else:
                return score

        # All retries exhausted
        self.score = 0.0
        self.success = False
        self.reason = (
            f"Metric execution failed after {self.max_retries + 1} attempts. "
            f"Last error: {last_error}"
        )
        self.error = last_error
        return 0.0

    def is_successful(self) -> bool:
        """Check if the metric evaluation was successful."""
        if self.success is not None:
            return self.success
        # Fallback: check if we have a valid score and no error
        return self.score is not None and self.error is None

    @property
    def __name__(self):
        """Return the original metric name for identification."""
        return getattr(
            self.base_metric, "__name__", self.base_metric.__class__.__name__
        )

    def __str__(self):
        """Return the original metric name as string."""
        return str(self.base_metric)

    def __repr__(self):
        """Return the original metric representation."""
        return repr(self.base_metric)

    def __getattr__(self, name):
        """Delegate attribute access to the base metric if not found in wrapper."""
        # Handle special naming attributes that DeepEval might use
        if name in ["__class__", "__name__", "__module__"] and hasattr(
            self.base_metric, name
        ):
            return getattr(self.base_metric, name)

        # This ensures any additional attributes or methods from the base metric
        # are accessible
        if hasattr(self.base_metric, name):
            return getattr(self.base_metric, name)
        error_msg = f"'{type(self).__name__}' object has no attribute '{name}'"
        raise AttributeError(error_msg)


def wrap_metrics_with_timeout(
    metrics: list[BaseMetric],
    timeout_seconds: int = 180,
    max_retries: int = 3,
    retry_base_delay: float = 5.0,
) -> list[BaseMetric]:
    """Wrap a list of metrics with timeout and retry functionality.

    Args:
        metrics: List of metrics to wrap
        timeout_seconds: Timeout in seconds (default: 180 = 3 minutes)
        max_retries: Maximum retries on timeout/error (default: 3)
        retry_base_delay: Base delay in seconds for exponential backoff (default: 5.0)

    Returns:
        List of timeout-wrapped metrics
    """
    wrapped_metrics = []
    for metric in metrics:
        original_class_name = metric.__class__.__name__

        DynamicTimeoutWrapper = type(
            original_class_name,
            (TimeoutMetricWrapper,),
            {},
        )

        wrapped_metric = DynamicTimeoutWrapper(
            metric, timeout_seconds, max_retries, retry_base_delay
        )
        wrapped_metrics.append(wrapped_metric)

    return wrapped_metrics
