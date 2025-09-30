"""Timeout wrapper for DeepEval metrics to prevent hanging test cases."""

import asyncio
import threading
from typing import Any

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class TimeoutMetricWrapper(BaseMetric):
    """Wrapper that adds timeout functionality to any DeepEval metric."""

    def __init__(self, base_metric: BaseMetric, timeout_seconds: int = 180):
        """Initialize the timeout wrapper.

        Args:
            base_metric: The original metric to wrap
            timeout_seconds: Maximum time allowed for metric execution
                (default: 360 seconds)
        """
        self.base_metric = base_metric
        self.timeout_seconds = timeout_seconds

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

    def _execute_with_timeout(self, method: Any, test_case: LLMTestCase) -> float:
        """Execute a method with timeout using threading."""
        result = {"score": None, "error": None, "completed": False}

        def target():
            try:
                score = method(test_case)
                # Copy over the attributes from the base metric
                self.score = self.base_metric.score
                self.success = self.base_metric.success
                self.reason = getattr(self.base_metric, "reason", None)
                self.error = getattr(self.base_metric, "error", None)

                result["score"] = score
                result["completed"] = True
            except Exception as e:  # noqa: BLE001
                result["error"] = str(e)
                result["completed"] = True

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout_seconds)

        if thread.is_alive():
            # Thread is still running, timeout occurred
            self.score = 0.0
            self.success = False
            self.reason = (
                f"Metric execution timed out after {self.timeout_seconds} seconds"
            )
            self.error = f"Timeout after {self.timeout_seconds} seconds"
            return 0.0

        if result["error"]:
            self.error = result["error"]
            raise RuntimeError(result["error"])

        if not result["completed"]:
            self.score = 0.0
            self.success = False
            self.reason = "Metric execution failed to complete"
            self.error = "Execution failed to complete"
            return 0.0

        return result["score"]

    async def _execute_async_with_timeout(
        self, method: Any, test_case: LLMTestCase
    ) -> float:
        """Execute an async method with timeout."""
        try:
            score = await asyncio.wait_for(
                method(test_case), timeout=self.timeout_seconds
            )
            # Copy over the attributes from the base metric
            self.score = self.base_metric.score
            self.success = self.base_metric.success
            self.reason = getattr(self.base_metric, "reason", None)
            self.error = getattr(self.base_metric, "error", None)

        except TimeoutError:
            self.score = 0.0
            self.success = False
            self.reason = (
                f"Metric execution timed out after {self.timeout_seconds} seconds"
            )
            self.error = f"Timeout after {self.timeout_seconds} seconds"
            return 0.0
        except Exception as e:
            self.error = str(e)
            raise
        else:
            return score

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
    metrics: list[BaseMetric], timeout_seconds: int = 180
) -> list[BaseMetric]:
    """Wrap a list of metrics with timeout functionality.

    Args:
        metrics: List of metrics to wrap
        timeout_seconds: Timeout in seconds (default: 180 = 3 minutes)

    Returns:
        List of timeout-wrapped metrics
    """
    wrapped_metrics = []
    for metric in metrics:
        # Create a dynamic class that inherits from TimeoutMetricWrapper
        # but has the same name as the original metric
        original_class_name = metric.__class__.__name__

        # Create a new class dynamically with the original metric's name
        DynamicTimeoutWrapper = type(
            original_class_name,  # Class name
            (TimeoutMetricWrapper,),  # Base classes
            {},  # Class attributes
        )

        # Create instance of the dynamic wrapper class
        wrapped_metric = DynamicTimeoutWrapper(metric, timeout_seconds)
        wrapped_metrics.append(wrapped_metric)

    return wrapped_metrics
