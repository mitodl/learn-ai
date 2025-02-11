"""Common text fixtures"""

import pytest


@pytest.fixture
def client():
    """DRF API client"""
    from rest_framework.test import APIClient

    return APIClient()
