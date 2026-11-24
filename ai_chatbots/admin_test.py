"""Tests for ai_chatbots admin"""

import pytest
from django.contrib.admin.sites import AdminSite
from django.test import RequestFactory

from ai_chatbots.admin import SupportWidgetTextAdmin
from ai_chatbots.models import SupportWidgetText
from main.factories import UserFactory

pytestmark = pytest.mark.django_db


@pytest.fixture
def text_admin_request():
    """Return the widget text admin and a superuser request for it."""
    request = RequestFactory().get("/admin/")
    request.user = UserFactory.create(is_staff=True, is_superuser=True)
    return SupportWidgetTextAdmin(SupportWidgetText, AdminSite()), request


@pytest.mark.usefixtures("support_widget_text")
def test_support_widget_text_admin_no_second_row(text_admin_request):
    """The widget reads one row, so a second can't be added."""
    text_admin, request = text_admin_request
    assert not text_admin.has_add_permission(request)


def test_support_widget_text_admin_add_when_row_missing(text_admin_request):
    """If the row is ever removed from a shell, the admin can put it back."""
    text_admin, request = text_admin_request
    SupportWidgetText.objects.all().delete()
    assert text_admin.has_add_permission(request)


def test_support_widget_text_admin_no_delete(text_admin_request, support_widget_text):
    """Clearing a field is how a string goes back to its default."""
    text_admin, request = text_admin_request
    assert not text_admin.has_delete_permission(request, support_widget_text)
