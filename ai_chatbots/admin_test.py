"""Tests for ai_chatbots admin"""

import pytest
from django.contrib.admin.sites import AdminSite
from django.test import RequestFactory

from ai_chatbots.admin import SupportWidgetCopyAdmin
from ai_chatbots.models import SupportWidgetCopy
from main.factories import UserFactory

pytestmark = pytest.mark.django_db


@pytest.fixture
def copy_admin_request():
    """Return the copy admin and a superuser request for it."""
    request = RequestFactory().get("/admin/")
    request.user = UserFactory.create(is_staff=True, is_superuser=True)
    return SupportWidgetCopyAdmin(SupportWidgetCopy, AdminSite()), request


@pytest.mark.usefixtures("support_widget_copy")
def test_support_widget_copy_admin_no_second_row(copy_admin_request):
    """The widget reads one row, so a second can't be added."""
    copy_admin, request = copy_admin_request
    assert not copy_admin.has_add_permission(request)


def test_support_widget_copy_admin_add_when_row_missing(copy_admin_request):
    """If the row is ever removed from a shell, the admin can put it back."""
    copy_admin, request = copy_admin_request
    SupportWidgetCopy.objects.all().delete()
    assert copy_admin.has_add_permission(request)


def test_support_widget_copy_admin_no_delete(copy_admin_request, support_widget_copy):
    """Clearing a field is how a string goes back to its default."""
    copy_admin, request = copy_admin_request
    assert not copy_admin.has_delete_permission(request, support_widget_copy)
