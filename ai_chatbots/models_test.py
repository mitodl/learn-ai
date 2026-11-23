"""Tests for ai_chatbots models"""

import pytest
from django.core.exceptions import ValidationError

from ai_chatbots.models import SupportWidgetCopy


def test_support_widget_copy_strips_whitespace():
    """A field of only spaces saves as blank, so the widget keeps its default."""
    copy = SupportWidgetCopy(launcher_label="  Need help?  ", badge="   ")
    copy.full_clean()
    assert copy.launcher_label == "Need help?"
    assert copy.badge == ""


@pytest.mark.parametrize(
    "greeting",
    [
        "Hi ![](https://example.com/pixel.png)",
        "Hi ![logo][1]\n\n[1]: https://example.com/logo.png",
    ],
)
def test_support_widget_copy_rejects_greeting_images(greeting):
    """The greeting renders on learner pages, so it can't load remote images."""
    with pytest.raises(ValidationError) as excinfo:
        SupportWidgetCopy(greeting=greeting).full_clean()
    assert "greeting" in excinfo.value.message_dict


def test_support_widget_copy_allows_greeting_links():
    """Links are fine; only images are refused."""
    SupportWidgetCopy(greeting="See [the help center](https://mit.edu).").full_clean()


@pytest.mark.parametrize(
    ("send_label", "cancel_label"),
    [
        ("Stop", "stop"),
        ("Cancel", ""),
        ("", "send request"),
    ],
)
def test_support_widget_copy_rejects_same_send_and_cancel(send_label, cancel_label):
    """
    Send and Cancel show side by side and are keyed by their text, so they can't
    match, counting a blank one as its default.
    """
    with pytest.raises(ValidationError) as excinfo:
        SupportWidgetCopy(send_label=send_label, cancel_label=cancel_label).full_clean()
    assert "send_label" in excinfo.value.message_dict
