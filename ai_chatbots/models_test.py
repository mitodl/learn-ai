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


@pytest.mark.parametrize("field", ["send_label", "cancel_label", "escalate_label"])
def test_support_widget_copy_rejects_send_as_a_chip_label(field):
    """The composer's submit button is already named "Send"."""
    with pytest.raises(ValidationError) as excinfo:
        SupportWidgetCopy(**{field: " send "}).full_clean()
    assert field in excinfo.value.message_dict


@pytest.mark.parametrize(
    "label",
    [
        "![](https://example.com/p.png)",
        "[Help](https://example.com)",
        "# Cancel",
        "1. Yes",
        "- Yes",
        "---",
        "**Stop**",
        "Talk_to_us",
        "x^2^",
        "`stop`",
        "> quote",
    ],
)
@pytest.mark.parametrize("field", ["send_label", "cancel_label", "escalate_label"])
def test_support_widget_copy_rejects_markdown_in_chip_labels(field, label):
    """A pressed chip's label shows in the learner's own message, which renders Markdown."""
    with pytest.raises(ValidationError) as excinfo:
        SupportWidgetCopy(**{field: label}).full_clean()
    assert field in excinfo.value.message_dict


@pytest.mark.parametrize(
    "label",
    ["Never mind", "Don\u2019t send it", "Yes, send it!", "Talk to a person (24/7)"],
)
def test_support_widget_copy_allows_plain_chip_labels(label):
    """Ordinary wording with simple punctuation is fine."""
    SupportWidgetCopy(cancel_label=label).full_clean()


def test_support_widget_copy_strips_a_pasted_byte_order_mark():
    """
    The widget trims U+FEFF like whitespace, so the admin has to as well or it
    would save a label the widget then treats differently.
    """
    with pytest.raises(ValidationError) as excinfo:
        SupportWidgetCopy(send_label="﻿Send").full_clean()
    assert "send_label" in excinfo.value.message_dict


@pytest.mark.django_db
def test_support_widget_copy_save_validates(support_widget_copy):
    """A save outside the admin gets the same cleanup and checks."""
    support_widget_copy.launcher_label = "  Need help?  "
    support_widget_copy.save()
    support_widget_copy.refresh_from_db()
    assert support_widget_copy.launcher_label == "Need help?"

    support_widget_copy.cancel_label = "Send request"
    with pytest.raises(ValidationError):
        support_widget_copy.save()
