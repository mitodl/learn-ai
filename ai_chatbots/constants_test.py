"""Tests for ai_chatbots constants."""

from ai_chatbots.constants import ChatbotCookie


def test_chatbot_cookie_is_sent_cross_site(settings):
    """
    The chat widgets are embedded on other origins, so the thread cookie has to come
    back on cross-site requests. A cookie with no SameSite attribute is treated as
    Lax and dropped, which starts a new thread on every message.
    """
    settings.AI_CHATBOTS_COOKIE_CROSS_SITE = True

    assert (
        str(ChatbotCookie(name="thread", value="abc", max_age=604800))
        == "thread=abc;Path=/;Max-Age=604800;SameSite=None;Secure;"
    )


def test_chatbot_cookie_drops_cross_site_attributes_when_disabled(settings):
    """
    SameSite=None only works alongside Secure, which browsers refuse over plain
    HTTP, so local development has to be able to turn both off.
    """
    settings.AI_CHATBOTS_COOKIE_CROSS_SITE = False

    assert (
        str(ChatbotCookie(name="thread", value="abc", max_age=604800))
        == "thread=abc;Path=/;Max-Age=604800;"
    )
