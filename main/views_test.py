"""Tests for the main views"""


def test_redirect_route(settings, user_client):
    """
    Simple Test that checks that we have a catch all redirect view
    so that is not accidently removed
    """
    response = user_client.get("/app", follow=True)
    assert response.redirect_chain[0][0] == settings.APP_BASE_URL
    assert response.redirect_chain[0][1] == 302
