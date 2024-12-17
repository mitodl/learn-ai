"""
Validate that our settings functions work
"""

import importlib
import sys
from unittest import mock

import pytest
import semantic_version
from django.conf import settings
from django.core import mail
from django.core.exceptions import ImproperlyConfigured
from django.test import TestCase

REQUIRED_SETTINGS = {
    "MAILGUN_SENDER_DOMAIN": "mailgun.fake.domain",
    "MAILGUN_KEY": "fake_mailgun_key",
    "MITOL_COOKIE_NAME": "cookie_monster",
    "MITOL_COOKIE_DOMAIN": "od.fake.domain",
    "MITOL_APP_BASE_URL": "http:localhost:8001/",
}


class TestSettings(TestCase):
    """Validate that settings work as expected."""

    def reload_settings(self):
        """
        Reload settings module with cleanup to restore it.

        Returns:
            dict: dictionary of the newly reloaded settings ``vars``
        """
        importlib.reload(sys.modules["main.settings"])
        # Restore settings to original settings after test
        self.addCleanup(importlib.reload, sys.modules["main.settings"])
        return vars(sys.modules["main.settings"])

    def test_admin_settings(self):
        """Verify that we configure email with environment variable"""

        with mock.patch.dict(
            "os.environ",
            {**REQUIRED_SETTINGS, "MITOL_ADMIN_EMAIL": ""},
            clear=True,
        ):
            settings_vars = self.reload_settings()
            assert not settings_vars.get("ADMINS", False)

        test_admin_email = "cuddle_bunnies@example.com"
        with mock.patch.dict(
            "os.environ",
            {**REQUIRED_SETTINGS, "MITOL_ADMIN_EMAIL": test_admin_email},
            clear=True,
        ):
            settings_vars = self.reload_settings()
            assert (("Admins", test_admin_email),) == settings_vars["ADMINS"]
        # Manually set ADMIN to our test setting and verify e-mail
        # goes where we expect
        settings.ADMINS = (("Admins", test_admin_email),)
        mail.mail_admins("Test", "message")
        assert test_admin_email in mail.outbox[0].to

    def test_db_ssl_enable(self):
        """Verify that we can enable/disable database SSL with a var"""

        # Check default state is SSL on
        with mock.patch.dict("os.environ", REQUIRED_SETTINGS, clear=True):
            settings_vars = self.reload_settings()
            assert settings_vars["DATABASES"]["default"]["OPTIONS"] == {
                "sslmode": "require"
            }

        # Check enabling the setting explicitly
        with mock.patch.dict(
            "os.environ",
            {**REQUIRED_SETTINGS, "MITOL_DB_DISABLE_SSL": "True"},
            clear=True,
        ):
            settings_vars = self.reload_settings()
            assert settings_vars["DATABASES"]["default"]["OPTIONS"] == {}

        # Disable it
        with mock.patch.dict(
            "os.environ",
            {**REQUIRED_SETTINGS, "MITOL_DB_DISABLE_SSL": "False"},
            clear=True,
        ):
            settings_vars = self.reload_settings()
            assert settings_vars["DATABASES"]["default"]["OPTIONS"] == {
                "sslmode": "require"
            }

    @staticmethod
    def test_semantic_version():
        """
        Verify that we have a semantic compatible version.
        """
        semantic_version.Version(settings.VERSION)

    def test_required_settings(self):
        """
        Assert that an exception is raised if any of the required settings are missing
        """
        for key in REQUIRED_SETTINGS:
            required_settings = {**REQUIRED_SETTINGS}
            del required_settings[key]
            with (
                mock.patch.dict("os.environ", required_settings, clear=True),
                pytest.raises(ImproperlyConfigured),
            ):
                self.reload_settings()

    def test_server_side_cursors_disabled(self):
        """DISABLE_SERVER_SIDE_CURSORS should be true by default"""
        with mock.patch.dict("os.environ", REQUIRED_SETTINGS):
            settings_vars = self.reload_settings()
            assert (
                settings_vars["DEFAULT_DATABASE_CONFIG"]["DISABLE_SERVER_SIDE_CURSORS"]
                is True
            )

    def test_server_side_cursors_enabled(self):
        """DISABLE_SERVER_SIDE_CURSORS should be false if MITOL_DB_DISABLE_SS_CURSORS is false"""
        with mock.patch.dict(
            "os.environ",
            {**REQUIRED_SETTINGS, "MITOL_DB_DISABLE_SS_CURSORS": "False"},
        ):
            settings_vars = self.reload_settings()
            assert (
                settings_vars["DEFAULT_DATABASE_CONFIG"]["DISABLE_SERVER_SIDE_CURSORS"]
                is False
            )
