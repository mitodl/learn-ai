"""APISIX User decode middleware."""

import logging

from django.contrib.auth import login, logout
from django.contrib.auth.middleware import PersistentRemoteUserMiddleware
from django.core.exceptions import ImproperlyConfigured

from main.utils import decode_apisix_headers, get_user_from_apisix_headers

log = logging.getLogger(__name__)


class ApisixUserMiddleware(PersistentRemoteUserMiddleware):
    """Checks for and processes APISIX-specific headers."""

    async_capable = False

    def process_request(self, request):
        """
        Check the request for an authenticated user, or authenticate using the
        APISIX data if there isn't one.
        """

        if not hasattr(request, "user"):
            msg = "ApisixUserMiddleware requires the authentication middleware."
            raise ImproperlyConfigured(msg)

        try:
            apisix_user = get_user_from_apisix_headers(request)
        except KeyError:
            apisix_user = None

        if apisix_user:
            already_logged_in = (
                request.user.is_authenticated and request.user == apisix_user
            )

            if request.user.is_authenticated and request.user != apisix_user:
                # The user is authenticated, but doesn't match the user we got
                # from APISIX. So, log them out so the APISIX user takes
                # precedence.
                log.debug(
                    "Forcing user logout because request user doesn't match APISIX user"
                )

                logout(request)

            request.user = apisix_user
            if not already_logged_in:
                login(
                    request,
                    apisix_user,
                    backend="django.contrib.auth.backends.ModelBackend",
                )
        elif request.user.is_authenticated:
            # APISIX is the source of truth for authentication. With no user
            # header, drop any lingering Django session so that gateway/Keycloak
            # logout propagates to the Django session instead of outliving it.
            log.debug("Logging out Django session due to missing APISIX user.")
            logout(request)

        request.api_gateway_userdata = decode_apisix_headers(request)
