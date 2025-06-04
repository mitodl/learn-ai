"""
Django settings for main.


For more information on this file, see
https://docs.djangoproject.com/en/1.10/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/1.10/ref/settings/

"""

# pylint:disable=wildcard-import,unused-wildcard-import)
import datetime
import logging
import os
import platform
from urllib.parse import urljoin

import dj_database_url
from celery.schedules import crontab
from django.core.exceptions import ImproperlyConfigured

from main.envs import (
    get_bool,
    get_float,
    get_int,
    get_list_of_str,
    get_string,
)
from main.sentry import init_sentry
from openapi.settings_spectacular import open_spectacular_settings

VERSION = "0.9.1"

log = logging.getLogger()

REDIS_URL = get_string("REDIS_URL", get_string("REDISCLOUD_URL", None))
CELERY_BROKER_URL = get_string("CELERY_BROKER_URL", REDIS_URL)
ENVIRONMENT = get_string("MITOL_ENVIRONMENT", "dev")
DEFAULT_AUTO_FIELD = "django.db.models.AutoField"

# initialize Sentry before doing anything else so we capture any config errors
SENTRY_DSN = get_string("SENTRY_DSN", "")
SENTRY_LOG_LEVEL = get_string("SENTRY_LOG_LEVEL", "ERROR")
SENTRY_TRACES_SAMPLE_RATE = get_float("SENTRY_TRACES_SAMPLE_RATE", 0)
SENTRY_PROFILES_SAMPLE_RATE = get_float("SENTRY_PROFILES_SAMPLE_RATE", 0)
init_sentry(
    dsn=SENTRY_DSN,
    environment=ENVIRONMENT,
    version=VERSION,
    log_level=SENTRY_LOG_LEVEL,
    traces_sample_rate=SENTRY_TRACES_SAMPLE_RATE,
    profiles_sample_rate=SENTRY_PROFILES_SAMPLE_RATE,
)

BASE_DIR = os.path.dirname(  # noqa: PTH120
    os.path.dirname(os.path.abspath(__file__))  # noqa: PTH100, PTH120
)

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/1.8/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = get_string("SECRET_KEY", "terribly_unsafe_default_secret_key")

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = get_bool("DEBUG", False)  # noqa: FBT003

ALLOWED_HOSTS = ["*"]

SECURE_SSL_REDIRECT = get_bool("MITOL_SECURE_SSL_REDIRECT", True)  # noqa: FBT003
SECURE_REDIRECT_EXEMPT = [
    "^health/startup/$",
    "^health/liveness/$",
    "^health/readiness/$",
    "^health/full/$",
]

SITE_ID = 1
APP_BASE_URL = get_string("MITOL_APP_BASE_URL", None)
if not APP_BASE_URL:
    msg = "MITOL_APP_BASE_URL is not set"
    raise ImproperlyConfigured(msg)
MITOL_TITLE = get_string("MITOL_TITLE", "MIT Learn")


# Application definition

INSTALLED_APPS = (
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.humanize",
    "django.contrib.sites",
    "django_filters",
    "django_scim",
    "rest_framework",
    "drf_spectacular",
    "oauth2_provider",
    "server_status",
    "anymail",
    "hijack",
    "hijack.contrib.admin",
    "guardian",
    "channels",
    # Put our apps after this point
    "main",
    "openapi",
    "users",
    "ai_chatbots",
    "health_check",
    "health_check.cache",
    "health_check.contrib.migrations",
    "health_check.contrib.celery_ping",
    "health_check.contrib.redis",
    "health_check.contrib.db_heartbeat",
)

HEALTH_CHECK = {
    "SUBSETS": {
        # The 'startup' subset includes checks that must pass before the application can
        # start.
        "startup": [
            "MigrationsHealthCheck",  # Ensures database migrations are applied.
            "CacheBackend",  # Verifies the cache backend is operational.
            "RedisHealthCheck",  # Confirms Redis is reachable and functional.
            "DatabaseHeartBeatCheck",  # Checks the database connection is alive.
        ],
        # The 'liveness' subset includes checks to determine if the application is
        # running.
        "liveness": ["DatabaseHeartBeatCheck"],  # Minimal check to ensure the app is
        # alive.
        # The 'readiness' subset includes checks to determine if the application is
        # ready to serve requests.
        "readiness": [
            "CacheBackend",  # Ensures the cache is ready for use.
            "RedisHealthCheck",  # Confirms Redis is ready for use.
            "DatabaseHeartBeatCheck",  # Verifies the database is ready for queries.
        ],
        # The 'full' subset includes all available health checks for a comprehensive
        # status report.
        "full": [
            "MigrationsHealthCheck",  # Ensures database migrations are applied.
            "CacheBackend",  # Verifies the cache backend is operational.
            "RedisHealthCheck",  # Confirms Redis is reachable and functional.
            "DatabaseHeartBeatCheck",  # Checks the database connection is alive.
            "CeleryPingHealthCheck",  # Verifies Celery workers are responsive.
        ],
    }
}

if not get_bool("RUN_DATA_MIGRATIONS", default=False):
    MIGRATION_MODULES = {"data_fixtures": None}

# Use the custom user model.
AUTH_USER_MODEL = "users.User"

# OAuth2TokenMiddleware must be before SCIMAuthCheckMiddleware
# in order to insert request.user into the request.
MIDDLEWARE = (
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "main.middleware.apisix_user.ApisixUserMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "hijack.middleware.HijackUserMiddleware",
)

# CORS--- Used when initializing starlette.middleware.cors
CORS_ALLOW_ORIGINS = get_list_of_str("CORS_ALLOW_ORIGINS", [])
CORS_ALLOW_METHODS = get_list_of_str(
    "CORS_ALLOW_METHODS",
    [
        "POST",
        "OPTIONS",
    ],
)
CORS_ALLOW_CREDENTIALS = get_bool("CORS_ALLOW_CREDENTIALS", True)  # noqa: FBT003
CORS_ALLOW_HEADERS = (
    # defaults
    "accept",
    "authorization",
    "content-type",
    "user-agent",
    "x-csrftoken",
    "x-requested-with",
    # sentry tracing
    "baggage",
    "sentry-trace",
)
CORS_ALLOWED_ORIGINS = get_list_of_str("CORS_ALLOWED_ORIGINS", [])
CORS_ALLOWED_ORIGIN_REGEXES = get_list_of_str("CORS_ALLOWED_ORIGIN_REGEXES", [])

SECURE_CROSS_ORIGIN_OPENER_POLICY = get_string(
    "SECURE_CROSS_ORIGIN_OPENER_POLICY",
    "same-origin",
)

CSRF_COOKIE_SECURE = get_bool("CSRF_COOKIE_SECURE", True)  # noqa: FBT003
CSRF_COOKIE_DOMAIN = get_string("CSRF_COOKIE_DOMAIN", None)
CSRF_COOKIE_NAME = get_string("CSRF_COOKIE_NAME", "csrftoken")

CSRF_HEADER_NAME = get_string("CSRF_HEADER_NAME", "HTTP_X_CSRFTOKEN")

CSRF_TRUSTED_ORIGINS = get_list_of_str("CSRF_TRUSTED_ORIGINS", [])

SESSION_COOKIE_DOMAIN = get_string("SESSION_COOKIE_DOMAIN", None)
SESSION_COOKIE_NAME = get_string("SESSION_COOKIE_NAME", "sessionid")


SESSION_ENGINE = "django.contrib.sessions.backends.signed_cookies"

ROOT_URLCONF = "main.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [
            BASE_DIR + "/templates/",
        ],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "social_django.context_processors.backends",
                "social_django.context_processors.login_redirect",
            ]
        },
    }
]

ASGI_APPLICATION = "main.asgi.application"

# Database
# https://docs.djangoproject.com/en/1.8/ref/settings/#databases
# Uses DATABASE_URL to configure with sqlite default:
# For URL structure:
# https://github.com/kennethreitz/dj-database-url
DEFAULT_DATABASE_CONFIG = dj_database_url.parse(
    get_string(
        "DATABASE_URL",
        "sqlite:///{}".format(os.path.join(BASE_DIR, "db.sqlite3")),  # noqa: PTH118
    )
)
DEFAULT_DATABASE_CONFIG["DISABLE_SERVER_SIDE_CURSORS"] = get_bool(
    "MITOL_DB_DISABLE_SS_CURSORS",
    True,  # noqa: FBT003
)
DEFAULT_DATABASE_CONFIG["CONN_MAX_AGE"] = get_int("MITOL_DB_CONN_MAX_AGE", 0)

if get_bool("MITOL_DB_DISABLE_SSL", False):  # noqa: FBT003
    DEFAULT_DATABASE_CONFIG["OPTIONS"] = {}
else:
    DEFAULT_DATABASE_CONFIG["OPTIONS"] = {"sslmode": "require"}

DATABASES = {"default": DEFAULT_DATABASE_CONFIG}

EXTERNAL_MODELS = ["programcertificate"]

# Internationalization
# https://docs.djangoproject.com/en/1.8/topics/i18n/

LANGUAGE_CODE = "en-us"

TIME_ZONE = "UTC"

USE_I18N = True

USE_L10N = True

USE_TZ = True

# Social Auth configurations - [START]
AUTHENTICATION_BACKENDS = (
    # the following needs to stay here to allow login of local users
    "django.contrib.auth.backends.ModelBackend",
    "guardian.backends.ObjectPermissionBackend",
)


# https://docs.djangoproject.com/en/1.8/howto/static-files/
STATIC_URL = "/static/"
CLOUDFRONT_DIST = get_string("CLOUDFRONT_DIST", None)
if CLOUDFRONT_DIST:
    STATIC_URL = urljoin(f"https://{CLOUDFRONT_DIST}.cloudfront.net", STATIC_URL)

STATIC_ROOT = "staticfiles"
STATICFILES_DIRS = [os.path.join(BASE_DIR, "static")]  # noqa: PTH118

# Important to define this so DEBUG works properly
INTERNAL_IPS = (get_string("HOST_IP", "127.0.0.1"),)

NOTIFICATION_ATTEMPT_RATE_LIMIT = "600/m"

NOTIFICATION_ATTEMPT_CHUNK_SIZE = 100

# Configure e-mail settings
EMAIL_BACKEND = get_string(
    "MITOL_EMAIL_BACKEND", "django.core.mail.backends.smtp.EmailBackend"
)
EMAIL_HOST = get_string("MITOL_EMAIL_HOST", "localhost")
EMAIL_PORT = get_int("MITOL_EMAIL_PORT", 25)
EMAIL_HOST_USER = get_string("MITOL_EMAIL_USER", "")
EMAIL_HOST_PASSWORD = get_string("MITOL_EMAIL_PASSWORD", "")
EMAIL_USE_TLS = get_bool("MITOL_EMAIL_TLS", False)  # noqa: FBT003
EMAIL_SUPPORT = get_string("MITOL_SUPPORT_EMAIL", "support@example.com")
DEFAULT_FROM_EMAIL = get_string("MITOL_FROM_EMAIL", "webmaster@localhost")

MAILGUN_SENDER_DOMAIN = get_string("MAILGUN_SENDER_DOMAIN", None)
if not MAILGUN_SENDER_DOMAIN:
    msg = "MAILGUN_SENDER_DOMAIN not set"
    raise ImproperlyConfigured(msg)
MAILGUN_KEY = get_string("MAILGUN_KEY", None)
if not MAILGUN_KEY:
    msg = "MAILGUN_KEY not set"
    raise ImproperlyConfigured(msg)
MAILGUN_RECIPIENT_OVERRIDE = get_string("MAILGUN_RECIPIENT_OVERRIDE", None)
MAILGUN_FROM_EMAIL = get_string("MAILGUN_FROM_EMAIL", "no-reply@example.com")
MAILGUN_BCC_TO_EMAIL = get_string("MAILGUN_BCC_TO_EMAIL", None)

ANYMAIL = {
    "MAILGUN_API_KEY": MAILGUN_KEY,
    "MAILGUN_SENDER_DOMAIN": MAILGUN_SENDER_DOMAIN,
}

NOTIFICATION_EMAIL_BACKEND = get_string(
    "MITOL_NOTIFICATION_EMAIL_BACKEND", "anymail.backends.test.EmailBackend"
)
# e-mail configurable admins
ADMIN_EMAIL = get_string("MITOL_ADMIN_EMAIL", "")
ADMINS = (("Admins", ADMIN_EMAIL),) if ADMIN_EMAIL != "" else ()

# embed.ly configuration
EMBEDLY_KEY = get_string("EMBEDLY_KEY", None)
EMBEDLY_EMBED_URL = get_string("EMBEDLY_EMBED_URL", "https://api.embed.ly/1/oembed")
EMBEDLY_EXTRACT_URL = get_string("EMBEDLY_EMBED_URL", "https://api.embed.ly/1/extract")

# Logging configuration
LOG_LEVEL = get_string("MITOL_LOG_LEVEL", "INFO")
DJANGO_LOG_LEVEL = get_string("DJANGO_LOG_LEVEL", "INFO")
OS_LOG_LEVEL = get_string("OS_LOG_LEVEL", "INFO")

# For logging to a remote syslog host
LOG_HOST = get_string("MITOL_LOG_HOST", "localhost")
LOG_HOST_PORT = get_int("MITOL_LOG_HOST_PORT", 514)

HOSTNAME = platform.node().split(".")[0]

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "filters": {"require_debug_false": {"()": "django.utils.log.RequireDebugFalse"}},
    "formatters": {
        "verbose": {
            "format": (
                "[%(asctime)s] %(levelname)s %(process)d [%(name)s] "
                "%(filename)s:%(lineno)d - "
                f"[{HOSTNAME}] - %(message)s"
            ),
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    },
    "handlers": {
        "console": {
            "level": LOG_LEVEL,
            "class": "logging.StreamHandler",
            "formatter": "verbose",
        },
        "syslog": {
            "level": LOG_LEVEL,
            "class": "logging.handlers.SysLogHandler",
            "facility": "local7",
            "formatter": "verbose",
            "address": (LOG_HOST, LOG_HOST_PORT),
        },
        "mail_admins": {
            "level": "ERROR",
            "filters": ["require_debug_false"],
            "class": "django.utils.log.AdminEmailHandler",
        },
    },
    "loggers": {
        "django": {
            "propagate": True,
            "level": DJANGO_LOG_LEVEL,
            "handlers": ["console", "syslog"],
        },
        "django.request": {
            "handlers": ["mail_admins"],
            "level": DJANGO_LOG_LEVEL,
            "propagate": True,
        },
        "opensearch": {"level": OS_LOG_LEVEL},
        "nplusone": {"handlers": ["console"], "level": "ERROR"},
        "boto3": {"handlers": ["console"], "level": "ERROR"},
    },
    "root": {"handlers": ["console", "syslog"], "level": LOG_LEVEL},
}

STATUS_TOKEN = get_string("STATUS_TOKEN", "")

GA_TRACKING_ID = get_string("GA_TRACKING_ID", "")
GA_G_TRACKING_ID = get_string("GA_G_TRACKING_ID", "")

REACT_GA_DEBUG = get_bool("REACT_GA_DEBUG", False)  # noqa: FBT003

RECAPTCHA_SITE_KEY = get_string("RECAPTCHA_SITE_KEY", "")
RECAPTCHA_SECRET_KEY = get_string("RECAPTCHA_SECRET_KEY", "")

MEDIA_ROOT = get_string("MEDIA_ROOT", "/var/media/")
MEDIA_URL = "/media/"
MITOL_USE_S3 = get_bool("MITOL_USE_S3", False)  # noqa: FBT003
AWS_ACCESS_KEY_ID = get_string("AWS_ACCESS_KEY_ID", False)  # noqa: FBT003
AWS_SECRET_ACCESS_KEY = get_string("AWS_SECRET_ACCESS_KEY", False)  # noqa: FBT003
AWS_STORAGE_BUCKET_NAME = get_string("AWS_STORAGE_BUCKET_NAME", False)  # noqa: FBT003
AWS_QUERYSTRING_AUTH = get_string("AWS_QUERYSTRING_AUTH", False)  # noqa: FBT003
# Provide nice validation of the configuration
if MITOL_USE_S3 and (
    not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY or not AWS_STORAGE_BUCKET_NAME
):
    msg = "You have enabled S3 support, but are missing one of AWS_ACCESS_KEY_ID, \
    AWS_SECRET_ACCESS_KEY, or AWS_STORAGE_BUCKET_NAME"
    raise ImproperlyConfigured(msg)
if MITOL_USE_S3:
    DEFAULT_FILE_STORAGE = "storages.backends.s3boto3.S3Boto3Storage"


# django cache back-ends
CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
        "LOCATION": "local-in-memory-cache",
    },
    # cache specific to widgets
    "external_assets": {
        "BACKEND": "django.core.cache.backends.db.DatabaseCache",
        "LOCATION": "external_asset_cache",
    },
    # general durable cache (redis should be considered ephemeral)
    "durable": {
        "BACKEND": "django.core.cache.backends.db.DatabaseCache",
        "LOCATION": "durable_cache",
    },
    "redis": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": CELERY_BROKER_URL,
        "OPTIONS": {"CLIENT_CLASS": "django_redis.client.DefaultClient"},
    },
}

# JWT authentication settings
MITOL_JWT_SECRET = get_string(
    "MITOL_JWT_SECRET", "terribly_unsafe_default_jwt_secret_key"
)

MITOL_COOKIE_NAME = get_string("MITOL_COOKIE_NAME", None)
if not MITOL_COOKIE_NAME:
    msg = "MITOL_COOKIE_NAME is not set"
    raise ImproperlyConfigured(msg)
MITOL_COOKIE_DOMAIN = get_string("MITOL_COOKIE_DOMAIN", None)
if not MITOL_COOKIE_DOMAIN:
    msg = "MITOL_COOKIE_DOMAIN is not set"
    raise ImproperlyConfigured(msg)

MITOL_UNSUBSCRIBE_TOKEN_MAX_AGE_SECONDS = get_int(
    "MITOL_UNSUBSCRIBE_TOKEN_MAX_AGE_SECONDS",
    60 * 60 * 24 * 7,  # 7 days
)

JWT_AUTH = {
    "JWT_SECRET_KEY": MITOL_JWT_SECRET,
    "JWT_VERIFY": True,
    "JWT_VERIFY_EXPIRATION": True,
    "JWT_EXPIRATION_DELTA": datetime.timedelta(seconds=60 * 60),
    "JWT_ALLOW_REFRESH": True,
    "JWT_REFRESH_EXPIRATION_DELTA": datetime.timedelta(days=7),
    "JWT_AUTH_COOKIE": MITOL_COOKIE_NAME,
    "JWT_AUTH_HEADER_PREFIX": "Bearer",
}


# features flags
def get_all_config_keys():
    """
    Returns all the configuration keys from both
    environment and configuration files
    """  # noqa: D401
    return list(os.environ.keys())


MITOL_FEATURES_PREFIX = get_string("MITOL_FEATURES_PREFIX", "FEATURE_")
MITOL_FEATURES_DEFAULT = get_bool("MITOL_FEATURES_DEFAULT", False)  # noqa: FBT003
FEATURES = {
    key[len(MITOL_FEATURES_PREFIX) :]: get_bool(key, False)  # noqa: FBT003
    for key in get_all_config_keys()
    if key.startswith(MITOL_FEATURES_PREFIX)
}

MIDDLEWARE_FEATURE_FLAG_QS_PREFIX = get_string(
    "MIDDLEWARE_FEATURE_FLAG_QS_PREFIX", None
)
MIDDLEWARE_FEATURE_FLAG_COOKIE_NAME = get_string(
    "MIDDLEWARE_FEATURE_FLAG_COOKIE_NAME", "MITOL_FEATURE_FLAGS"
)
MIDDLEWARE_FEATURE_FLAG_COOKIE_MAX_AGE_SECONDS = get_int(
    "MIDDLEWARE_FEATURE_FLAG_COOKIE_MAX_AGE_SECONDS", 60 * 60
)
SEARCH_PAGE_CACHE_DURATION = get_int("SEARCH_PAGE_CACHE_DURATION", 60 * 60 * 24)
if MIDDLEWARE_FEATURE_FLAG_QS_PREFIX:
    MIDDLEWARE = (
        *MIDDLEWARE,
        "main.middleware.feature_flags.QueryStringFeatureFlagMiddleware",
        "main.middleware.feature_flags.CookieFeatureFlagMiddleware",
    )

# django debug toolbar only in debug mode
if DEBUG:
    INSTALLED_APPS += ("debug_toolbar",)
    # it needs to be enabled before other middlewares
    MIDDLEWARE = ("debug_toolbar.middleware.DebugToolbarMiddleware", *MIDDLEWARE)


REST_FRAMEWORK = {
    "DEFAULT_PERMISSION_CLASSES": ("rest_framework.permissions.IsAuthenticated",),
    "DEFAULT_AUTHENTICATION_CLASSES": (
        "rest_framework.authentication.SessionAuthentication",
    ),
    "EXCEPTION_HANDLER": "main.exceptions.api_exception_handler",
    "TEST_REQUEST_DEFAULT_FORMAT": "json",
    "TEST_REQUEST_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
        "rest_framework.renderers.MultiPartRenderer",
    ],
    "DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema",
    "DEFAULT_VERSIONING_CLASS": "rest_framework.versioning.NamespaceVersioning",
    "ALLOWED_VERSIONS": ["v0"],
    "ORDERING_PARAM": "sortby",
}


USE_X_FORWARDED_PORT = get_bool("USE_X_FORWARDED_PORT", False)  # noqa: FBT003
USE_X_FORWARDED_HOST = get_bool("USE_X_FORWARDED_HOST", False)  # noqa: FBT003

# Hijack
HIJACK_ALLOW_GET_REQUESTS = True
HIJACK_LOGOUT_REDIRECT_URL = "/admin/auth/user"

# Guardian
# disable the anonymous user creation
ANONYMOUS_USER_NAME = None

REQUESTS_TIMEOUT = get_int("REQUESTS_TIMEOUT", 30)

CELERY_BEAT_SCHEDULE = {
    "delete_stale_chat_sessions": {
        "task": "ai_chatbots.tasks.delete_stale_sessions",
        "schedule": crontab(minute=0, hour=4),
    },
}


if DEBUG:
    # allow for all IPs to be routable, including localhost, for testing
    IPWARE_PRIVATE_IP_PREFIX = ()

KEYCLOAK_BASE_URL = get_string(
    name="KEYCLOAK_BASE_URL",
    default="http://mit-keycloak-base-url.edu",
)
KEYCLOAK_REALM_NAME = get_string(
    name="KEYCLOAK_REALM_NAME",
    default="olapps",
)

POSTHOG_PROJECT_API_KEY = get_string(
    name="POSTHOG_PROJECT_API_KEY",
    default="",
)
POSTHOG_PERSONAL_API_KEY = get_string(
    name="POSTHOG_PERSONAL_API_KEY",
    default=None,
)
POSTHOG_API_HOST = get_string(
    name="POSTHOG_API_HOST",
    default="https://us.posthog.com",
)
POSTHOG_TIMEOUT_MS = get_int(
    name="POSTHOG_TIMEOUT_MS",
    default=3000,
)
POSTHOG_PROJECT_ID = get_int(
    name="POSTHOG_PROJECT_ID",
    default=None,
)


# Keycloak API settings
KEYCLOAK_ADMIN_CLIENT_ID = get_string("KEYCLOAK_ADMIN_CLIENT_ID", False)  # noqa: FBT003
KEYCLOAK_ADMIN_CLIENT_SECRET = get_string("KEYCLOAK_ADMIN_CLIENT_SECRET", False)  # noqa: FBT003
KEYCLOAK_REALM = get_string("KEYCLOAK_REALM", False)  # noqa: FBT003
KEYCLOAK_ADMIN_URL = get_string("KEYCLOAK_ADMIN_URL", False)  # noqa: FBT003
KEYCLOAK_ADMIN_SECURE = get_bool("KEYCLOAK_ADMIN_SECURE", True)  # noqa: FBT003

# Channel settings
REDIS_DOMAIN = get_string("REDIS_DOMAIN", "redis://redis:6379/0")
REDIS_SSL_CERT_REQS = get_string("REDIS_SSL_CERT_REQS", None)
REDIS_HOST_PARAMS = (
    {"ssl_cert_reqs": REDIS_SSL_CERT_REQS} if REDIS_SSL_CERT_REQS else {}
)
CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels_redis.core.RedisChannelLayer",
        "CONFIG": {"hosts": [{"address": REDIS_DOMAIN, **REDIS_HOST_PARAMS}]},
    },
}

# AI settings
AI_DEBUG = get_bool("AI_DEBUG", False)  # noqa: FBT003
AI_MIT_SEARCH_VECTOR_URL = get_string(
    name="AI_MIT_SEARCH_VECTOR_URL",
    default="https://api.learn.mit.edu/api/v0/vector_learning_resources_search/",
)
AI_MIT_SEARCH_ELASTIC_URL = get_string(
    name="AI_MIT_SEARCH_ELASTIC_URL",
    default="https://api.learn.mit.edu/api/v1/learning_resources_search/",
)
AI_MIT_SEARCH_URL = get_string(
    name="AI_MIT_SEARCH_URL",
    default=AI_MIT_SEARCH_ELASTIC_URL,
)
AI_MIT_SEARCH_DETAIL_URL = get_string(
    name="AI_MIT_SEARCH_DETAIL_URL",
    default="https://learn.mit.edu/?resource=",
)
AI_MIT_SYLLABUS_URL = get_string(
    "AI_MIT_SYLLABUS_URL",
    "https://api.learn.mit.edu/api/v0/vector_content_files_search/",
)
AI_MIT_VIDEO_TRANSCRIPT_URL = get_string(
    "AI_MIT_VIDEO_TRANSCRIPT_URL",
    "https://api.learn.mit.edu/api/v0/vector_content_files_search/",
)
AI_MIT_CONTENTFILE_URL = get_string(
    name="AI_MIT_CONTENTFILE_URL",
    default="https://api.learn.mit.edu/api/v1/contentfiles/",
)
LEARN_ACCESS_TOKEN = get_string(
    name="LEARN_ACCESS_TOKEN",
    default="",
)
AI_MIT_SEARCH_LIMIT = get_int(name="AI_MIT_SEARCH_LIMIT", default=10)
AI_MIT_CONTENT_SEARCH_LIMIT = get_int(name="AI_MIT_CONTENT_SEARCH_LIMIT", default=20)
AI_MIT_TRANSCRIPT_SEARCH_LIMIT = get_int(name="AI_MIT_CONTENT_SEARCH_LIMIT", default=5)
AI_DEFAULT_MODEL = get_string(name="AI_DEFAULT_MODEL", default="openai/gpt-4o")
AI_DEFAULT_RECOMMENDATION_MODEL = get_string(
    "AI_DEFAULT_RECOMMENDATION_MODEL", AI_DEFAULT_MODEL
)
AI_DEFAULT_RECOMMENDATION_MAX_TOKENS = get_int(
    "AI_DEFAULT_RECOMMENDATION_MAX_TOKENS", 7500
)
AI_DEFAULT_SUMMARY_MODEL = get_string("AI_DEFAULT_SUMMARY_MODEL", AI_DEFAULT_MODEL)
AI_DEFAULT_SYLLABUS_MODEL = get_string("AI_DEFAULT_SYLLABUS_MODEL", AI_DEFAULT_MODEL)
# AI agents that return lengthy tool/search results need larger token limits
AI_DEFAULT_SYLLABUS_MAX_TOKENS = get_int("AI_DEFAULT_SYLLABUS_MAX_TOKENS", 16384)
AI_DEFAULT_TUTOR_MODEL = get_string("AI_DEFAULT_TUTOR_MODEL", AI_DEFAULT_MODEL)
AI_DEFAULT_VIDEO_GPT_MODEL = get_string("AI_DEFAULT_VIDEO_GPT_MODEL", AI_DEFAULT_MODEL)
AI_DEFAULT_VIDEO_GPT_MAX_TOKENS = get_int("AI_DEFAULT_VIDEO_GPT_MAX_TOKENS", 16384)
AI_DEFAULT_TEMPERATURE = get_float(name="AI_DEFAULT_TEMPERATURE", default=0.1)
OPENAI_API_KEY = get_string(name="OPENAI_API_KEY", default="")
AI_CHATBOTS_SESSION_EXPIRY_DAYS = get_int(
    name="AI_CHATBOTS_SESSION_EXPIRY_DAYS", default=10
)
AI_CHATBOTS_COOKIE_MAX_AGE = get_int(
    name="AI_CHATBOTS_COOKIE_MAX_AGE", default=24 * 60 * 60 * 7
)
AI_MAX_TOKEN_BIND = get_int(name="AI_MAX_TOKEN_BIND", default=16384)
AI_PROMPT_CACHE_DURATION = get_int(
    name="AI_PROMPT_CACHE_DURATION", default=60 * 60 * 24 * 28
)  # 28 days

# AI proxy settings (aka LiteLLM)
AI_PROXY_CLASS = get_string(name="AI_PROXY_CLASS", default="")
AI_PROXY_URL = get_string(name="AI_PROXY_URL", default="")
AI_PROXY_AUTH_TOKEN = get_string(name="AI_PROXY_AUTH_TOKEN", default="")
AI_MAX_PARALLEL_REQUESTS = get_int(name="AI_MAX_PARALLEL_REQUESTS", default=10)
AI_TPM_LIMIT = get_int(name="AI_TPM_LIMIT", default=5000)
AI_RPM_LIMIT = get_int(name="AI_RPM_LIMIT", default=10)
AI_BUDGET_DURATION = get_string(name="AI_BUDGET_DURATION", default="60m")
AI_MAX_BUDGET = get_float(name="AI_MAX_BUDGET", default=0.05)
AI_ANON_LIMIT_MULTIPLIER = get_float(name="AI_ANON_LIMIT_MULTIPLIER", default=10.0)
AI_UNSUPPORTED_TEMP_MODELS = get_list_of_str(
    name="AI_UNSUPPORTED_TEMP_MODELS", default=["openai/o3-mini"]
)
AI_PROMPT_CACHE_FUNCTION = get_string(
    name="AI_PROMPT_CACHE_FUNCTION", default="ai_chatbots.utils.get_django_cache"
)
LANGSMITH_API_KEY = get_string(name="LANGSMITH_API_KEY", default=None)

# APISIX middleware settings
APISIX_USERDATA_MAP = {
    "users_user": {
        "email": "email",
        "global_id": "sub",
        "username": "preferred_username",
        "name": "name",
    },
    "users_userprofile": {
        "country_code": None,
    },
}

# OpenTelemetry configuration
OPENTELEMETRY_ENABLED = get_bool("OPENTELEMETRY_ENABLED", default=False)
OPENTELEMETRY_SERVICE_NAME = get_string("OPENTELEMETRY_SERVICE_NAME", "learn-ai")
OPENTELEMETRY_INSECURE = get_bool("OPENTELEMETRY_INSECURE", default=True)
OPENTELEMETRY_ENDPOINT = get_string("OPENTELEMETRY_ENDPOINT", None)
OPENTELEMETRY_BATCH_SIZE = get_int("OPENTELEMETRY_BATCH_SIZE", 512)
OPENTELEMETRY_EXPORT_TIMEOUT_MS = get_int("OPENTELEMETRY_EXPORT_TIMEOUT_MS", 5000)

SPECTACULAR_SETTINGS = open_spectacular_settings


CONSUMER_THROTTLE_CLASSES = get_list_of_str(
    "CONSUMER_THROTTLE_CLASSES",
    ["main.consumer_throttles.UserScopedRateThrottle"],
)
