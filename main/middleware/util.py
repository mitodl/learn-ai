"""Middleware utilities"""


def apply_middleware(middlewares, app):
    """
    Apply middleware to an ASGI application.
    """
    for middleware in middlewares:
        app = middleware(app)
    return app
