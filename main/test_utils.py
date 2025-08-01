"""Testing utils"""

import abc
import json
import re
import traceback
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock

import pytest
from django.conf import settings
from django.http.response import HttpResponse
from rest_framework.renderers import JSONRenderer


def any_instance_of(*classes):
    """
    Returns a type that evaluates __eq__ in isinstance terms

    Args:
        classes (list of types): variable list of types to ensure equality against

    Returns:
        AnyInstanceOf: dynamic class type with the desired equality
    """  # noqa: D401

    class AnyInstanceOf(abc.ABC):  # noqa: B024
        """Dynamic class type for __eq__ in terms of isinstance"""

        def __init__(self, classes):
            self.classes = classes

        def __eq__(self, other):
            return isinstance(other, self.classes)

        def __str__(self):  # pragma: no cover
            return f"AnyInstanceOf({', '.join([str(c) for c in self.classes])})"

        def __repr__(self):  # pragma: no cover
            return str(self)

    for c in classes:
        AnyInstanceOf.register(c)
    return AnyInstanceOf(classes)


@contextmanager
def assert_not_raises():
    """Used to assert that the context does not raise an exception"""  # noqa: D401
    try:
        yield
    except AssertionError:
        raise
    except Exception:  # pylint: disable=broad-except  # noqa: BLE001
        pytest.fail(f"An exception was not raised: {traceback.format_exc()}")


class MockResponse(HttpResponse):
    """
    Mocked HTTP response object that can be used as a stand-in for request.Response and
    django.http.response.HttpResponse objects
    """

    def __init__(self, content, status_code):
        """
        Args:
            content (str): The response content
            status_code (int): the response status code
        """
        self.status_code = status_code
        self.decoded_content = content
        super().__init__(content=(content or "").encode("utf-8"), status=status_code)

    def json(self):
        """Return content as json"""
        return json.loads(self.decoded_content)


def drf_datetime(dt):
    """
    Returns a datetime formatted as a DRF DateTimeField formats it

    Args:
        dt(datetime): datetime to format

    Returns:
        str: ISO 8601 formatted datetime
    """  # noqa: D401
    return dt.isoformat().replace("+00:00", "Z")


def _sort_values_for_testing(obj):
    """
    Sort an object recursively if possible to do so

    Args:
        obj (any): A dict, list, or some other JSON type

    Returns:
        any: A sorted version of the object passed in, or the same object if no sorting can be done
    """
    if isinstance(obj, dict):
        return {key: _sort_values_for_testing(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        items = [_sort_values_for_testing(value) for value in obj]
        # this will produce incorrect results since everything is converted to a string
        # for example [10, 9] will be sorted like that
        # but here we only care that the items are compared in a consistent way so tests pass
        return sorted(items, key=json.dumps)
    else:
        return obj


def assert_json_equal(obj1, obj2, sort=False):  # noqa: FBT002
    """
    Asserts that two objects are equal after a round trip through JSON serialization/deserialization.
    Particularly helpful when testing DRF serializers where you may get back OrderedDict and other such objects.

    Args:
        obj1 (object): the first object
        obj2 (object): the second object
        sort (bool): If true, sort items which are iterable before comparing
    """  # noqa: D401
    renderer = JSONRenderer()
    converted1 = json.loads(renderer.render(obj1))
    converted2 = json.loads(renderer.render(obj2))
    if sort:
        converted1 = _sort_values_for_testing(converted1)
        converted2 = _sort_values_for_testing(converted2)
    assert converted1 == converted2


class PickleableMock(Mock):
    """
    A Mock that can be passed to pickle.dumps()

    Source: https://github.com/testing-cabal/mock/issues/139#issuecomment-122128815
    """

    def __reduce__(self):
        """Required method for being pickleable"""  # noqa: D401
        return (Mock, ())


def load_json_with_settings(file_path: str) -> dict:
    """Load JSON file and replace {{settings.SETTING_NAME}} with actual Django setting values."""
    with Path.open(file_path) as f:
        content = f.read()

    # Simple regex to find and replace {{settings.SETTING_NAME}} patterns
    def replace_setting(match):
        setting_name = match.group(1)
        return str(getattr(settings, setting_name, match.group(0)))

    # Replace all {{settings.SETTING_NAME}} patterns
    content = re.sub(r"\{\{settings\.([A-Z_][A-Z0-9_]*)\}\}", replace_setting, content)

    return json.loads(content)
