"""Tests for the Django-backed LangGraph store"""

import pytest

from ai_chatbots.memorystores import DjangoMemoryStore
from ai_chatbots.models import MemoryStoreItem

pytestmark = pytest.mark.django_db


def test_put_get_roundtrip():
    """A stored value comes back as an Item under the same namespace and key"""
    store = DjangoMemoryStore()
    store.put(("memories", "u1"), "default", {"kind": "M", "content": {"a": 1}})
    item = store.get(("memories", "u1"), "default")
    assert item.namespace == ("memories", "u1")
    assert item.key == "default"
    assert item.value == {"kind": "M", "content": {"a": 1}}
    assert item.created_at <= item.updated_at


def test_put_overwrites_in_place():
    """Putting the same namespace and key again replaces the value, not adds a row"""
    store = DjangoMemoryStore()
    store.put(("memories", "u1"), "default", {"v": 1})
    store.put(("memories", "u1"), "default", {"v": 2})
    assert MemoryStoreItem.objects.count() == 1
    assert store.get(("memories", "u1"), "default").value == {"v": 2}


def test_get_missing_returns_none():
    """Missing keys return None rather than raising"""
    assert DjangoMemoryStore().get(("memories", "nobody"), "default") is None


def test_namespaces_are_isolated():
    """One learner's memory is never visible under another learner's namespace"""
    store = DjangoMemoryStore()
    store.put(("memories", "u1"), "default", {"who": "u1"})
    store.put(("memories", "u2"), "default", {"who": "u2"})
    assert store.get(("memories", "u2"), "default").value == {"who": "u2"}
    assert [i.value for i in store.search(("memories", "u1"))] == [{"who": "u1"}]
    assert store.search(("memories", "u3")) == []


def test_search_prefix_and_filter():
    """Search matches namespace prefixes and applies equality filters on the value"""
    store = DjangoMemoryStore()
    store.put(("memories", "u1"), "a", {"kind": "x", "n": 1})
    store.put(("memories", "u1", "sub"), "b", {"kind": "y", "n": 2})
    store.put(("memories", "u10"), "c", {"kind": "x", "n": 3})
    found = store.search(("memories", "u1"))
    assert sorted(i.key for i in found) == ["a", "b"]
    assert [i.key for i in store.search(("memories", "u1"), filter={"kind": "y"})] == [
        "b"
    ]
    assert [i.key for i in store.search(("memories", "u1"), limit=1, offset=1)] == ["b"]


def test_search_with_query_ignores_query():
    """No embedding index: a query returns the namespace contents unranked"""
    store = DjangoMemoryStore()
    store.put(("memories", "u1"), "default", {"v": 1})
    results = store.search(("memories", "u1"), query="anything")
    assert [i.key for i in results] == ["default"]
    assert results[0].score is None


def test_delete():
    """Delete removes the row; deleting a missing key is a no-op"""
    store = DjangoMemoryStore()
    store.put(("memories", "u1"), "default", {"v": 1})
    store.delete(("memories", "u1"), "default")
    store.delete(("memories", "u1"), "default")
    assert store.get(("memories", "u1"), "default") is None
    assert MemoryStoreItem.objects.count() == 0


def test_list_namespaces():
    """Namespaces list distinct tuples, honoring prefix and max_depth"""
    store = DjangoMemoryStore()
    store.put(("memories", "u1"), "a", {})
    store.put(("memories", "u1", "sub"), "b", {})
    store.put(("other", "u1"), "c", {})
    assert store.list_namespaces(prefix=("memories",)) == [
        ("memories", "u1"),
        ("memories", "u1", "sub"),
    ]
    assert store.list_namespaces(prefix=("memories",), max_depth=2) == [
        ("memories", "u1")
    ]
    assert store.list_namespaces(suffix=("sub",)) == [("memories", "u1", "sub")]


async def test_async_roundtrip():
    """The async methods work from an event loop"""
    store = DjangoMemoryStore()
    await store.aput(("memories", "u1"), "default", {"v": 1})
    assert (await store.aget(("memories", "u1"), "default")).value == {"v": 1}
    assert [i.key for i in await store.asearch(("memories", "u1"))] == ["default"]
    await store.adelete(("memories", "u1"), "default")
    assert await store.aget(("memories", "u1"), "default") is None
