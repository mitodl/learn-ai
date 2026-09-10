"""Tests for the Django-backed LangGraph store"""

import pytest
from django.contrib.auth import get_user_model
from langgraph.store.base import InvalidNamespaceError

from ai_chatbots.memorystores import DjangoMemoryStore
from ai_chatbots.models import LearnerMemoryNote
from main.factories import UserFactory

pytestmark = pytest.mark.django_db


@pytest.fixture
def ns():
    """Namespace of a real learner; put() needs the user row to exist"""
    return ("memories", UserFactory.create().global_id)


def test_put_get_roundtrip(ns):
    """A stored value comes back as an Item under the same namespace and key"""
    store = DjangoMemoryStore()
    store.put(ns, "about", {"text": "nurse"})
    item = store.get(ns, "about")
    assert item.namespace == ns
    assert item.key == "about"
    assert item.value == {"text": "nurse"}
    assert item.created_at <= item.updated_at


def test_put_overwrites_in_place(ns):
    """Putting the same namespace and key again replaces the row, not adds one"""
    store = DjangoMemoryStore()
    store.put(ns, "about", {"text": "v1"})
    store.put(ns, "about", {"text": "v2"})
    assert LearnerMemoryNote.objects.count() == 1
    assert store.get(ns, "about").value == {"text": "v2"}


def test_rows_belong_to_the_user(ns):
    """The user FK is set, so deleting the user cascades to their notes"""
    store = DjangoMemoryStore()
    store.put(ns, "about", {"text": "nurse"})
    note = LearnerMemoryNote.objects.get()
    assert note.user.global_id == ns[1]
    note.user.delete()
    assert store.get(ns, "about") is None


def test_get_missing_returns_none(ns):
    """Missing keys and unknown learners return None rather than raising"""
    store = DjangoMemoryStore()
    assert store.get(ns, "about") is None
    assert store.get(("memories", "nobody"), "about") is None


def test_put_for_unknown_user_raises(ns):
    """Notes can't be created for a learner who doesn't exist"""
    with pytest.raises(get_user_model().DoesNotExist):
        DjangoMemoryStore().put(("memories", "nobody"), "about", {"text": "x"})


@pytest.mark.parametrize(
    "bad", [("memories",), ("other", "u1"), ("memories", "u1", "x")]
)
def test_namespace_shape_is_enforced(bad):
    """Only ('memories', global_id) is a valid namespace"""
    with pytest.raises(InvalidNamespaceError):
        DjangoMemoryStore().get(bad, "about")


def test_namespaces_are_isolated(ns):
    """One learner's memory is never visible under another learner's namespace"""
    store = DjangoMemoryStore()
    other = ("memories", UserFactory.create().global_id)
    store.put(ns, "about", {"text": "me"})
    store.put(other, "about", {"text": "them"})
    assert store.get(other, "about").value == {"text": "them"}
    assert [i.value for i in store.search(ns)] == [{"text": "me"}]
    assert store.search(("memories", "u3")) == []


def test_search_prefix_filter_and_paging(ns):
    """Search by learner or across all learners, with an equality filter on text"""
    store = DjangoMemoryStore()
    other = ("memories", UserFactory.create().global_id)
    store.put(ns, "about", {"text": "nurse"})
    store.put(ns, "instructions", {"text": "short"})
    store.put(other, "about", {"text": "poet"})
    assert sorted(i.key for i in store.search(ns)) == ["about", "instructions"]
    assert len(store.search(("memories",))) == 3
    assert [i.key for i in store.search(ns, filter={"text": "short"})] == [
        "instructions"
    ]
    assert [i.key for i in store.search(ns, limit=1, offset=1)] == ["instructions"]
    assert store.search(("other",)) == []


def test_search_with_query_ignores_query(ns):
    """No embedding index: a query returns the namespace contents unranked"""
    store = DjangoMemoryStore()
    store.put(ns, "about", {"text": "nurse"})
    results = store.search(ns, query="anything")
    assert [i.key for i in results] == ["about"]
    assert results[0].score is None


def test_delete(ns):
    """Delete removes the row; deleting a missing key is a no-op"""
    store = DjangoMemoryStore()
    store.put(ns, "about", {"text": "nurse"})
    store.delete(ns, "about")
    store.delete(ns, "about")
    assert store.get(ns, "about") is None
    assert LearnerMemoryNote.objects.count() == 0


def test_list_namespaces(ns):
    """One namespace per learner with notes, honoring prefix, suffix and max_depth"""
    store = DjangoMemoryStore()
    other = ("memories", UserFactory.create().global_id)
    store.put(ns, "about", {"text": "a"})
    store.put(ns, "instructions", {"text": "b"})
    store.put(other, "about", {"text": "c"})
    assert sorted(store.list_namespaces(prefix=("memories",))) == sorted([ns, other])
    assert store.list_namespaces(prefix=("memories",), max_depth=1) == [("memories",)]
    assert store.list_namespaces(suffix=(ns[1],)) == [ns]
    assert store.list_namespaces(prefix=("other",)) == []


@pytest.mark.django_db(transaction=True)
async def test_async_roundtrip(ns):
    """The async methods work from an event loop"""
    store = DjangoMemoryStore()
    await store.aput(ns, "about", {"text": "nurse"})
    assert (await store.aget(ns, "about")).value == {"text": "nurse"}
    assert [i.key for i in await store.asearch(ns)] == ["about"]
    await store.adelete(ns, "about")
    assert await store.aget(ns, "about") is None
