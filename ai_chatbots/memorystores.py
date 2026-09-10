"""Django-backed LangGraph BaseStore for per-learner long-term memory."""

from collections.abc import Iterable

from asgiref.sync import sync_to_async
from django.contrib.auth import get_user_model
from langgraph.store.base import (
    BaseStore,
    GetOp,
    InvalidNamespaceError,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)

from ai_chatbots.models import LearnerMemoryNote

ROOT = "memories"


def _global_id(namespace: tuple[str, ...]) -> str:
    """Namespaces are exactly ("memories", global_id)."""
    if len(namespace) != 2 or namespace[0] != ROOT or not namespace[1]:  # noqa: PLR2004
        msg = f"Expected ('{ROOT}', global_id), got {namespace!r}"
        raise InvalidNamespaceError(msg)
    return namespace[1]


def _item(row: LearnerMemoryNote, cls: type[Item] = Item) -> Item:
    return cls(
        value={"text": row.text},
        key=row.key,
        namespace=(ROOT, row.user.global_id),
        created_at=row.created_on,
        updated_at=row.updated_on,
    )


class DjangoMemoryStore(BaseStore):
    """
    LangGraph store whose items are LearnerMemoryNote rows.

    Namespace ("memories", global_id), key = note section, value {"text": ...};
    put(value=None) deletes. The user FK on the row is what makes "forget me"
    a cascade delete, which LangGraph's own PostgresStore can't offer.

    ponytail: no embedding index; search ignores `query` and returns the namespace
    contents. Synchronous ORM, so a caller's transaction.atomic() covers the writes.
    Only `batch`/`abatch` are abstract on BaseStore; everything else fans into them.
    """

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        return [self._run(op) for op in ops]

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        return await sync_to_async(self.batch)(list(ops))

    def _run(self, op: Op) -> Result:
        if isinstance(op, GetOp):
            row = (
                LearnerMemoryNote.objects.filter(
                    user__global_id=_global_id(op.namespace), key=op.key
                )
                .select_related("user")
                .first()
            )
            return _item(row) if row else None
        if isinstance(op, PutOp):
            self._put(op)
            return None
        if isinstance(op, SearchOp):
            return self._search(op)
        if isinstance(op, ListNamespacesOp):
            return self._list_namespaces(op)
        msg = f"Unsupported op {type(op).__name__}"
        raise TypeError(msg)

    def _put(self, op: PutOp) -> None:
        global_id = _global_id(op.namespace)
        if op.value is None:
            LearnerMemoryNote.objects.filter(
                user__global_id=global_id, key=op.key
            ).delete()
            return
        user = get_user_model().objects.get(global_id=global_id)
        LearnerMemoryNote.objects.update_or_create(
            user=user, key=op.key, defaults={"text": op.value["text"]}
        )

    def _search(self, op: SearchOp) -> list[SearchItem]:
        prefix = op.namespace_prefix
        if not prefix or prefix[0] != ROOT or len(prefix) > 2:  # noqa: PLR2004
            return []
        qs = LearnerMemoryNote.objects.select_related("user").order_by(
            "user__global_id", "key"
        )
        if len(prefix) == 2:  # noqa: PLR2004
            qs = qs.filter(user__global_id=prefix[1])
        # ponytail: equality filter on the one value field; langgraph's $gt/$in unused
        for field, expected in (op.filter or {}).items():
            qs = qs.filter(**{field: expected})
        rows = qs[op.offset : op.offset + op.limit]
        return [_item(row, SearchItem) for row in rows]

    def _list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        global_ids = (
            LearnerMemoryNote.objects.order_by("user__global_id")
            .values_list("user__global_id", flat=True)
            .distinct()
        )
        namespaces = [(ROOT, gid) for gid in global_ids]
        for cond in op.match_conditions or ():
            width = len(cond.path)
            if cond.match_type == "prefix":
                namespaces = [n for n in namespaces if n[:width] == cond.path]
            else:
                namespaces = [n for n in namespaces if n[-width:] == cond.path]
        if op.max_depth is not None:
            namespaces = list(dict.fromkeys(n[: op.max_depth] for n in namespaces))
        return namespaces[op.offset : op.offset + op.limit]
