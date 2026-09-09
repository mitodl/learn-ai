"""Django-backed LangGraph BaseStore for per-learner long-term memory."""

from collections.abc import Iterable

from asgiref.sync import sync_to_async
from django.db.models import Q
from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)

from ai_chatbots.models import MemoryStoreItem

SEP = "/"


def _ns(namespace: tuple[str, ...]) -> str:
    return SEP.join(namespace)


def _item(row: MemoryStoreItem, cls: type[Item] = Item) -> Item:
    return cls(
        value=row.value,
        key=row.key,
        namespace=tuple(row.namespace.split(SEP)),
        created_at=row.created_on,
        updated_at=row.updated_on,
    )


def _prefix_q(prefix: tuple[str, ...]) -> Q:
    ns = _ns(prefix)
    return Q(namespace=ns) | Q(namespace__startswith=ns + SEP)


class DjangoMemoryStore(BaseStore):
    """
    BaseStore whose items live in the MemoryStoreItem table.

    ponytail: no embedding index; search ignores `query` and returns the namespace
    contents. Add a vector column if memory ever becomes many items per learner.
    Only `batch`/`abatch` are abstract on BaseStore; everything else fans into them.
    """

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        return [self._run(op) for op in ops]

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        return await sync_to_async(self.batch)(list(ops))

    def _run(self, op: Op) -> Result:
        if isinstance(op, GetOp):
            row = MemoryStoreItem.objects.filter(
                namespace=_ns(op.namespace), key=op.key
            ).first()
            return _item(row) if row else None
        if isinstance(op, PutOp):
            if op.value is None:
                MemoryStoreItem.objects.filter(
                    namespace=_ns(op.namespace), key=op.key
                ).delete()
            else:
                MemoryStoreItem.objects.update_or_create(
                    namespace=_ns(op.namespace),
                    key=op.key,
                    defaults={"value": op.value},
                )
            return None
        if isinstance(op, SearchOp):
            return self._search(op)
        if isinstance(op, ListNamespacesOp):
            return self._list_namespaces(op)
        msg = f"Unsupported op {type(op).__name__}"
        raise TypeError(msg)

    def _search(self, op: SearchOp) -> list[SearchItem]:
        qs = MemoryStoreItem.objects.filter(_prefix_q(op.namespace_prefix)).order_by(
            "namespace", "key"
        )
        # ponytail: equality filters only; langgraph's $gt/$in operators unused here
        for field, expected in (op.filter or {}).items():
            qs = qs.filter(**{f"value__{field}": expected})
        rows = qs[op.offset : op.offset + op.limit]
        return [_item(row, SearchItem) for row in rows]

    def _list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        qs = MemoryStoreItem.objects.order_by("namespace")
        for cond in op.match_conditions or ():
            if cond.match_type == "prefix":
                qs = qs.filter(_prefix_q(cond.path))
            else:
                qs = qs.filter(namespace__endswith=_ns(cond.path))
        seen: dict[tuple[str, ...], None] = {}
        for ns in qs.values_list("namespace", flat=True).distinct():
            parts = tuple(ns.split(SEP))
            if op.max_depth is not None:
                parts = parts[: op.max_depth]
            seen.setdefault(parts)
        return list(seen)[op.offset : op.offset + op.limit]
