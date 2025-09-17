"""Checkpointers for Django, extending the Langgraph BaseCheckpointSaver."""

import json
from collections.abc import AsyncGenerator
from typing import (
    Any,
    Optional,
)

from django.conf import settings
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    CheckpointMetadata,
    CheckpointTuple,
    PendingWrite,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from ai_chatbots.api import WRITES_MAPPING
from ai_chatbots.models import DjangoCheckpoint, DjangoCheckpointWrite, UserChatSession

USER_MODEL = settings.AUTH_USER_MODEL


def calculate_writes(checkpoint: dict) -> dict[str, Any]:
    """
    Calculate the writes that were made between the previous checkpoint and this one.
    This is to maintain backwards compatibility with older checkpoints made before
    the langgraph upgrade and to make data platform reporting easier/more consistent.

    Args:
        checkpoint (dict): The current checkpoint data.

    Returns:
        dict[str, Any]: A dict with a writes attribute
    """
    writes = None

    updated_channels = checkpoint.get("updated_channels", [])
    if "messages" in updated_channels:
        channel_values = checkpoint.get("channel_values", {})
        native_keys = ["context", "__pregel_tasks", "llm_input_messages", "messages"]
        messages = channel_values.get("messages", [])
        if messages:
            last_message = messages[-1]
            writes_key = WRITES_MAPPING.get(
                last_message.get("kwargs", {}).get("type", None), None
            )
            if writes_key:
                writes = {
                    writes_key: {
                        "messages": [last_message],
                        **{
                            key: value
                            for key, value in channel_values.items()
                            if key not in native_keys
                        },
                    }
                }
    return writes


class CheckpointMetadataWithWrites(CheckpointMetadata):
    """Metadata associated with a checkpoint, incuding writes."""

    """The writes that were made between the previous checkpoint and this one."""
    writes: dict[str, Any]


def _parse_django_checkpoint_writes_key(
    checkpoint_write: DjangoCheckpointWrite,
) -> dict:
    """
    Return a dict of the DjangoCheckpointWrite object needed
    to load it into the checkpointer.
    """
    return {
        "thread_id": checkpoint_write.thread_id,
        "checkpoint_ns": checkpoint_write.checkpoint_ns,
        "checkpoint_id": checkpoint_write.checkpoint_id,
        "task_id": checkpoint_write.task_id,
        "idx": checkpoint_write.idx,
    }


def _load_writes(
    serde: JsonPlusSerializer, task_id_to_data: dict[tuple[str, str], dict]
) -> list[PendingWrite]:
    """
    Deserialize pending writes.
    """
    return [
        (
            task_id,
            data["channel"],
            serde.loads_typed((data["type"], data["value"])),
        )
        for (task_id, _), data in task_id_to_data.items()
    ]


def _parse_checkpoint_data(
    serde: JsonPlusSerializer,
    data: DjangoCheckpoint,
    pending_writes: Optional[list[PendingWrite]] = None,
) -> Optional[CheckpointTuple]:
    """
    Parse checkpoint data retrieved from the database.
    """
    if not data:
        return None

    thread_id = data.thread_id
    checkpoint_ns = data.checkpoint_ns
    checkpoint_id = data.checkpoint_id
    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
        }
    }
    checkpoint = serde.loads(json.dumps(data.checkpoint))
    metadata = data.metadata
    parent_checkpoint_id = data.parent_checkpoint_id
    parent_config = (
        {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": parent_checkpoint_id,
            }
        }
        if parent_checkpoint_id
        else None
    )
    return CheckpointTuple(
        config=config,
        checkpoint=checkpoint,
        metadata=metadata,
        parent_config=parent_config,
        pending_writes=pending_writes,
    )


class AsyncDjangoSaver(BaseCheckpointSaver):
    """
    Async Django ORM-based checkpoint saver for Langgraph.
    Based on the AsyncRedisSaver reference implementation:
    https://langchain-ai.github.io/langgraph/how-tos/persistence_redis/
    """

    session = None
    json_serializer = JsonPlusSerializer()

    @classmethod
    async def create_with_session(  # noqa: PLR0913
        cls,
        thread_id: str,
        message: str,
        agent: str,
        user: Optional[USER_MODEL] = None,
        dj_session_key: Optional[str] = "",
        object_id: Optional[str] = "",
    ):
        """
        Initialize the DjangoSaver and create a UserChatSession if applicable.
        """
        self = cls()
        if not (thread_id and message and agent):
            msg = "thread_id, message, and agent are required"
            raise ValueError(msg)
        chat_session, created = await UserChatSession.objects.select_related(
            "user"
        ).aget_or_create(
            thread_id=thread_id,
            defaults={
                "user": user if user and not user.is_anonymous else None,
                "dj_session_key": dj_session_key
                if (not user or user.is_anonymous)
                else "",
                "title": message[:255],
                "agent": agent,
                "object_id": object_id or "",
            },
        )
        if (
            chat_session
            and not created
            and not chat_session.user
            and user
            and not user.is_anonymous
        ):
            chat_session.user = user
            chat_session.dj_session_key = ""
            await chat_session.asave()
        self.session = chat_session
        if chat_session.user is None and user and not user.is_anonymous:
            # Thread was created when user was not logged in.
            chat_session.user = user
            await chat_session.asave()
        return self

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: DjangoCheckpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,  # noqa: ARG002
    ) -> RunnableConfig:
        """
        Save a checkpoint to the database asynchronously.

        This method saves a checkpoint to the database. The checkpoint is associated
        with the provided config and its parent config (if any).

        Args:
            config (RunnableConfig): The config to associate with the checkpoint.
            checkpoint (Checkpoint): The checkpoint to save.
            metadata (CheckpointMetadata): Additional metadata to save with checkpoint.
            new_versions (ChannelVersions): New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = checkpoint["id"]
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")

        type_, _ = self.serde.dumps_typed(checkpoint)
        serialized_checkpoint = json.loads(self.serde.dumps(checkpoint))
        serialized_metadata = json.loads(self.serde.dumps(metadata))
        if not serialized_metadata.get("writes"):
            serialized_metadata["writes"] = calculate_writes(serialized_checkpoint)

        data = {
            "checkpoint": serialized_checkpoint,
            "type": type_,
            "metadata": serialized_metadata,
            "parent_checkpoint_id": parent_checkpoint_id
            if parent_checkpoint_id
            else "",
        }

        await DjangoCheckpoint.objects.aupdate_or_create(
            session=self.session,
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            checkpoint_id=checkpoint_id,
            defaults=data,
        )

        # Delete old writes, they are no longer needed
        await DjangoCheckpointWrite.objects.filter(thread_id=thread_id).adelete()

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: list[tuple[str, Any]],
        task_id: str,
    ) -> None:
        """
        Store intermediate writes linked to a checkpoint asynchronously.

        This method saves intermediate writes associated with a checkpoint
        to the database.

        Args:
            config (RunnableConfig): Configuration of the related checkpoint.
            writes (Sequence[Tuple[str, Any]]): List of writes to store,
              each as (channel, value) pair.
            task_id (str): Identifier for the task creating the writes.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = config["configurable"]["checkpoint_id"]

        for idx, (channel, value) in enumerate(writes):
            type_, serialized_value = self.serde.dumps_typed(value)
            data = {"channel": channel, "type": type_, "value": serialized_value}
            if all(w[0] in WRITES_IDX_MAP for w in writes):
                # Overwrite any existing values
                await DjangoCheckpointWrite.objects.aupdate_or_create(
                    session=self.session,
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id=checkpoint_id,
                    task_id=task_id,
                    idx=idx,
                    defaults={
                        "channel": data["channel"],
                        "type": data["type"],
                        "blob": data["value"],
                    },
                )
            else:
                # Do not overwrite existing values
                await DjangoCheckpointWrite.objects.aget_or_create(
                    session=self.session,
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id=checkpoint_id,
                    task_id=task_id,
                    idx=idx,
                    defaults={
                        "channel": data["channel"],
                        "type": data["type"],
                        "blob": data["value"],
                    },
                )

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database asynchronously.

        This method retrieves a checkpoint tuple from the database based on the
        provided config. If the config contains a "checkpoint_id" key, the checkpoint
        with the matching thread ID and checkpoint ID is retrieved. Otherwise, the
        latest checkpoint for the given thread ID is retrieved.

        Args:
            config (RunnableConfig): The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple,
              or None if no matching checkpoint was found.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        if checkpoint_id:
            checkpoint_data = await DjangoCheckpoint.objects.filter(
                checkpoint_id=checkpoint_id
            ).afirst()
        else:
            checkpoint_data = (
                await DjangoCheckpoint.objects.filter(
                    thread_id=thread_id, checkpoint_ns=checkpoint_ns
                )
                .order_by("-checkpoint_id")
                .afirst()
            )

        if not checkpoint_data:
            return None
        # load pending writes
        checkpoint_id = checkpoint_data.checkpoint_id
        pending_writes = await self._aload_pending_writes(
            thread_id, checkpoint_ns, checkpoint_id
        )
        return _parse_checkpoint_data(
            self.serde, checkpoint_data, pending_writes=pending_writes
        )

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,  # noqa: ARG002, A002
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncGenerator[CheckpointTuple, None]:
        """List checkpoints from the database asynchronously.

        This method retrieves a list of checkpoint tuples from the database based
        on the provided config. The checkpoints are ordered by checkpoint ID in
        descending order (newest first).

        Args:
            config (Optional[RunnableConfig]): Base config for filtering checkpoints.
            filter (Optional[Dict[str, Any]]): Filtering criteria for metadata.
            before (Optional[str]): If provided, only checkpoints before the
              specified checkpoint ID are returned. Defaults to None.
            limit (Optional[int]): Maximum number of checkpoints to return.

        Yields:
            AsyncIterator[CheckpointTuple]: An asynchronous iterator
            of matching checkpoint tuples.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoints = DjangoCheckpoint.objects.filter(
            thread_id=thread_id, checkpoint_ns=checkpoint_ns
        )
        if before:
            checkpoints = checkpoints.filter(checkpoint_id__lt=before)
        if limit:
            checkpoints = checkpoints[:limit]
        checkpoint_qs = [
            checkpoint async for checkpoint in checkpoints.order_by("-checkpoint_id")
        ]
        for checkpoint in checkpoint_qs:
            if checkpoint.checkpoint and checkpoint.metadata:
                checkpoint_id = checkpoint.checkpoint_id
                pending_writes = await self._aload_pending_writes(
                    thread_id, checkpoint_ns, checkpoint_id
                )
                yield _parse_checkpoint_data(
                    self.serde, checkpoint, pending_writes=pending_writes
                )

    async def _aload_pending_writes(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> list[PendingWrite]:
        """
        Load pending writes for a checkpoint.
        """
        matching_writes = [
            mwrite
            async for mwrite in DjangoCheckpointWrite.objects.filter(
                checkpoint_id=checkpoint_id,
                checkpoint_ns=checkpoint_ns,
                thread_id=thread_id,
            )
        ]
        parsed_writes = [
            _parse_django_checkpoint_writes_key(mwrite) for mwrite in matching_writes
        ]
        return _load_writes(
            self.serde,
            {
                (parsed_key["task_id"], parsed_key["idx"]): {
                    "channel": mwrite.channel,
                    "type": mwrite.type,
                    "value": mwrite.blob,
                }
                for mwrite, parsed_key in sorted(
                    zip(matching_writes, parsed_writes), key=lambda x: x[1]["idx"]
                )
            },
        )
