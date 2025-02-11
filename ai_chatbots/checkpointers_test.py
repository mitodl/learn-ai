"""Tests for checkpointers"""

from unittest.mock import ANY
from uuid import uuid4

import pytest
from asgiref.sync import sync_to_async
from django.contrib.auth.models import AnonymousUser
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.base import CheckpointTuple

from ai_chatbots.checkpointers import AsyncDjangoSaver
from ai_chatbots.factories import (
    CheckpointFactory,
    CheckpointWriteFactory,
    UserChatSessionFactory,
)
from ai_chatbots.models import DjangoCheckpoint, DjangoCheckpointWrite

pytestmark = [pytest.mark.django_db]


@pytest.mark.parametrize(
    ("has_user", "anon_user"),
    [
        (True, False),
        (False, False),
        (True, True),
    ],
)
async def test_create_with_session(async_user, has_user, anon_user):
    """Test creating a checkpoint with a session."""
    thread_id = uuid4()
    message = "test_message " * 100
    test_agent = "test_agent"
    user = None if not has_user else AnonymousUser() if anon_user else async_user
    saver = await AsyncDjangoSaver.create_with_session(
        thread_id=thread_id,
        message=message,
        user=user,
        agent=test_agent,
    )
    assert saver.session.thread_id == thread_id
    assert saver.session.user == (user if has_user and not anon_user else None)
    assert saver.session.agent == test_agent
    assert saver.session.title == message[:255]


async def test_create_with_session_assign_user(async_user):
    """A previously anonymous user should be assigned to an existing session."""
    thread_id = uuid4().hex
    original_message = "hello"
    session = await sync_to_async(UserChatSessionFactory.create)(
        thread_id=thread_id, user=None, title=original_message
    )
    assert session.thread_id == thread_id
    assert (await sync_to_async(lambda: session.user)()) is None
    saver = await AsyncDjangoSaver.create_with_session(
        thread_id=thread_id,
        message="new message",
        user=async_user,
        agent="test agent",
    )
    assert saver.session.id == session.id
    assert saver.session.thread_id == thread_id
    assert saver.session.title == original_message
    assert (await sync_to_async(lambda: saver.session.user)()) == async_user


@pytest.mark.parametrize(
    ("thread_id", "message", "agent"),
    [
        (uuid4().hex, "test_message", None),
        (uuid4().hex, None, "test_agent"),
        (None, "test_message", "test_agent"),
    ],
)
async def test_create_with_session_missing_info(thread_id, message, agent):
    """A ValueError should be raised if required info is missing."""
    with pytest.raises(ValueError, match="thread_id, message, and agent are required"):
        await AsyncDjangoSaver.create_with_session(
            thread_id=thread_id,
            message=message,
            agent=agent,
        )


async def test_aput():
    """Test saving a checkpoint and deleting old checkpoint writes."""
    checkpoint_id = uuid4().hex
    thread_id = uuid4().hex
    checkpoint_ns = uuid4().hex
    metadata = {
        "source": "test",
        "step": 1,
        "writes": {"agent": {"messages": [{"content": "test"}]}},
    }
    await sync_to_async(CheckpointWriteFactory.create_batch)(3, thread_id=thread_id)
    assert await DjangoCheckpointWrite.objects.filter(thread_id=thread_id).acount() == 3
    saver = AsyncDjangoSaver()
    result = await saver.aput(
        {"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}},
        {"id": checkpoint_id, "type": "message"},
        metadata,
        [],
    )
    assert result == {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
        }
    }
    assert await DjangoCheckpointWrite.objects.filter(thread_id=thread_id).acount() == 0
    assert await DjangoCheckpoint.objects.filter(thread_id=thread_id).acount() == 1
    assert (
        await DjangoCheckpoint.objects.filter(thread_id=thread_id).afirst()
    ).metadata == metadata


async def test_aget_tuple():
    """Test getting a checkpoint tuple."""
    sample_checkpoint = await sync_to_async(CheckpointFactory.create)()
    config = {
        "configurable": {
            "thread_id": sample_checkpoint.thread_id,
            "checkpoint_ns": sample_checkpoint.checkpoint_ns,
        }
    }
    cp_tuple = await AsyncDjangoSaver().aget_tuple(config)
    assert cp_tuple == CheckpointTuple(
        config={
            "configurable": {
                "thread_id": ANY,
                "checkpoint_ns": ANY,
                "checkpoint_id": ANY,
            }
        },
        checkpoint={
            "v": 1,
            "id": ANY,
            "ts": ANY,
            "pending_sends": [],
            "versions_seen": {
                "agent": {"start:agent": 2},
                "__input__": {},
                "__start__": {"__start__": 1},
            },
            "channel_values": {
                "agent": "agent",
                "messages": [ANY, ANY, ANY],
            },
            "channel_versions": {
                "agent": 3,
                "messages": 3,
                "__start__": 2,
                "start:agent": 3,
            },
        },
        metadata=sample_checkpoint.metadata,
        parent_config={
            "configurable": {
                "thread_id": ANY,
                "checkpoint_ns": ANY,
                "checkpoint_id": ANY,
            }
        },
        pending_writes=[],
    )

    tuple_messages = cp_tuple.checkpoint["channel_values"]["messages"]
    assert tuple_messages[0].content == "Answer questions"
    assert isinstance(tuple_messages[0], SystemMessage)
    assert tuple_messages[1].content == "Tell me about ocean currents"
    assert isinstance(tuple_messages[1], HumanMessage)
    assert tuple_messages[2].content == "Ocean currents are....."
    assert isinstance(tuple_messages[2], AIMessage)


async def test_alist():
    """Alist should return a list of checkpoint tuples"""
    await DjangoCheckpoint.objects.all().adelete()
    await DjangoCheckpointWrite.objects.all().adelete()
    # The checkpoints that should be returned
    checkpoints = await sync_to_async(CheckpointFactory.create_batch)(
        3, checkpoint_ns="value1", thread_id="value1"
    )
    # Create a write for each checkpoint
    [
        await sync_to_async(CheckpointWriteFactory.create)(
            checkpoint_ns="value1", thread_id="value1", checkpoint_id=cp.checkpoint_id
        )
        for cp in checkpoints
    ]

    # other checkpoints
    await sync_to_async(CheckpointFactory.create_batch)(
        2, checkpoint_ns="value3", thread_id="value1"
    )
    # Other writes
    await sync_to_async(CheckpointWriteFactory.create_batch)(
        2, checkpoint_ns="value3", thread_id="value1"
    )

    config = {"configurable": {"thread_id": "value1", "checkpoint_ns": "value1"}}

    idx = 0
    check_ids = []
    async for cp_tuple in AsyncDjangoSaver().alist(config):
        check_ids.append(cp_tuple.config["configurable"]["checkpoint_id"])
        assert cp_tuple.config == {
            "configurable": {
                "thread_id": "value1",
                "checkpoint_ns": "value1",
                "checkpoint_id": ANY,
            }
        }
        assert cp_tuple.config["configurable"]["checkpoint_id"] in [
            cp.checkpoint_id for cp in checkpoints
        ]
        assert len(cp_tuple.pending_writes) == 1
        idx += 1
    assert idx == 3
