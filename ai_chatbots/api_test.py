"""Tests for api functions."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ai_chatbots import api


@pytest.fixture(autouse=True)
def mock_async_pg_save(mocker):
    """Mock the AsyncPostgresSaver classes."""
    mock_pool = mocker.patch(
        "ai_chatbots.api.AsyncConnectionPool", return_value=AsyncMock()
    )
    mock_pg_saver = mocker.patch(
        "ai_chatbots.api.AsyncPostgresSaver", return_value=AsyncMock()
    )
    return SimpleNamespace(pool=mock_pool, pg_saver=mock_pg_saver)


def mock_db_settings(sslmode):
    return {
        "NAME": "postgres",
        "USER": "fake_user",
        "PASSWORD": "fake_password",
        "HOST": "db",
        "PORT": 5433,
        "CONN_MAX_AGE": 0,
        "CONN_HEALTH_CHECKS": False,
        "DISABLE_SERVER_SIDE_CURSORS": sslmode,
        "ENGINE": "django.db.backends.postgresql",
        "AUTOCOMMIT": True,
        "TIME_ZONE": None,
    }


@pytest.mark.parametrize("persist", [False, True])
def test_chat_memory_persist(settings, persist):
    """ChatMemory should return the expected checkpointer class."""
    settings.AI_PERSISTENT_MEMORY = persist
    checkpoint = api.ChatMemory()
    if persist:
        assert isinstance(checkpoint.checkpointer, AsyncMock)
    else:
        assert isinstance(checkpoint.checkpointer, api.MemorySaver)
    # Reset the singleton instance
    del api.ChatMemory._instances[api.ChatMemory]  # noqa: SLF001


def test_chat_memory_singleton():
    """There can be only one."""
    checkpoint_1 = api.ChatMemory()
    checkpoint_2 = api.ChatMemory()
    assert checkpoint_1.checkpointer == checkpoint_2.checkpointer


@pytest.mark.parametrize("sslmode", [True, False])
def test_persistence_db(settings, sslmode):
    """
    Should return the db connection string needed by PostgresSaver.
    """
    test_db = "test_db"
    settings.DATABASES[test_db] = mock_db_settings(sslmode)
    expected_mode = "disable" if sslmode else "require"
    db = settings.DATABASES[test_db]
    conn_string = api.persistence_db(test_db)
    for key in ["USER", "PASSWORD", "HOST", "PORT", "NAME"]:
        assert str(db[key]) in conn_string
    assert f"sslmode={expected_mode}" in conn_string


@pytest.mark.parametrize("pool_size", [10, 20])
async def test_get_postgres_saver(settings, mocker, mock_async_pg_save, pool_size):
    """get_postgres_server should make expected calls and return a AsyncPostgresSaver."""

    settings.AI_PERSISTENT_POOL_SIZE = pool_size
    mock_db = "postgresql://foo:bar@localhost:5432/db"
    mocker.patch("ai_chatbots.api.persistence_db", return_value=mock_db)

    result = api.get_postgres_saver()

    mock_async_pg_save.pool.assert_called_once_with(
        conninfo=mock_db,
        max_size=pool_size,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
        },
    )
    mock_async_pg_save.pg_saver.assert_called_once_with(
        mock_async_pg_save.pool.return_value
    )
    assert result == mock_async_pg_save.pg_saver.return_value
