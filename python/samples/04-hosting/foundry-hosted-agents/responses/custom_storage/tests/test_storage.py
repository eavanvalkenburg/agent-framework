# Copyright (c) Microsoft. All rights reserved.

"""The Cosmos sample preserves sandbox identity and conditional session updates."""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from agent_framework import AgentSession
from azure.core import MatchConditions
from azure.cosmos.aio import ContainerProxy
from azure.cosmos.exceptions import CosmosHttpResponseError, CosmosResourceNotFoundError

_sample = runpy.run_path(str(Path(__file__).resolve().parents[1] / "main.py"), run_name="sample_cosmos_store")
CosmosSessionStore = cast(Any, _sample["CosmosSessionStore"])


def _container() -> MagicMock:
    container = MagicMock(spec=ContainerProxy)
    container.read_item = AsyncMock()
    container.create_item = AsyncMock()
    container.replace_item = AsyncMock()
    container.upsert_item = AsyncMock()
    return container


async def test_existing_session_updates_only_from_its_etag() -> None:
    container = _container()
    session = AgentSession(session_id="maf-session")
    container.read_item.return_value = {
        "hosted_session_id": "sandbox",
        "user_id": "user",
        "session": session.to_dict(),
        "_etag": "version-1",
    }
    container.replace_item.side_effect = [{"_etag": "version-2"}, {"_etag": "version-3"}]
    store = CosmosSessionStore(container=container, user_id="user", hosted_session_id="sandbox")

    assert (await store.get("conversation")).session_id == session.session_id
    await store.set("conversation", session)
    await store.set("conversation", session)

    assert container.read_item.await_args.kwargs["partition_key"] == "user"
    first = container.replace_item.await_args_list[0].kwargs
    assert first["etag"] == "version-1"
    assert first["match_condition"] is MatchConditions.IfNotModified
    assert first["body"]["hosted_session_id"] == "sandbox"
    assert first["body"]["user_id"] == "user"
    assert container.replace_item.await_args_list[1].kwargs["etag"] == "version-2"
    container.upsert_item.assert_not_awaited()


async def test_missing_session_uses_create_only() -> None:
    container = _container()
    container.read_item.side_effect = CosmosResourceNotFoundError(message="not found")
    container.create_item.return_value = {"_etag": "version-1"}
    store = CosmosSessionStore(container=container, user_id="user", hosted_session_id="sandbox")

    assert await store.get("conversation") is None
    await store.set("conversation", AgentSession(session_id="maf-session"))

    container.create_item.assert_awaited_once()
    container.upsert_item.assert_not_awaited()


async def test_cross_sandbox_record_cannot_be_loaded() -> None:
    container = _container()
    container.read_item.return_value = {
        "hosted_session_id": "other-sandbox",
        "session": AgentSession(session_id="maf-session").to_dict(),
        "_etag": "version-1",
    }
    store = CosmosSessionStore(container=container, user_id="user", hosted_session_id="sandbox")

    with pytest.raises(RuntimeError, match="another Foundry hosted session"):
        await store.get("conversation")
    container.replace_item.assert_not_awaited()


async def test_concurrent_update_fails_without_overwriting() -> None:
    container = _container()
    container.read_item.return_value = {
        "hosted_session_id": "sandbox",
        "session": AgentSession(session_id="maf-session").to_dict(),
        "_etag": "stale",
    }
    container.replace_item.side_effect = CosmosHttpResponseError(status_code=412, message="etag mismatch")
    store = CosmosSessionStore(container=container, user_id="user", hosted_session_id="sandbox")

    assert await store.get("conversation") is not None
    with pytest.raises(RuntimeError, match="Another request advanced"):
        await store.set("conversation", AgentSession(session_id="maf-session"))
    container.upsert_item.assert_not_awaited()
