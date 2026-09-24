# Copyright (c) Microsoft. All rights reserved.

"""Workflow checkpoint pointers are bound to response lineage and sandbox."""

from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from azure.ai.agentserver.core.storage import FoundryStorageConflictError

from agent_framework_foundry_hosting._scope import FoundryRequestScope
from agent_framework_foundry_hosting._workflow_state import FoundryWorkflowBindingStore, WorkflowBinding


def _binding(*, session_id: str = "sandbox-1", response_id: str = "response-1") -> WorkflowBinding:
    return WorkflowBinding(
        response_id=response_id,
        checkpoint_id="checkpoint-1",
        lineage_id="conversation-1",
        workflow_name="review",
        graph_hash="graph-1",
        session_id=session_id,
        conversation_id="conversation-1",
    )


def _store() -> MagicMock:
    store = MagicMock()
    store.__aenter__ = AsyncMock(return_value=store)
    store.__aexit__ = AsyncMock(return_value=None)
    store.create_item = AsyncMock()
    store.set_item = AsyncMock()
    store.get_item = AsyncMock()
    return store


async def test_immutable_response_binding_and_etag_conversation_head() -> None:
    scope = FoundryRequestScope(session_id="sandbox-1", user_id="user-1", call_id="call-1", is_hosted=True)
    record = _binding()
    store = _store()
    store.get_item = AsyncMock(
        side_effect=[
            SimpleNamespace(value=asdict(record)),
            SimpleNamespace(value=asdict(record), etag="etag-1"),
        ]
    )
    with patch(
        "agent_framework_foundry_hosting._workflow_state.FoundryStateStore.get_or_create",
        new=AsyncMock(return_value=store),
    ) as get_or_create:
        bindings = FoundryWorkflowBindingStore(scope)
        await bindings.save_response(record)
        assert await bindings.get_response(record.response_id) == record
        head, etag = await bindings.get_conversation_head("conversation-1")
        assert head == record
        assert etag == "etag-1"
        await bindings.advance_conversation("conversation-1", record, expected_etag=etag)

    assert get_or_create.await_args_list[0].args == (f"workflow_bindings/v2/{scope.storage_key}",)
    assert all(call.kwargs == {"user_isolation": True} for call in get_or_create.await_args_list)
    assert store.create_item.await_args.kwargs["call_id"] == "call-1"
    assert store.set_item.await_args.kwargs == {
        "if_match": "etag-1",
        "call_id": "call-1",
    }


async def test_cross_sandbox_response_pointer_is_refused_even_if_backend_returns_it() -> None:
    scope = FoundryRequestScope(session_id="sandbox-2", user_id="user-1", call_id="call-2", is_hosted=True)
    store = _store()
    store.get_item = AsyncMock(return_value=SimpleNamespace(value=asdict(_binding())))
    with (
        patch(
            "agent_framework_foundry_hosting._workflow_state.FoundryStateStore.get_or_create",
            new=AsyncMock(return_value=store),
        ),
        pytest.raises(PermissionError, match="another Foundry session"),
    ):
        await FoundryWorkflowBindingStore(scope).get_response("response-1")


async def test_concurrent_head_advance_does_not_silently_overwrite() -> None:
    scope = FoundryRequestScope(session_id="sandbox-1", user_id="user-1", call_id="call-1", is_hosted=True)
    store = _store()
    store.set_item = AsyncMock(side_effect=FoundryStorageConflictError("etag conflict"))
    with (
        patch(
            "agent_framework_foundry_hosting._workflow_state.FoundryStateStore.get_or_create",
            new=AsyncMock(return_value=store),
        ),
        pytest.raises(RuntimeError, match="Another request advanced"),
    ):
        await FoundryWorkflowBindingStore(scope).advance_conversation(
            "conversation-1", _binding(), expected_etag="stale-etag"
        )


async def test_recovery_rejects_rebinding_a_response_to_different_checkpoint() -> None:
    scope = FoundryRequestScope(session_id="sandbox-1", user_id="user-1", call_id="call-1", is_hosted=True)
    stored = _binding()
    store = _store()
    store.create_item = AsyncMock(side_effect=FoundryStorageConflictError("already exists"))
    store.get_item = AsyncMock(return_value=SimpleNamespace(value=asdict(stored)))
    with patch(
        "agent_framework_foundry_hosting._workflow_state.FoundryStateStore.get_or_create",
        new=AsyncMock(return_value=store),
    ):
        other = _binding(response_id="response-1")
        other = WorkflowBinding(**{**asdict(other), "checkpoint_id": "checkpoint-2"})
        with pytest.raises(RuntimeError, match="different workflow checkpoint"):
            await FoundryWorkflowBindingStore(scope).save_response(other)


async def test_recovery_advances_a_response_pointer_only_from_its_persisted_snapshot() -> None:
    scope = FoundryRequestScope(session_id="sandbox-1", user_id="user-1", call_id="call-1", is_hosted=True)
    first = _binding()
    later = WorkflowBinding(**{**asdict(first), "checkpoint_id": "checkpoint-2"})
    store = _store()
    store.create_item = AsyncMock(side_effect=FoundryStorageConflictError("already exists"))
    store.get_item = AsyncMock(return_value=SimpleNamespace(value=asdict(first), etag="etag-1"))
    with patch(
        "agent_framework_foundry_hosting._workflow_state.FoundryStateStore.get_or_create",
        new=AsyncMock(return_value=store),
    ):
        await FoundryWorkflowBindingStore(scope).save_response(later, recovery_from="checkpoint-1")

    assert store.set_item.await_args.kwargs == {"if_match": "etag-1", "call_id": "call-1"}
