# Copyright (c) Microsoft. All rights reserved.

"""Response-to-checkpoint bindings for a single Foundry hosted sandbox."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, cast

from azure.ai.agentserver.core.storage import FoundryStateStore, FoundryStorageConflictError

from ._scope import FoundryRequestScope


def _key(kind: str, value: str) -> str:
    if not value:
        raise ValueError(f"{kind} must be a non-empty identifier.")
    return f"{kind}-{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


@dataclass(frozen=True)
class WorkflowBinding:
    """The exact checkpoint that produced one caller-visible response."""

    response_id: str
    checkpoint_id: str
    lineage_id: str
    workflow_name: str
    graph_hash: str
    session_id: str
    conversation_id: str | None

    @classmethod
    def from_value(cls, value: Any, scope: FoundryRequestScope) -> WorkflowBinding:
        if not isinstance(value, dict):
            raise ValueError("Invalid persisted workflow binding.")
        payload = cast(Mapping[str, object], value)
        expected = {
            "response_id",
            "checkpoint_id",
            "lineage_id",
            "workflow_name",
            "graph_hash",
            "session_id",
            "conversation_id",
        }
        if set(payload) != expected:
            raise ValueError("The persisted workflow binding has unexpected fields.")

        def required(name: str) -> str:
            identifier = payload[name]
            if not isinstance(identifier, str) or not identifier:
                raise ValueError(f"The persisted workflow binding has an invalid {name}.")
            return identifier

        response_id = required("response_id")
        checkpoint_id = required("checkpoint_id")
        lineage_id = required("lineage_id")
        workflow_name = required("workflow_name")
        graph_hash = required("graph_hash")
        session_id = required("session_id")
        conversation_id = payload["conversation_id"]
        if conversation_id is not None and not isinstance(conversation_id, str):
            raise ValueError("The persisted workflow binding has an invalid conversation_id.")
        if session_id != scope.session_id:
            raise PermissionError("The workflow checkpoint belongs to another Foundry session.")
        return cls(
            response_id=response_id,
            checkpoint_id=checkpoint_id,
            lineage_id=lineage_id,
            workflow_name=workflow_name,
            graph_hash=graph_hash,
            session_id=session_id,
            conversation_id=conversation_id,
        )


@dataclass(frozen=True)
class WorkflowApproval:
    """Bind a caller-facing approval to the paused request that authorized it."""

    wire_id: str
    request_id: str
    response_id: str
    lineage_id: str
    checkpoint_id: str
    session_id: str
    consumed: bool = False

    @classmethod
    def from_value(cls, value: Any, scope: FoundryRequestScope) -> WorkflowApproval:
        if not isinstance(value, dict):
            raise ValueError("Invalid persisted workflow approval.")
        payload = cast(Mapping[str, object], value)
        fields = ("wire_id", "request_id", "response_id", "lineage_id", "checkpoint_id", "session_id")
        if set(payload) != {*fields, "consumed"}:
            raise ValueError("Invalid persisted workflow approval.")
        for name in fields:
            if not isinstance(payload[name], str) or not payload[name]:
                raise ValueError(f"Invalid persisted workflow approval {name}.")
        if not isinstance(payload["consumed"], bool):
            raise ValueError("Invalid persisted workflow approval decision state.")
        record = cls(
            wire_id=cast(str, payload["wire_id"]),
            request_id=cast(str, payload["request_id"]),
            response_id=cast(str, payload["response_id"]),
            lineage_id=cast(str, payload["lineage_id"]),
            checkpoint_id=cast(str, payload["checkpoint_id"]),
            session_id=cast(str, payload["session_id"]),
            consumed=payload["consumed"],
        )
        if record.session_id != scope.session_id:
            raise PermissionError("This approval belongs to another Foundry session.")
        return record


class FoundryWorkflowBindingStore:
    """Persist immutable response pointers and one CAS-protected conversation head."""

    ROOT_SCOPE = "workflow_bindings"

    def __init__(self, scope: FoundryRequestScope) -> None:
        self.scope = scope

    async def _get_store(self) -> FoundryStateStore:
        return await FoundryStateStore.get_or_create(
            f"{self.ROOT_SCOPE}/v2/{self.scope.storage_key}",
            user_isolation=True,
        )

    async def get_response(self, response_id: str) -> WorkflowBinding | None:
        store = await self._get_store()
        async with store:
            item = await store.get_item(_key("response", response_id), call_id=self.scope.call_id)
        if item is None:
            return None
        binding = WorkflowBinding.from_value(item.value, self.scope)
        if binding.response_id != response_id:
            raise ValueError("Persisted workflow response binding does not match the requested ID.")
        return binding

    async def save_response(self, binding: WorkflowBinding, *, recovery_from: str | None = None) -> None:
        if binding.session_id != self.scope.session_id:
            raise PermissionError("Cannot save a checkpoint for another Foundry session.")
        store = await self._get_store()
        async with store:
            try:
                await store.create_item(
                    _key("response", binding.response_id),
                    asdict(binding),
                    call_id=self.scope.call_id,
                )
            except FoundryStorageConflictError as exc:
                existing = await store.get_item(_key("response", binding.response_id), call_id=self.scope.call_id)
                if existing is None:
                    raise RuntimeError("A workflow response binding disappeared during recovery.") from exc
                previous = WorkflowBinding.from_value(existing.value, self.scope)
                if previous == binding:
                    return
                compatible = previous.checkpoint_id == recovery_from and {
                    key: value for key, value in asdict(previous).items() if key != "checkpoint_id"
                } == {key: value for key, value in asdict(binding).items() if key != "checkpoint_id"}
                if not compatible:
                    raise RuntimeError("A different workflow checkpoint is already bound to this response.") from exc
                try:
                    await store.set_item(
                        _key("response", binding.response_id),
                        asdict(binding),
                        if_match=existing.etag,
                        call_id=self.scope.call_id,
                    )
                except FoundryStorageConflictError as conflict:
                    raise RuntimeError("Another recovery attempt advanced this response checkpoint.") from conflict

    async def get_conversation_head(self, conversation_id: str) -> tuple[WorkflowBinding | None, str | None]:
        binding, etag = await self._get_head("conversation", conversation_id)
        if binding is not None and binding.conversation_id != conversation_id:
            raise ValueError("Persisted workflow conversation binding does not match the requested ID.")
        return binding, etag

    async def get_lineage_head(self, lineage_id: str) -> tuple[WorkflowBinding | None, str | None]:
        binding, etag = await self._get_head("lineage", lineage_id)
        if binding is not None and binding.lineage_id != lineage_id:
            raise ValueError("Persisted workflow lineage binding does not match the requested ID.")
        return binding, etag

    async def _get_head(self, kind: str, key: str) -> tuple[WorkflowBinding | None, str | None]:
        store = await self._get_store()
        async with store:
            item = await store.get_item(_key(kind, key), call_id=self.scope.call_id)
        if item is None:
            return None, None
        binding = WorkflowBinding.from_value(item.value, self.scope)
        return binding, item.etag

    async def advance_conversation(
        self,
        conversation_id: str,
        binding: WorkflowBinding,
        *,
        expected_etag: str | None,
    ) -> None:
        if binding.conversation_id != conversation_id or binding.session_id != self.scope.session_id:
            raise PermissionError("Cannot advance a workflow conversation outside its trusted scope.")
        await self._advance_head("conversation", conversation_id, binding, expected_etag)

    async def advance_lineage(
        self,
        lineage_id: str,
        binding: WorkflowBinding,
        *,
        expected_etag: str | None,
    ) -> None:
        if binding.lineage_id != lineage_id or binding.session_id != self.scope.session_id:
            raise PermissionError("Cannot advance a workflow lineage outside its trusted scope.")
        await self._advance_head("lineage", lineage_id, binding, expected_etag)

    async def _advance_head(self, kind: str, key: str, binding: WorkflowBinding, expected_etag: str | None) -> None:
        store = await self._get_store()
        async with store:
            try:
                if expected_etag is None:
                    await store.create_item(
                        _key(kind, key),
                        asdict(binding),
                        call_id=self.scope.call_id,
                    )
                else:
                    await store.set_item(
                        _key(kind, key),
                        asdict(binding),
                        if_match=expected_etag,
                        call_id=self.scope.call_id,
                    )
            except FoundryStorageConflictError as exc:
                raise RuntimeError("Another request advanced this workflow conversation.") from exc

    async def save_approval(self, approval: WorkflowApproval) -> None:
        if approval.session_id != self.scope.session_id:
            raise PermissionError("Cannot save an approval for another Foundry session.")
        store = await self._get_store()
        async with store:
            try:
                await store.create_item(
                    _key("approval", approval.wire_id), asdict(approval), call_id=self.scope.call_id
                )
            except FoundryStorageConflictError as exc:
                raise RuntimeError("A workflow approval with this ID already exists.") from exc

    async def get_approval(
        self,
        wire_id: str,
        *,
        lineage_id: str,
    ) -> WorkflowApproval:
        """Check the pending authority without consuming it during request validation."""
        store = await self._get_store()
        async with store:
            item = await store.get_item(_key("approval", wire_id), call_id=self.scope.call_id)
        if item is None:
            raise KeyError("The workflow approval is not available in this Foundry session.")
        approval = WorkflowApproval.from_value(item.value, self.scope)
        if approval.wire_id != wire_id or approval.lineage_id != lineage_id or approval.consumed:
            raise PermissionError("The workflow approval does not belong to this pending checkpoint.")
        return approval

    async def consume_approval(
        self,
        wire_id: str,
        *,
        lineage_id: str,
        checkpoint_id: str,
    ) -> WorkflowApproval:
        store = await self._get_store()
        async with store:
            item = await store.get_item(_key("approval", wire_id), call_id=self.scope.call_id)
            if item is None:
                raise KeyError("The workflow approval is not available in this Foundry session.")
            approval = WorkflowApproval.from_value(item.value, self.scope)
            if (
                approval.wire_id != wire_id
                or approval.lineage_id != lineage_id
                or approval.checkpoint_id != checkpoint_id
                or approval.consumed
            ):
                raise PermissionError("The workflow approval does not belong to this pending checkpoint.")
            try:
                await store.set_item(
                    _key("approval", wire_id),
                    asdict(
                        WorkflowApproval(
                            wire_id=approval.wire_id,
                            request_id=approval.request_id,
                            response_id=approval.response_id,
                            lineage_id=approval.lineage_id,
                            checkpoint_id=approval.checkpoint_id,
                            session_id=approval.session_id,
                            consumed=True,
                        )
                    ),
                    if_match=item.etag,
                    call_id=self.scope.call_id,
                )
            except FoundryStorageConflictError as exc:
                raise RuntimeError("The workflow approval was already handled by another request.") from exc
        return approval
