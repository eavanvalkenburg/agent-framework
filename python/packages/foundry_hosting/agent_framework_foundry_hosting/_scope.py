# Copyright (c) Microsoft. All rights reserved.

"""Trusted platform identity for one hosted request."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from azure.ai.agentserver.core import AgentConfig, FoundryAgentRequestContext


@dataclass(frozen=True)
class FoundryRequestScope:
    """Keep the hosted sandbox separate from Responses and MAF continuation IDs."""

    session_id: str
    user_id: str | None
    call_id: str | None
    is_hosted: bool

    @classmethod
    def from_context(
        cls,
        config: AgentConfig,
        context: FoundryAgentRequestContext,
        *,
        local_session_id: str | None = None,
    ) -> FoundryRequestScope:
        session_id = context.session_id or (local_session_id if not config.is_hosted else None)
        if not session_id:
            raise RuntimeError("A Foundry agent session ID is required to handle the request.")
        if config.is_hosted and (not context.user_id or not context.call_id):
            raise RuntimeError("Foundry hosted requests require a trusted user ID and call ID.")
        return cls(
            session_id=session_id,
            user_id=context.user_id,
            call_id=context.call_id,
            is_hosted=config.is_hosted,
        )

    @property
    def storage_key(self) -> str:
        """Bound state-store names without putting user identifiers into them."""
        return hashlib.sha256(self.session_id.encode("utf-8")).hexdigest()
