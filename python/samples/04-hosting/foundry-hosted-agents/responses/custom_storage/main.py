# Copyright (c) Microsoft. All rights reserved.

import hashlib
import json
import os
from contextlib import suppress
from typing import Any

from agent_framework import Agent, AgentSession, SessionStore
from agent_framework.foundry import FoundryChatClient
from agent_framework_foundry_hosting import ResponsesHostServer, StoreProvider
from azure.ai.agentserver.core import AgentConfig, FoundryAgentRequestContext
from azure.core import MatchConditions
from azure.cosmos.aio import ContainerProxy, CosmosClient
from azure.cosmos.exceptions import CosmosHttpResponseError, CosmosResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential
from dotenv import load_dotenv

"""Host an agent with a custom session storage provider.

The provider uses an in-memory store when the agent runs locally and Azure
Cosmos DB when the agent runs in Foundry. Create the database and container
before deploying the agent. The container must use /user_id as its partition key;
the item ID also scopes snapshots to the trusted Foundry hosted session.

Environment variables:
    FOUNDRY_PROJECT_ENDPOINT: Microsoft Foundry project endpoint.
    AZURE_AI_MODEL_DEPLOYMENT_NAME: Model deployment name.
    AZURE_COSMOS_ENDPOINT: Azure Cosmos DB account endpoint.
    COSMOS_DATABASE_NAME: Existing database name.
    COSMOS_CONTAINER_NAME: Existing container name partitioned by /user_id.

The hosted agent's managed identity needs Cosmos DB Built-in Data Contributor
on the container; no account key is sent to the hosted agent.
"""

load_dotenv()


class CosmosSessionStore(SessionStore):
    """Persist Agent Framework session snapshots in Azure Cosmos DB."""

    def __init__(self, *, container: ContainerProxy, user_id: str, hosted_session_id: str) -> None:
        super().__init__()
        self._container = container
        self._user_id = user_id
        self._hosted_session_id = hosted_session_id
        self._etags: dict[str, str | None] = {}

    def _item_id(self, session_id: str) -> str:
        """Keep response/conversation keys distinct between a user's hosted sessions."""
        self.validate_session_id(session_id)
        scope = json.dumps([self._hosted_session_id, session_id], separators=(",", ":"))
        return hashlib.sha256(scope.encode("utf-8")).hexdigest()

    async def get(self, session_id: str) -> AgentSession | None:
        """Load a session snapshot, or return None when it does not exist."""
        try:
            item = await self._container.read_item(item=self._item_id(session_id), partition_key=self._user_id)
        except CosmosResourceNotFoundError:
            self._etags[session_id] = None
            return None
        if item["hosted_session_id"] != self._hosted_session_id:
            raise RuntimeError("A stored MAF session belongs to another Foundry hosted session.")
        etag = item.get("_etag")
        if not isinstance(etag, str) or not etag:
            raise ValueError("Stored Cosmos session is missing its concurrency token.")
        self._etags[session_id] = etag
        return AgentSession.from_dict(item["session"])

    async def set(self, session_id: str, session: AgentSession) -> None:
        """Create or replace a session snapshot."""
        item: dict[str, Any] = {
            "id": self._item_id(session_id),
            "user_id": self._user_id,
            "hosted_session_id": self._hosted_session_id,
            "session": session.to_dict(),
        }
        try:
            if session_id in self._etags:
                etag = self._etags[session_id]
                if etag is None:
                    result = await self._container.create_item(item)
                else:
                    result = await self._container.replace_item(
                        item=item["id"],
                        body=item,
                        etag=etag,
                        match_condition=MatchConditions.IfNotModified,
                    )
            else:
                result = await self._container.upsert_item(item)
        except CosmosHttpResponseError as exc:
            if exc.status_code not in (409, 412):
                raise
            raise RuntimeError("Another request advanced this MAF session.") from exc
        new_etag = result.get("_etag")
        if not isinstance(new_etag, str) or not new_etag:
            raise ValueError("Cosmos did not return a session concurrency token.")
        self._etags[session_id] = new_etag

    async def delete(self, session_id: str) -> None:
        """Delete a session snapshot when it exists."""
        with suppress(CosmosResourceNotFoundError):
            await self._container.delete_item(item=self._item_id(session_id), partition_key=self._user_id)


class CustomSessionStoreProvider(StoreProvider[SessionStore]):
    """Provide in-memory storage locally and Cosmos-backed storage when hosted."""

    def __init__(self) -> None:
        self._local_store: SessionStore | None = None
        self._cosmos_client: CosmosClient | None = None
        self._cosmos_container: ContainerProxy | None = None
        self._cosmos_credential: AsyncDefaultAzureCredential | None = None

    def get_store(self, *, config: AgentConfig, platform_context: FoundryAgentRequestContext) -> SessionStore:
        """Return the session store for the current hosting environment."""
        if not config.is_hosted:
            if self._local_store is None:
                self._local_store = SessionStore()
            return self._local_store

        if not platform_context.user_id or not platform_context.session_id:
            raise RuntimeError("Foundry-hosted session storage requires user and hosted session IDs.")

        if self._cosmos_container is None:
            self._cosmos_credential = AsyncDefaultAzureCredential()
            self._cosmos_client = CosmosClient(
                url=os.environ["AZURE_COSMOS_ENDPOINT"],
                credential=self._cosmos_credential,
            )
            database = self._cosmos_client.get_database_client(os.environ["COSMOS_DATABASE_NAME"])
            self._cosmos_container = database.get_container_client(os.environ["COSMOS_CONTAINER_NAME"])

        return CosmosSessionStore(
            container=self._cosmos_container,
            user_id=platform_context.user_id,
            hosted_session_id=platform_context.session_id,
        )


def main() -> None:
    client = FoundryChatClient(
        project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
        model=os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"],
        credential=DefaultAzureCredential(),
    )
    agent = Agent(
        client=client,
        instructions="You are a friendly assistant. Keep your answers brief.",
    )

    server = ResponsesHostServer(
        agent=agent,
        inner_history="host",
        agent_session_store_provider=CustomSessionStoreProvider(),
    )
    server.run()


if __name__ == "__main__":
    main()
