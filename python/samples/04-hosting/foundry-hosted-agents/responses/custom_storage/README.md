# Custom session storage provider

This sample shows how to provide custom session storage to `ResponsesHostServer`.
The `CustomSessionStoreProvider` selects storage based on the resolved hosting
configuration:

- Local runs use the in-memory `SessionStore` and do not require Cosmos DB.
- Foundry-hosted runs use the custom `CosmosSessionStore` implementation.

The sample customizes MAF `AgentSession` persistence only. The host's response provider separately persists
caller-facing responses when the request sets `store=true`; default checkpoint and approval providers retain their
own scopes. `inner_history="host"` feeds the stored outer transcript to the model and forces downstream `store=false`.

## Azure Cosmos DB setup

Before deploying the sample, create an Azure Cosmos DB database and container. The
container must use `/user_id` as its partition key. The provider reads the user ID and hosted `agent_session_id`
from the **trusted platform context**, never from request options. Cosmos partitions records by the user and hashes
the pair `(hosted session ID, response/conversation storage key)` into each item ID. A caller cannot resume a MAF
session from another sandbox by supplying its response ID. Custom providers must preserve both boundaries.
Updates to an already loaded session use the Cosmos ETag; a concurrent turn fails instead of silently overwriting it.
Set these environment variables to
the existing resources:

- `AZURE_COSMOS_ENDPOINT`
- `COSMOS_DATABASE_NAME`
- `COSMOS_CONTAINER_NAME`

The hosted agent authenticates through its managed identity, not a connection string. Assign that identity
**Cosmos DB Built-in Data Contributor** only on the container's `/dbs/<database>/colls/<container>` scope.

See [main.py](main.py) for the complete provider and store implementations.

## Running locally

Copy `.env.example` to `.env`, set the Foundry project and model values, and leave
the Cosmos values unset. The provider creates one in-memory store for the lifetime
of the local server process.

Follow [Running the Agent Host Locally](../../README.md#running-the-agent-host-locally)
in the parent README, then send a request:

```bash
curl -X POST http://localhost:8088/responses -H "Content-Type: application/json" \
  -d '{"input": "Hi", "store": true}'
```

Local session data is lost when the process exits.

## Deploying to Foundry

Set all variables in `.env.example`, including the Cosmos settings, and follow
[Deploying the Agent to Foundry](../../README.md#deploying-the-agent-to-foundry)
in the parent README. The hosted provider initializes Cosmos DB lazily on its first
request and reuses the client and container for later requests. Each request receives
a session store scoped to both the non-empty user ID and hosted session ID supplied by Foundry.
