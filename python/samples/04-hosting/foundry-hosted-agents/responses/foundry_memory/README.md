# What this sample demonstrates

An [Agent Framework](https://github.com/microsoft/agent-framework) agent with persistent semantic memory backed by a **Microsoft Foundry Memory Store**, hosted using the **Responses protocol**. The agent remembers facts the user has shared (e.g., dietary preferences, name) across sessions by retrieving and updating memories around every model invocation via `FoundryMemoryProvider`.

## How It Works

### Model Integration

The agent uses `FoundryChatClient` from the Agent Framework to create a Responses client from the project endpoint and model deployment. `allow_preview=True` is passed so the same `AIProjectClient` can also call the preview `beta.memory_stores` API.

### Memory via Foundry Memory Store

`FoundryMemoryProvider` is wired into the agent as a context provider. Around each model invocation it:

1. **Retrieves user-profile memories** for the configured `scope` (e.g., user id) on the first turn of a session.
2. **Searches for contextual memories** matching the current user message and injects them into the model context.
3. **Updates the store** with new facts inferred from the conversation.

This external memory store is independent of the caller's Responses `store` flag. `store=false` prevents the host
from storing a response or updated MAF session; it cannot prevent an explicitly configured external context
provider from updating its own memory. Use this sample only where that persistence is intended.
The host builds an agent for each request and hashes the **trusted platform user ID** into the memory scope.
`FoundryMemoryProvider` forwards its `scope` literally, so `{{$userId}}` is not a substitute for this
request-time binding. Memories intentionally persist across hosted sessions for the same user, not across users.
Local runs use a single-user `local-user` scope and do not prove hosted identity isolation.

Crucially, the provider is constructed with `project_client=client.project_client` — i.e. it reuses the `AIProjectClient` that `FoundryChatClient` already created, instead of allocating a second one. This keeps a single authentication context and connection pool for both chat and memory operations.

See [main.py](main.py) for the full implementation.

### Agent Hosting

The agent is hosted using the [Agent Framework](https://github.com/microsoft/agent-framework) with the `ResponsesHostServer`, which provisions a REST API endpoint compatible with the OpenAI Responses protocol.

## Prerequisites

- A Microsoft Foundry project with:
  - A deployed chat model (e.g., `gpt-4.1-mini`)
  - A deployed embedding model (e.g., `text-embedding-3-small`) — used by the memory store itself, not by the agent at runtime
- Azure CLI logged in (`az login`)

### Required RBAC

Your identity and the hosted agent's Managed Identity need **Foundry User** on the Foundry project scope to
read and write memories. Without it, model responses can still complete while the memory provider logs
`agents/read` or `agents/write` permission failures and retains nothing.

## Provisioning the memory store (one time)

[`provision_memory_store.py`](provision_memory_store.py) creates a Foundry Memory Store with the user-profile capability enabled (and chat-summary disabled) using `AIProjectClient.beta.memory_stores.create`. It is safe to re-run: if a store with the same name already exists, the script leaves it alone.

From this directory, with the venv activated and `az login` done:

```bash
export FOUNDRY_PROJECT_ENDPOINT="https://<account>.services.ai.azure.com/api/projects/<project>"
export AZURE_AI_MODEL_DEPLOYMENT_NAME="gpt-4.1-mini"
export AZURE_AI_EMBEDDING_MODEL_DEPLOYMENT_NAME="text-embedding-3-small"
export MEMORY_STORE_NAME="agent_framework_memory"
python provision_memory_store.py
```

Or in PowerShell:

```powershell
$env:FOUNDRY_PROJECT_ENDPOINT="https://<account>.services.ai.azure.com/api/projects/<project>"
$env:AZURE_AI_MODEL_DEPLOYMENT_NAME="gpt-4.1-mini"
$env:AZURE_AI_EMBEDDING_MODEL_DEPLOYMENT_NAME="text-embedding-3-small"
$env:MEMORY_STORE_NAME="agent_framework_memory"
python provision_memory_store.py
```

Expected output (first run):

```text
Creating memory store 'agent_framework_memory'...
Created memory store 'agent_framework_memory' (id=memstore_...).
```

> To delete the store manually, call `project.beta.memory_stores.delete("<name>")` on an `AIProjectClient` constructed with `allow_preview=True`.

## Running the Agent Host

Follow the instructions in the [Running the Agent Host Locally](../../README.md#running-the-agent-host-locally) section of the README in the parent directory to run the agent host.

In addition to the standard environment variables, this sample requires:

```bash
export MEMORY_STORE_NAME="agent_framework_memory"
```

Or in PowerShell:

```powershell
$env:MEMORY_STORE_NAME="agent_framework_memory"
```

You can also place these in a `.env` file next to `main.py` — see [`.env.example`](.env.example).

## Interacting with the agent

> Depending on how you run the agent host, you can invoke the agent using `curl` (`Invoke-WebRequest` in PowerShell) or `azd`. Please refer to the [parent README](../../README.md) for more details.

Send a POST request to the server with a JSON body containing an `"input"` field to interact with the agent. The first request seeds a memory; subsequent requests (especially in new sessions) should be able to recall it because memories are persisted across Foundry Hosted Agents sessions.

> The request-scoped factory obtains the platform user ID and hashes it before constructing `FoundryMemoryProvider`;
> the same user's sessions share memories, while different users cannot read each other's scope.

```bash
# 1. Tell the agent something to remember.
curl -X POST http://localhost:8088/responses -H "Content-Type: application/json" \
  -d '{"input": "I prefer dark roast coffee and enjoy reading novels."}'

# Wait a few seconds for the memory to be stored, then start a fresh conversation:
curl -X POST http://localhost:8088/responses -H "Content-Type: application/json" \
  -d '{"input": "Can you suggest a coffee for my next reading session?"}'

curl -X POST http://localhost:8088/responses -H "Content-Type: application/json" \
  -d '{"input": "What do you remember about my preferences?"}'
```

## Deploying the Agent to Foundry

To host the agent on Foundry, follow the instructions in the [Deploying the Agent to Foundry](../../README.md#deploying-the-agent-to-foundry) section of the README in the parent directory.

When deploying, make sure `MEMORY_STORE_NAME` is set in your `azd` environment so it gets injected into the hosted container per [`agent.manifest.yaml`](agent.manifest.yaml):

```bash
azd env set MEMORY_STORE_NAME "agent_framework_memory"
```

If these are not set, running `azd ai agent init -m <agent.manifest.yaml>` will prompt you to enter them interactively.

The deployed agent's Managed Identity needs **Foundry User** on the Foundry project to read and write memories.
Run `provision_memory_store.py` against that same project before deploying.
