# Foundry Hosted Agent Samples

This directory contains Python samples for hosting [Agent Framework](https://github.com/microsoft/agent-framework)
agents and native workflows in Microsoft Foundry. AgentServer remains responsible for the Responses and Invocations
protocols; the Foundry-hosting package supplies MAF execution and session-scoped storage.

> [!IMPORTANT]
> The high-level host entry points use the breaking API implemented in this worktree's
> `agent-framework-foundry-hosting` package. Install it from this source checkout; older published betas do not
> support this API. The Invocations `break_glass` and `telegram` examples intentionally use AgentServer directly.
> All hosted manifests continue to declare the supported container protocol version `2.0.0`.

## Samples

### Responses API

| # | Sample | Description |
|---|--------|-------------|
| 1 | [Basic and history modes](responses/basic/) | Caller `store`, `inner_history="host" / "service" / "agent"`, runtime option hooks and `extra_body`, and client polling via outer `response.id`. |
| 2 | [Tools](responses/tools/) | Benign local tools and a stored, cross-turn approval request. |
| 3 | [MCP](responses/mcp/) | An agent connected to a remote MCP server (GitHub), demonstrating external MCP tool provider integration. |
| 4 | [Foundry Toolbox](responses/foundry_toolbox/) | An agent using Azure Foundry Toolbox, demonstrating toolbox provisioning and querying available tools at runtime. |
| 5 | [Native workflows and approval](responses/workflows/) | A required Responses-to-workflow parser, a request-aware factory, start-executor state, scoped checkpoints, and native approval resume. No `workflow.as_agent()`. |
| 6 | [Files](responses/files/) | Read only files in a dedicated directory inside the current Foundry hosted session. |
| 7 | [Observability](responses/observability/) | A sample demonstrating how to enable observability for the agent deployed to Foundry. |
| 8 | [Azure AI Search RAG](responses/azure_search_rag/) | An agent with Retrieval Augmented Generation (RAG) capabilities backed by Azure AI Search, grounding answers in documents indexed in a pre-provisioned search index. |
| 9 | [Foundry Memory](responses/foundry_memory/) | An agent with persistent semantic memory backed by a Microsoft Foundry Memory Store, using `FoundryMemoryProvider` to remember user facts across sessions. |
| 10 | [Monty CodeAct](responses/monty_codeact/) | An agent with a Monty-backed CodeAct context provider, exposing a single `execute_code` tool that runs Python in a [pydantic-monty](https://github.com/pydantic/monty) interpreter and invokes typed host tools (`compute`, `fetch_data`) from inside the sandbox. Uses the beta `agent-framework-monty` package. |
| 11 | [Foundry Toolbox MCP Skills](responses/foundry_toolbox_mcp_skills/) | An agent that discovers MCP-based skills attached to a Foundry Toolbox and serves them via `SkillsProvider(MCPSkillsSource(...))`, fetching `SKILL.md` bodies and supplementary resources on demand. |
| 12 | [Custom Storage](responses/custom_storage/) | A Cosmos-backed MAF session provider scoped by both platform user and hosted sandbox session. |
| 13 | [Resilient Long-Running Workflow](responses/resilient_long_running_workflow/) | A native workflow whose stored background response and exact checkpoint survive host replacement. |
| 14 | [Steerable Long-Running Agent](responses/steerable_long_running_agent/) | A non-workflow agent whose newer conversation turn can preempt an active turn. |
| 15 | [Using deployed agent](responses/using_deployed_agent.py) | A client that manages the separate hosted sandbox session lifecycle through MAF `FoundryAgent`. |

## Session Identifiers

Do not collapse these independent identities into a single `session_id`:

| Value | Owner | Purpose |
|-------|-------|---------|
| Foundry `agent_session_id` | Platform | Routes to one isolated sandbox and its persisted `$HOME`/uploaded files. Responses `conversation` binds one automatically; `previous_response_id` alone does not. Invocations reuses it only through the `agent_session_id` **query parameter**. |
| Responses `conversation.id` / `response.id` | AgentServer/Responses | Caller-facing conversation history or one stored turn. A stored response ID is also the polling handle for `background=true`; it is not a workflow checkpoint ID. |
| MAF `AgentSession.session_id` | Agent Framework/application | Identifies the inner agent's state. The host persists it under a user-, sandbox-, and lineage-scoped key when the caller chooses storage. A workflow checkpoint already includes nested AgentExecutor sessions. |
| MAF `AgentSession.service_session_id` | Inner model service | Optional downstream model continuation in `inner_history="service"` mode. It is persisted privately, not propagated as the caller's Responses ID. |
| MAF workflow checkpoint ID | Workflow engine | Captures graph, executor, pending approval/user input, and nested agent state. The host associates the **exact** checkpoint with its outer response; it does not expose it to the caller. |

**Caller-side `FoundryAgent` is different from hosting an inner agent.** The MAF client in
[`using_deployed_agent.py`](responses/using_deployed_agent.py) retains both the remote hosted session ID and its
Responses continuation in a caller-owned `AgentSession`:

```python
session.service_session_id
# Response or conversation continuation handle

session.state[FOUNDRY_HOSTED_AGENT_SESSION_ID_KEY]
# Foundry hosted-agent session ID
```

Keep the same `AgentSession` across turns so Agent Framework can forward both values correctly. When cleaning up,
read the Foundry `agent_session_id` from `session.state` and pass that value to the Foundry session deletion API.
See [Using deployed agent](responses/using_deployed_agent.py) for service-created and user-created lifecycle examples.

Inside the hosted process, use **trusted platform user and `agent_session_id`** plus the response/conversation
lineage for MAF sessions, checkpoints, and approvals. User isolation in Foundry State Store does not, by itself,
separate two sandboxes owned by the same user. Custom stores must enforce that additional boundary. The checkpoint
store is durable Foundry state *associated with* the sandbox, not necessarily a file in its `$HOME`.

## Responses execution choices

The caller's `store` controls **outer** Responses persistence. `store=false` does not durably save framework-managed
session/approval/checkpoint state; cross-turn approvals and user input require `store=true`.
`background=true` also requires `store=true`. The developer selects a separate inner history source: `host`
(AgentServer transcript, downstream `store=false`), `service` (downstream `store=true`, only current input), or
`agent` (the agent's configured MAF history provider). AgentServer handles background scheduling and returns
`response.id` promptly; it does **not** automatically send `background=true` to the inner chat client. For a
provider that supports it, [`provider_background.py`](responses/basic/provider_background.py) shows an opt-in mode.

All supported, agent-relevant CreateResponse options are translated to MAF run options. Flattened `extra_body`
values win on translated-key collisions. A developer hook may remove a caller option so the agent's own
`default_options` wins, or replace it; unsupported-option handling can ignore, warn, or error. Protocol-owned
`store`, streaming/background, identity, input, and continuation fields are not blindly forwarded as model options.

### Invocations API

| # | Sample | Description |
|---|--------|-------------|
| 1 | [Basic agent and native workflow](invocations/basic/) | A request parser maps application JSON into MAF messages/options or a typed workflow input; the host persists session state by trusted Foundry scope. |
| 2 | [Break Glass](invocations/break_glass/) | Raw AgentServer routes for applications requiring full control over the Invocations wire contract. |
| 3 | [Telegram](invocations/telegram/) | Raw AgentServer plus APIM and durable Cosmos history for a custom Telegram webhook/streaming contract. |

## Running the Agent Host Locally

The commands below require this worktree's `agent-framework-foundry-hosting` package (or a release that includes
the redesigned API). A deployed host accepted `extra_body={"max_tokens": ...}` and forwarded custom
`slogan_style` to a workflow parser; other custom fields still need gateway verification.

### Using `azd`

#### Prerequisites

1. **Azure Developer CLI (`azd`)**

    - [Install azd](https://learn.microsoft.com/en-us/azure/developer/azure-developer-cli/install-azd) and the AI agent extension: `azd ext install azure.ai.agents`
    - Authenticated: `azd auth login`

2. **Azure Subscription**

#### Create a new project

**No cloning required**. Create a new folder, point azd at the manifest on GitHub.

```bash
mkdir hosted-agent-framework-agent && cd hosted-agent-framework-agent

# Initialize from the manifest
azd ai agent init -m https://github.com/microsoft/agent-framework/blob/main/python/samples/04-hosting/foundry-hosted-agents/responses/basic/agent.manifest.yaml
```

Follow the instructions from `azd ai agent init` to complete the agent initialization. If you don't have an existing Foundry project and a model deployment, `azd ai agent init` will guide you through creating them.

#### Provision Azure Resources

> This step is only needed if you don't have an existing Foundry project and model deployment.

Run the following command to provision the necessary Azure resources:

```bash
azd provision
```

This will create the following Azure resources:

- A new resource group named `rg-[project_name]-dev`. In this guide, `[project_name]` will be `hosted-agent-framework-agent`.
- Within the resource group, among other resources, the most important ones are:
  - A new Foundry instance
  - A new Foundry project, within which a new model deployment will be created
  - An Application Insights instance
  - A container registry, which will be used to store the container images for the hosted agent

#### Set Environment Variables

```bash
export FOUNDRY_PROJECT_ENDPOINT="https://<account>.services.ai.azure.com/api/projects/<project>"
export AZURE_AI_MODEL_DEPLOYMENT_NAME="<your-model-deployment-name>"
# And any other environment variables required by the sample
```

Or in PowerShell:

```powershell
$env:FOUNDRY_PROJECT_ENDPOINT="https://<account>.services.ai.azure.com/api/projects/<project>"
$env:AZURE_AI_MODEL_DEPLOYMENT_NAME="<your-model-deployment-name>"
# And any other environment variables required by the sample
```

> Note: The environment variables set above are only for the current session. You will need to set them again if you open a new terminal session. if you want to set the environment variables permanently in the azd environment, you can use `azd env set <name> <value>`.

#### Running the Agent Host

```bash
azd ai agent run
```

Right now, the agent host should be running on `http://localhost:8088`

#### Invoking the Agent

Open another terminal, **navigate to the project directory**, and run the following command to invoke the agent:

```bash
azd ai agent invoke --local "Hello!"
```

Or you can in another terminal, without navigating to the project directory, run the following command to invoke the agent:

```bash
curl -X POST http://localhost:8088/responses -H "Content-Type: application/json" -d '{"input": "Hello!"}'
```

Or in PowerShell:

```powershell
(Invoke-WebRequest -Uri http://localhost:8088/responses -Method POST -ContentType "application/json" -Body '{"input": "Hello!"}').Content
```

### Using `python`

#### Prerequisites

1. An existing Foundry project
2. A deployed model in your Foundry project
3. Azure CLI installed and authenticated
4. Python 3.10 or later

#### Running the Agent Host with Python

Clone the repository containing the sample code:

```bash
git clone https://github.com/microsoft/agent-framework.git
cd agent-framework/python/samples/04-hosting/foundry-hosted-agents/responses
```

#### Environment setup

1. Navigate to the sample directory you want to explore. Create and activate a virtual environment using [uv](https://docs.astral.sh/uv/) (recommended):

   ```bash
   uv venv .venv
   ```

   ```bash
   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1

   # Windows (Command Prompt)
   .venv\Scripts\activate.bat

   # macOS/Linux
   source .venv/bin/activate
   ```

   > **Note:** `python -m venv .venv` also works, but can hang indefinitely on Windows with Microsoft Store Python due to a known `ensurepip` issue. Use `uv venv .venv` to avoid this.

2. Install dependencies:

   ```bash
   uv pip install -r requirements.txt
   ```

3. Create a `.env` file with your Foundry configuration following the `env.example` file in the sample.

4. Make sure you are logged in with the Azure CLI:

   ```bash
   az login
   ```

#### Running the Agent Host

```bash
python main.py
```

Right now, the agent host should be running on `http://localhost:8088`

#### Invoking the Agent

On another terminal, run the following command to invoke the agent:

```bash
curl -X POST http://localhost:8088/responses -H "Content-Type: application/json" -d '{"input": "Hello!"}'
```

Or in PowerShell:

```powershell
(Invoke-WebRequest -Uri http://localhost:8088/responses -Method POST -ContentType "application/json" -Body '{"input": "Hello!"}').Content
```

## Deploying the Agent to Foundry

Once you've tested locally, deploy to Microsoft Foundry.

### With an Existing Foundry Project

If you already have a Foundry project and the necessary Azure resources provisioned, you can skip the setup steps and proceed directly to deploying the agent.

After running `azd ai agent init -m <agent.manifest.yaml>` and following the prompts to configure your agent, you will have a project ready for deployment.

### Setting Up a New Foundry Project

Follow the steps in [Using `azd`](#using-azd) to set up the project and provision the necessary Azure resources for your Foundry deployment.

### Deploying the Agent

Once the project is setup and resources are provisioned, you can deploy the agent to Foundry by running:

```bash
azd deploy
```

> The Foundry hosting infrastructure will inject the following environment variables into your agent at runtime:
>
> - `FOUNDRY_PROJECT_ENDPOINT`: The endpoint URL for the Foundry project where the agent is deployed.
> - `AZURE_AI_MODEL_DEPLOYMENT_NAME`: The name of the model deployment in your Foundry project. This is configured during the agent initialization process with `azd ai agent init`.
> - `APPLICATIONINSIGHTS_CONNECTION_STRING`: The connection string for Application Insights to enable telemetry for your agent.

This will package your agent and deploy it to the Foundry environment, making it accessible through the Foundry project endpoint. Once it's deployed, you can also access the agent through the Foundry UI.

For the full deployment guide, see the [official deployment guide](https://learn.microsoft.com/en-us/azure/foundry/agents/how-to/deploy-hosted-agent).

Once deployed, learn more about how to manage deployed agents in the [official management guide](https://learn.microsoft.com/en-us/azure/foundry/agents/how-to/manage-hosted-agent).
