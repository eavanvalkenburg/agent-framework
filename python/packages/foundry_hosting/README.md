# Foundry Hosting

`agent-framework-foundry-hosting` runs Agent Framework `Agent` and native `Workflow` targets behind Microsoft
Foundry Hosted Agents. It builds on `azure-ai-agentserver-responses` and `azure-ai-agentserver-invocations` for
protocol parsing, response IDs, streaming, background work, health probes, and lifecycle. The hosting package
connects those protocols to MAF execution and scoped state; it does not require `workflow.as_agent()`.

See the [hosted-agent samples](../../samples/04-hosting/foundry-hosted-agents/) for complete entry points and
client requests.

## Agents: caller storage versus inner history

```python
agent = Agent(client=client, instructions="Be concise.")
ResponsesHostServer(agent=agent, inner_history="host").run()
```

`agent` accepts an instance or a zero-argument synchronous/asynchronous factory. An instance belongs to that
server. Use a factory when it has mutable state or request-owned tools outside `AgentSession`.

The caller's Responses `store` flag controls the **outer** response. `response_store=` optionally configures the
AgentServer response backend; it does not override the caller's flag. The developer independently selects an
inner history source:

| Caller / mode | Model input and persistence |
|---|---|
| `store=False` | Return the result without a durable outer response, updated MAF session, checkpoint, or approval. The inner model must not store the turn. The returned protocol `response.id` is transient and cannot be retrieved. Application-owned tools and external memory stores can still have their own side effects. |
| `store=True`, `inner_history="host"` (default) | AgentServer supplies prior Responses history. The inner storing chat client receives `store=False`; no second model transcript is loaded. |
| `store=True`, `inner_history="service"` | The inner client receives only this request's input with `store=True`. Its `AgentSession.service_session_id` is persisted privately; callers receive only outer Responses IDs. |
| `store=True`, `inner_history="agent"` | The inner client receives only new input with `store=False`. The agent's configured `HistoryProvider` supplies history from its `AgentSession` and chosen backing store. |

Host mode rejects a load-enabled agent history provider and fixed downstream continuation defaults; otherwise the
model would see duplicate history. A non-storing client with a developer-owned `default_options["store"]` also
fails configuration instead of having the host mutate that default. `store=False` rejects fixed downstream
continuation defaults and custom agents whose storage behavior the host cannot control.

An agent instance or factory can still be passed through `agent=`. `history_source="agent_server"|"agent"` and
the former constructor `store=` are accepted for existing code; prefer `inner_history=` and `response_store=`
in new code.

### Runtime options

The host translates the native CreateResponse generation fields to MAF run options (for example,
`max_output_tokens` to `max_tokens` and `parallel_tool_calls` to `allow_multiple_tool_calls`), then overlays
other flattened request fields supplied through the OpenAI client's `extra_body`. Extra fields win if they
collide with a translated MAF key. When `extra_body` duplicates an identical *wire* key, the client has already
selected one value before hosting sees it.

An optional `prepare_options(request, options)` hook can remove or replace **caller runtime** options. Removing
one lets the developer's `Agent.default_options` win through ordinary MAF precedence; the host does not edit
developer defaults. `unsupported_options="ignore"|"warn"|"error"` controls what happens when a custom agent
does not accept runtime options. Provider-level validation failures are not swallowed. Identity, input, outer
`store`, background/stream selection, and Responses continuation fields remain protocol controls rather than
model options. The azd `session_id`, platform user/call IDs, and private downstream `conversation_id`,
`service_session_id`, and `continuation_token` are also never forwarded as caller model options. The
`HostedResponseRequest` view exposes its trusted `scope`, original request, and normalized `options`.

Locally AgentServer preserves unknown JSON fields. A deployed Foundry agent accepted
`extra_body={"max_tokens": ...}` and forwarded a custom `slogan_style` field to the workflow parser.
Check any other custom fields against your target gateway before relying on them.

### Background responses

The caller sends `store=True, background=True`; AgentServer returns a queued/in-progress **outer**
`response.id` immediately. Poll `GET /responses/{id}` or reconnect to its stored SSE stream. A later
conversation turn is a separate POST, not a background poll. `background=True` does not automatically reach
the inner chat client, so ordinary agents work even when their client has no provider-background support.
`background=True` with `store=False` is rejected by AgentServer.

Without a recovery strategy, an interrupted ordinary agent run is not safely replayed; its outer response
may remain in progress after an ungraceful process crash. Optional `inner_background="provider"` with
`inner_history="service"` and
`ResponsesServerOptions(resilient_background=True)` uses a storing client that returns an inner MAF continuation
token. Hosting keeps that token private under the outer response ID and polls it until completion. On recovery it
polls a stored token; if the process died before a token was recorded, it fails rather than starting another job.
Provider background is not enabled for workflows. In a deployed host, the agent's managed identity needs
**Foundry User** at the project scope to retrieve its privately stored inner response; model creation alone
may succeed without that read permission.

For a non-workflow agent, `steerable_conversations=True` enables AgentServer's multi-turn task manager so new
input on the same active chain can queue and preempt the current turn. It does not make an interrupted model
run safe to replay after a process crash.

## Native workflows

`workflow=` takes a request-aware factory returning an already **built** `Workflow`. The factory runs for each
hosted turn, including checkpoint restoration. Give each definition stable workflow/executor IDs and build fresh
mutable executors. `parse_response=` is required: the start executor has an application-specific input type,
which the host cannot infer from the Responses wire format.

```python
async def parse_response(request):
    if replies := await request.get_workflow_responses():
        return WorkflowTurn(responses=replies)
    text = await request.get_input_text()
    if not text:
        raise ValueError("Input is required.")
    return WorkflowTurn(input=text)


ResponsesHostServer(workflow=build_workflow, parse_response=parse_response).run()
```

The parser sees only the **current turn's** input; earlier workflow state comes from its checkpoint. It may
convert input items to MAF messages using `request.get_input_messages()`. For application-owned shared state,
include the data in the typed start input and call `WorkflowContext.set_state()` in the start executor. An
executor response handler can update it after a human reply. Passing initial shared state or per-executor
`Agent.run(options=...)` directly through `Workflow.run` is a separate core proposal
([microsoft/agent-framework#8711](https://github.com/microsoft/agent-framework/issues/8711)); this host does
not mutate private workflow runner state. Existing `client_kwargs` and `function_invocation_kwargs` are
forwarded when supplied through `WorkflowTurn`; core includes them in checkpoints, so never put short-lived
credentials or call IDs there.

For `store=True`, Foundry State Store holds the workflow checkpoints and a binding from each outer
`response.id` to the **exact** checkpoint corresponding to that response. A `conversation` has a
conditionally updated head. Checkpoint restoration validates the workflow graph and Foundry sandbox before
delivering the new turn. Function approval and `request_info` replies are matched to the pending IDs in that
checkpoint and passed to `Workflow.run(responses=...)`. A stale or repeated decision is refused; an
external tool's side effects are not guaranteed exactly-once across crashes, so use idempotency for
effectful operations. Existing Responses approval items (`mcp_approval_request` /
`mcp_approval_response`) carry the caller-facing decision. No `WorkflowAgent` wrapper is constructed.

Cross-turn approval or user input needs `store=True`. Under `store=False`, a one-shot workflow runs without a
durable checkpoint and fails clearly if it requests external input. A Responses conversation automatically
reuses its hosted sandbox; with `previous_response_id` alone, also supply the original `agent_session_id` to
restore its checkpoint. Workflow forks from an earlier response are rejected until checkpoint/file-fork
semantics exist.

Stored background workflows can opt into `ResponsesServerOptions(resilient_background=True)`. The host pairs
persisted output snapshots with checkpoint IDs; on process recovery it resumes the exact paired checkpoint
without silently discarding newly produced output or selecting the latest timestamp across runs.
The older `WorkflowAgent` path remains available for compatibility, but does not have native workflows'
response-to-checkpoint bindings; its background recovery fails closed if no paired snapshot was persisted.
Use `workflow=` for new resumable applications.

## Invocations

Invocations has no platform-managed conversation history or native CreateResponse options. Supply an
application parser returning `InvocationRun(messages=..., options=..., stream=...)` for agents, or a
`WorkflowTurn(input=... | responses=..., stream=...)` for native workflows. A developer option hook and
`unsupported_options` policy work here too.

```python
async def parse_request(request):
    payload = await request.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("message"), str):
        raise ValueError("message must be a string")
    return InvocationRun(messages=payload["message"], options=payload.get("options", {}))


InvocationsHostServer(agent=agent, parse_request=parse_request).run()
```

The SDK binds a request to a Foundry sandbox from **`?agent_session_id=...`** on the Invocations URL—not
from the JSON body or a custom header. The host reloads MAF `AgentSession` state for each turn across process restarts and
uses session-scoped workflow checkpoints. Non-streaming agent requests return JSON with `response` text;
streaming requests use actual SSE `delta`/`done` or `error` events. Native workflows return their output and
request-info events as JSON or SSE.

## Isolation and state lifecycle

| Identity | Scope |
|---|---|
| Foundry `agent_session_id` | The platform's sandbox, `$HOME`, and session files. |
| Responses `conversation.id` / `response.id` | Caller-visible history or one stored turn. A background response's ID is the caller's polling handle. |
| MAF `AgentSession` | Inner agent/provider state. Its `service_session_id` is a separate, optional downstream service continuation ID. |
| MAF checkpoint ID | Graph, executor, nested-agent, and pending-input state. It is not exposed as the Foundry session ID. |

Hosted state stores use the platform-supplied user/call ID for user isolation and additionally use a safe namespace
derived from the trusted Foundry session ID. Workflow checkpoints and response cursors are separated by workflow
lineage. The default MAF session store conditionally updates a loaded conversation; concurrent turns conflict
instead of silently overwriting its state. Already-completed external side effects cannot be rolled back.
Missing hosted identity fails closed. The `FoundryStateStore` default persists independently of the
sandbox filesystem; deleting or expiring one does **not** imply automatic deletion of the other. Align retention
and clean up application-owned stores when deleting a session. Custom store providers must enforce the same
boundaries. Local runs use AgentServer's single-user file-backed fallback and cannot establish the deployed
platform's cross-user isolation guarantee.

The hosted session-file API limits access to the platform sandbox. Application file tools, remote MCP tools,
memory providers, and custom databases must separately enforce their own paths, identity and authorization.
Never treat a caller-provided Responses option as the trusted platform user or session scope.

## Live integration coverage

The integration suite exercises real-model Responses and Invocations, the `store=False`/`store=True`
foreground paths, SSE without storage, `background=True` polling and invalid combinations, service-managed
history, provider background polling, and native workflow checkpoints. An additional set runs against an
**active deployed** Responses agent to verify Foundry gateway behavior rather than only the local ASGI host.

With `az login` complete and `FOUNDRY_PROJECT_ENDPOINT` plus `FOUNDRY_MODEL` in your local `.env`, run from
`python/`:

```bash
FOUNDRY_DEPLOYED_AGENT_NAME=your-active-agent \
  uv run --env-file /path/to/your/.env --directory packages/foundry_hosting \
  pytest -q -m integration tests
```

Omit `FOUNDRY_DEPLOYED_AGENT_NAME` when no agent is deployed; only the deployed-gateway checks are skipped.
The test prompts are synthetic, but live model and hosted invocations can incur usage charges.

## Upstream dependencies

AgentServer currently includes failed response inputs in later conversation history
([Azure/azure-sdk-for-python#48929](https://github.com/Azure/azure-sdk-for-python/issues/48929)).
The standard MCP call's approval-request link is also waiting on an AgentServer builder addition
([Azure/azure-sdk-for-python#49037](https://github.com/Azure/azure-sdk-for-python/pull/49037)).
These are not fixed by inventing a second Responses persistence or output builder here.
