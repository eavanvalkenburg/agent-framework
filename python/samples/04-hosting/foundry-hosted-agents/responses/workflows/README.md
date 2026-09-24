# Responses: native workflow with typed input

[`main.py`](main.py) serves a native `Workflow`, not a `WorkflowAgent`. Its **required** `parse_response` callback
converts the current Responses input into a `SloganRequest`, the type accepted by the start executor. On each turn
the host calls `build_workflow(request)`; that function must call `.build()` itself and return a fresh workflow with
the same workflow name and executor IDs so an authorized checkpoint can be restored.

The start executor writes `slogan_style` to shared workflow state with `ctx.set_state()`, then passes an
`AgentExecutorRequest` to the writer. The writer, legal reviewer, and formatter use distinct stable IDs; only the
formatter yields caller-facing output. This is the current core-supported path for setting application state before
the rest of the graph runs. A generic `Workflow.run(state_updates=..., agent_options=...)` is tracked separately in
[microsoft/agent-framework#8711](https://github.com/microsoft/agent-framework/issues/8711), not assumed here.

The factory also reads normalized `request.options` (native Responses options followed by `extra_body` on a mapped
collision) and uses `max_tokens` to set the writer's `default_options` for this request. The other agents retain
their own defaults; host options are not broadcast blindly to every agent. The current core
`function_invocation_kwargs` and `client_kwargs` can target specific executor IDs when needed, but those values are
checkpointed, so they must not contain passwords or short-lived call IDs.

## Checkpoints and Foundry sessions

When the caller uses `store=True`, the host associates each caller-facing `response.id` with the **exact**
workflow checkpoint that produced that output. The Foundry `agent_session_id` owns the sandbox and files, while the
workflow checkpoint holds graph state and nested MAF `AgentSession` values. These IDs are not interchangeable.
Checkpoint records use Foundry State Store by default, isolated by platform user and the verified Foundry session,
workflow, and response/conversation lineage; they are not implicitly stored in `$HOME`.

A `conversation` automatically binds a Foundry session. If the caller uses `previous_response_id` instead and needs
to resume checkpointed work, it must also reuse the matching `agent_session_id`. Restoring into a different sandbox
must fail. A workflow branch from an older response is not supported until checkpoint and shared-file fork semantics
are defined; ordinary linear continuation is supported.

## Run

After the new host API is implemented, set `FOUNDRY_PROJECT_ENDPOINT` and `AZURE_AI_MODEL_DEPLOYMENT_NAME`, then
run `python main.py` or follow the [parent guide](../../README.md) to deploy it. Send a stored request:

```bash
curl -X POST http://localhost:8088/responses -H "Content-Type: application/json" \
  -d '{"input": "An affordable, fun electric SUV", "store": true, "slogan_style": "retro"}'
```

The caller can instead use the OpenAI client's `extra_body={"slogan_style": "retro"}` to provide the same custom
field. The extra field is for this application's parser; it is not automatically sent to the writer model.

## Approval and human-input continuation

[`approval.py`](approval.py) is a second, self-contained workflow host in this folder. It demonstrates an
`AgentExecutor` whose `publish_summary` tool requires approval. The first **stored** response includes an
`mcp_approval_request`. The caller sends an `mcp_approval_response` with the returned approval ID and the first
`response.id` as `previous_response_id`; it must also use the matching Foundry `agent_session_id`. The parser uses
the host's validated `get_workflow_responses()` conversion, so an approval for another session or an unknown pending
request cannot be injected into this workflow. The host restores the exact prior checkpoint before delivering the
approval via `Workflow.run(responses=...)`.

```bash
curl -X POST http://localhost:8088/responses -H "Content-Type: application/json" \
  -d '{"input": "Summarize the ticket and publish it", "store": true}'

curl -X POST http://localhost:8088/responses -H "Content-Type: application/json" \
  -d '{"input": [{"type": "mcp_approval_response", "approval_request_id": "REPLACE_WITH_APPROVAL_ID", "approve": true}], "previous_response_id": "REPLACE_WITH_RESPONSE_ID", "agent_session_id": "REPLACE_WITH_AGENT_SESSION_ID", "store": true}'
```

This tool only **simulates** publishing; it makes no external changes. With `store=False`, cross-turn approval or
user input is not resumable, so the host must fail clearly rather than return an unusable approval request.
