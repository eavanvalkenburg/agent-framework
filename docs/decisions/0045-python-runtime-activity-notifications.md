---
status: proposed
date: 2026-09-30
deciders: []
---

# Python runtime activity notifications

## Context and Problem Statement

Developers need live notifications to update a UI during agent and workflow execution: chat requests, tool calls,
function-loop iterations, MCP connection lifecycle, context-provider contributions, compaction, and workflow steps.
These notifications describe what is happening, not the content being produced or the state needed to resume execution.

This record compares listener scope, delivery, and instrumentation options. These are separate choices and can be
combined. API sketches below are illustrative; no design has been selected.

## Decision Drivers

- **Do not make the simple case more complex.** The feature must be purely additive and opt-in. Existing applications
  keep their setup, calls, return types, stream item types, and persistence behavior. No mandatory observer, consumer,
  queue, or telemetry configuration is introduced.
- **Keep events transient.** The framework must not accumulate notifications in sessions, responses, checkpoints, or
  streaming history, or automatically export them through logging or OpenTelemetry. Applications own any forwarding,
  delivery buffering, or deliberate storage.
- **Keep responsibilities clear.** Notifications report activity; content remains in responses and streaming, and
  continuation state remains in sessions and checkpoints. Observation is not an approval or interception mechanism.
- **Keep opt-in usage simple.** A basic UI should be able to supply one callable without adopting a new event framework.

## Event Shape (Illustrative)

A common envelope could carry a type discriminator, source identity, operation nesting, optional run/request
correlation, and small type-specific metadata. This sketch does not select a schema, serialization format, or Python
representation:

```json
{
  "type": "compaction.complete",
  "source": "history",
  "component_id": "compactor-1",
  "operation_id": "operation-3",
  "parent_operation_id": "operation-2",
  "run_id": "run-1",
  "correlation_id": "ui-request-1",
  "data": {
    "strategy": "SlidingWindowStrategy",
    "phase": "before_run",
    "changed": true,
    "included_messages_before": 24,
    "included_messages_after": 12
  }
}
```

`component_id` identifies the emitting instance; `operation_id` identifies an operation and links its lifecycle events, while
`parent_operation_id` expresses nesting and is null for root operations. `run_id` is null for activity outside an
invocation, such as independent MCP setup or teardown. `correlation_id` is optional application-supplied UI/request
routing metadata, not session state.

Type-specific data could contain tool names/call IDs, added tool names, middleware categories, iteration numbers,
strategy/phase, counts, or sanitized error categories. Payloads are detached metadata snapshots, not live objects,
instruction/message text, tool arguments/results, or conversation state.

### Proposed Event Types

This initial catalog is for discussion, not a stable contract or a choice of delivery mechanism. Names use dotted nouns
and short action verbs, such as `agent.start`; they report activity rather than issuing commands.

| Area | Proposed types | Purpose |
| --- | --- | --- |
| Agent | `agent.start`, `agent.complete`, `agent.fail`, `agent.cancel`, `agent.pause` | Invocation lifecycle, including waiting for user input. |
| Workflow | `workflow.start`, `workflow.complete`, `workflow.fail`, `workflow.cancel`, `workflow.pause`, `workflow.restore` | Execution lifecycle and checkpoint restoration. |
| Workflow executor | `workflow.executor.start`, `workflow.executor.complete`, `workflow.executor.fail`, `workflow.executor.cancel`, `workflow.executor.pause`, `workflow.executor.skip` | Individual executor/step activity, including replay or cache bypasses. |
| Workflow superstep | `workflow.superstep.start`, `workflow.superstep.complete` | Iteration progress, with the superstep number. |
| Chat client | `chat.start`, `chat.complete`, `chat.fail`, `chat.cancel` | One actual model request, not each forwarding layer. |
| Function loop | `loop.start`, `loop.iteration.start`, `loop.iteration.complete`, `loop.stop`, `loop.fail`, `loop.cancel` | Model/tool iteration progress and why the loop ended. |
| Tool call | `tool.start`, `tool.complete`, `tool.fail`, `tool.cancel`, `tool.pause`, `tool.observe` | Call-handling lifecycle, including preparation and finalization, or provider-reported activity. |
| Tool arguments | `tool.arguments.parse`, `tool.arguments.validate` | Completion of argument parsing/coercion and schema validation/preparation. |
| Tool invocation | `tool.invoke.start`, `tool.invoke.complete`, `tool.invoke.fail`, `tool.invoke.cancel` | Actual function/MCP invocation after argument and policy checks, distinct from call handling. |
| Tool result | `tool.result.parse`, `tool.result.format`, `tool.result.skip` | Return-value parsing/normalization, final function-result envelope construction, or a deliberate parser bypass. |
| Approval and input | `tool.approval.request`, `tool.approval.resolve`, `input.request`, `input.resolve` | Advisory approval status or non-approval input requests, identified by request IDs. |
| Context provider | `context.start`, `context.complete`, `context.fail`, `context.cancel` | Provider hook lifecycle, with the before/after-run phase. |
| Context contributions | `context.tools.add`, `context.instructions.add`, `context.messages.add`, `context.middleware.add` | Each batch added through the context provider's tools, instructions, messages, or middleware contribution APIs. |
| Tool exposure | `tools.update` | Effective additions/removals available to the next model request. |
| Compaction | `compaction.start`, `compaction.complete`, `compaction.fail`, `compaction.cancel` | Strategy/phase and available before/after counts; completion includes `changed`. |
| MCP lifecycle | `mcp.connect.start`, `mcp.connect.complete`, `mcp.connect.fail`, `mcp.connect.cancel`, `mcp.disconnect.start`, `mcp.disconnect.complete`, `mcp.disconnect.fail`, `mcp.disconnect.cancel` | Connection setup and teardown, including activity outside a run. |

- Lifecycle notifications describe actual outcomes: a recovered `tool.fail` need not fail the agent, and input pauses
  are not failures. Replay/cache skips do not re-emit historical tool activity.
- Argument/result parse, validate, and format events are completion milestones for work the pipeline already performs,
  not additional processing passes. `tool.complete` follows the applicable preparation, invocation, and result handling;
  `tool.invoke.*` distinguishes actual invocation from validation, approval, or middleware short-circuits.
- Tool failures/cancellations identify the phase. `tool.result.skip` carries a reason such as disabled parsing or an
  already-parsed result; parser fallbacks are reported as metadata rather than treated as successful primary parsing.
- Context contribution events report each add/extend batch, not just a final summary. Metadata identifies the provider
  and phase, with item counts, tool names, or middleware categories; text and live objects remain outside the payload.
  Contributing tools is distinct from their effective exposure through `tools.update`.
- `loop.stop` carries a reason such as final response, approval/input, safety limit, or middleware termination.
- Reconnection uses the MCP connect types with a reconnect reason rather than a separate event family.
  `compaction.complete` may have `changed=false`.
- Approval/input notifications do not carry authority or replace existing response/request handling. `tool.observe`
  reports only provider-supplied facts, without inventing local execution timing.

For example, `tool.arguments.parse` could carry
`{"input_format": "json", "argument_names": ["query", "limit"], "argument_count": 2}`, while `tool.result.parse` could carry
`{"item_count": 1, "content_types": ["text"], "fallback": false}`. These describe argument/result shape and processing
outcome, not argument values, result content, or host-only runtime inputs.

## Considered Options

### Listener scope

| Option | Pros for developers | Cons for developers |
| --- | --- | --- |
| Run-scoped callback: `run(..., on_event=callback)` or `get_response(..., on_event=callback)` | One explicit argument; precise ownership for a single invocation; works without content streaming. | Registration is repeated across calls; MCP setup and teardown outside the run are not observed. |
| Object-scoped callback on `Agent` or `WorkflowBuilder` construction | Configure once for repeated runs and managed resource lifecycles. | Shared objects need UI routing; independently managed resources remain outside the object's scope. |
| Application-wide registration, similar to OpenTelemetry setup | One startup registration covers instrumented operations, including independently managed resources. | Process-wide configuration needs teardown, test isolation, and request routing; interaction with local listeners must be defined. |
| Scoped observation: `with observe_events(callback)` | No changes to run calls or object configuration; separate request scopes can observe shared objects independently. | The scope must cover actual stream consumption and resource teardown; spawned work must respect the listener's lifetime. |

### Delivery and instrumentation

| Option | Pros for developers | Cons for developers |
| --- | --- | --- |
| Caller-owned async channel, using a sender such as `channel.send` as the callback | UI code can consume events with `async for`; buffering and backpressure are application-owned. | Requires producer/consumer coordination, cancellation, and channel closure; a slow receiver can stall execution. |
| Multicast subscriptions: `subscribe(callback)` | Multiple UI views and monitoring consumers can subscribe independently. | Requires filters, unsubscribe handling, and policies for slow or failing listeners. |
| Middleware-based observation | Reuses existing agent, chat, and function middleware; useful for incremental adoption. | Invocation interception is not proof of actual execution; MCP lifecycle, provider contributions, and compaction need additional emission points. |
| Structured logging with an application handler | Familiar application-wide handler and configuration model. | Levels, filters, and ordinary exporters can suppress or retain notifications; async UI delivery needs a bridge and isolated configuration. |

### Unified content and notification streaming

- **Pros:** One iterator delivers content and runtime status; familiar to developers already consuming workflow events.
- **Cons:** Expands stream item types and forces consumers to distinguish status from content. The current
  [`ResponseStream`](../../python/packages/core/agent_framework/_types.py) accumulates updates for response finalization,
  so transient notifications would also require special accumulation and filtering behavior.

**Dismissed.** Changing existing stream contracts makes the simple case more complex and mixes transient observation
with retained output. This approach does not meet the purely additive requirement.

## Decision Outcome

Pending. No remaining listener scope, delivery mechanism, or instrumentation approach has been selected.
Deciders must be nominated before review.
