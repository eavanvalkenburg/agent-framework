# Invocations: explicit request parsing

The Invocations protocol accepts application-defined JSON, not a native Responses option schema. [`main.py`](main.py)
validates `{"message": "...", "options": {...}, "stream": false}` and returns an `InvocationRun`. The host applies
the developer's option hook (if configured), then MAF's normal runtime-over-default option merge. Invalid request
shapes fail as client errors; unsupported runtime options use the configured `"ignore"`, `"warn"`, or `"error"` mode.
The example hook drops caller-supplied `store` and `conversation_id`, leaving the agent's developer-configured
`store=False` in effect instead of letting an arbitrary JSON body enable downstream model storage.

Invocations has **no platform-managed conversation history**. The host uses durable MAF session state scoped by the
trusted platform user and Foundry `agent_session_id`, rather than an in-process dictionary that disappears when the
container is suspended. The Foundry session also owns the sandbox files, but it is not an `AgentSession.session_id`.
The application request body cannot select a different user's session.

To invoke, start the host and send:

```bash
curl -i -X POST http://localhost:8088/invocations \
  -H "Content-Type: application/json" \
  -d '{"message": "Hi", "options": {"temperature": 0.3}}'
```

The host returns the reply and the platform supplies `x-agent-session-id` in the response headers. To continue in
that sandbox, put the returned ID in the **Invocations query parameter**, not in the JSON body:

```bash
curl -i -X POST "http://localhost:8088/invocations?agent_session_id=REPLACE_WITH_SESSION_ID" \
  -H "Content-Type: application/json" \
  -d '{"message": "What did I ask earlier?", "stream": true}'
```

The parser controls the payload-to-MAF mapping; `stream=true` selects proper SSE framing rather than raw text
mislabelled as an event stream. Follow the [parent hosting guide](../../README.md) for local and Foundry setup.

## Native workflow alternative

[`workflow.py`](workflow.py) is a self-contained alternative entry point using a typed `Ticket` start input. Its
parser validates the application JSON and the `TicketStart` executor records the ticket ID with `ctx.set_state()`.
The host calls the request-aware workflow factory on each invocation, restores a checkpoint scoped to the trusted
Foundry session, and delivers the new input. There is no Responses conversation ID or `workflow.as_agent()` wrapper.

```bash
curl -i -X POST http://localhost:8088/invocations \
  -H "Content-Type: application/json" \
  -d '{"ticket_id": "T-123", "question": "What is the status?"}'
```

To deploy this alternative, change the sample service's entry point from `main.py` to `workflow.py`.
