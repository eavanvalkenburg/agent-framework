# What this sample demonstrates

A realistic **multi-turn** [Agent Framework](https://github.com/microsoft/agent-framework) **declarative workflow** — defined entirely in YAML — hosted using the **Responses protocol**. The host runs the native workflow without `Workflow.as_agent()`.

Read more about declarative workflows in the [Agent Framework documentation](https://learn.microsoft.com/en-us/agent-framework/workflows/declarative/?pivots=programming-language-python).

## How It Works

### The Workflow

[`workflow.yaml`](workflow.yaml) describes a customer-support triage flow:

1. `InvokeAzureAgent: TriageAgent` — looks at the full conversation so far and emits a structured `TriageResponse` (`Category`, `NeedsClarification`, `ClarificationQuestion`, `Reply`).
2. `ConditionGroup` routes on the triage decision:
   - **NeedsClarification** → `SendActivity` asks one focused follow-up question and ends the turn.
   - **Category = "Technical"** → `SendActivity` confirms the handoff, then `InvokeAzureAgent: TechSupportAgent` answers with `autoSend: true` so its reply streams directly to the caller.
   - **Category = "Billing"** → same pattern, routed to `BillingAgent`.
   - **else** → `SendActivity` returns the triage agent's `Reply` directly (good for greetings or general questions).

The required `parse_response` callback passes only the **current turn's** `list[Message]` to the workflow. On continuation, the host restores the scoped checkpoint first; prior `Conversation.messages` is already present in workflow state. Feeding the entire Responses transcript again would duplicate earlier turns and possibly tool results. The workflow updates its conversation state as it runs.

### Agent Hosting

[`main.py`](main.py) gives `ResponsesHostServer` a request-aware callable that builds three `Agent` instances on top of
a shared `FoundryChatClient`, registers them with `WorkflowFactory`, and returns a **built** `Workflow`. Each request
receives a fresh workflow and agents. The host binds its checkpoint to the trusted Foundry `agent_session_id` and
the caller's response/conversation lineage; this is not the model service's conversation ID.

The triage agent is configured with `response_format=TriageResponse` (a Pydantic model) so the workflow can read its structured fields via `Local.Triage.*`. The specialist agents are plain text and use `autoSend: true` to deliver their reply straight to the caller.

## Running the Agent Host

Follow the instructions in the [Running the Agent Host Locally](../../README.md#running-the-agent-host-locally) section of the README in the parent directory to run the agent host.

## Interacting with the agent

> Depending on how you run the agent host, you can invoke the agent using `curl` (`Invoke-WebRequest` in PowerShell) or `azd`. Please refer to the [parent README](../../README.md) for more details. Use this README for sample queries you can send to the agent.

Send a POST request to the server with a JSON body containing an `"input"` field to interact with the agent. For example:

```bash
curl -X POST http://localhost:8088/responses -H "Content-Type: application/json" -d '{"input": "I have a problem"}'
```

Invoke with `azd`:

```bash
azd ai agent invoke --local "I was double-charged this month"
# → "Connecting you with billing support..."
# → BillingAgent: "I'm sorry about that. Can you share the last 4 digits of the card on file?"
```

## Deploying the Agent to Foundry

To host the agent on Foundry, follow the instructions in the [Deploying the Agent to Foundry](../../README.md#deploying-the-agent-to-foundry) section of the README in the parent directory.

> [!IMPORTANT]
> Deploy this sample as a **container** (not Code/ZIP). Its declarative workflow uses Power Fx, which needs the .NET runtime included in the `Dockerfile`. Choose **Container** in every deploy flow.