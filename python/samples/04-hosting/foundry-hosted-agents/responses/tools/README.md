# What this sample demonstrates

An [Agent Framework](https://github.com/microsoft/agent-framework) agent with **locally-defined Python tools** hosted using the **Responses protocol**. It shows how to define custom tools with the `@tool` decorator and register them with the agent so the model can call them during a conversation.

## How It Works

### Model Integration

The agent uses `FoundryChatClient` from the Agent Framework to create a Responses client from the project endpoint and model deployment. The agent supports both streaming (SSE events) and non-streaming (JSON) response modes.

See [main.py](main.py) for the full implementation.

### Tools

Local tools are Python functions decorated with the Agent Framework's `@tool` decorator and registered with the agent. When the model chooses to call a tool during a conversation, the agent executes the corresponding function and returns the result to the model.

`get_weather` is read-only and does not require approval. `save_forecast` simulates a write and requires approval; it
does not execute a shell command or change an external service.

When `save_forecast` is called, the agent host emits an `mcp_approval_request` containing an approval request ID.
The client replies with an `mcp_approval_response` using that ID and the same Foundry session, after the host has
stored the original response. Cross-turn approval requires `store=true`; the pending request and agent state are
bound to the trusted user, sandbox, and response lineage.

> IMPORTANT: We are temporarily reusing the **mcp_approval_request** and **mcp_approval_response** message types defined in the [AzureAI AgentServer SDK](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/agentserver/azure-ai-agentserver-responses/docs/handler-implementation-guide.md#other-tool-call-types) because they map closely to this approval flow. They will likely be superseded by a more formal tool-approval content type in the Responses protocol in the future.

### Agent Hosting

The agent is hosted using the [Agent Framework](https://github.com/microsoft/agent-framework) with the `ResponsesHostServer`, which provisions a REST API endpoint compatible with the OpenAI Responses protocol.

## Running the Agent Host

Follow the instructions in the [Running the Agent Host Locally](../../README.md#running-the-agent-host-locally) section of the README in the parent directory to run the agent host.

## Interacting with the agent

> Depending on how you run the agent host, you can invoke the agent using `curl` (`Invoke-WebRequest` in PowerShell) or `azd`. Please refer to the [parent README](../../README.md) for more details. Use this README for sample queries you can send to the agent.

Send a POST request to the server with a JSON body containing an `"input"` field to interact with the agent. For example:

```bash
curl -X POST http://localhost:8088/responses -H "Content-Type: application/json" \
  -d '{"input": "What is the weather in Seattle?", "store": true}'
```

Send a POST request that triggers a tool call configured with `always_require` to see the approval flow in action:

```bash
curl -X POST http://localhost:8088/responses -H "Content-Type: application/json" \
  -d '{"input": "Save a forecast for Seattle.", "store": true}'
```

The output includes a caller-facing `response.id`, `agent_session_id`, and an `mcp_approval_request` ID. To approve:

```bash
curl -X POST http://localhost:8088/responses -H "Content-Type: application/json" \
  -d '{"input": [{"type": "mcp_approval_response", "approval_request_id": "REPLACE_WITH_APPROVAL_ID", "approve": true}], "previous_response_id": "REPLACE_WITH_RESPONSE_ID", "agent_session_id": "REPLACE_WITH_AGENT_SESSION_ID", "store": true}'
```

## Deploying the Agent to Foundry

To host the agent on Foundry, follow the instructions in the [Deploying the Agent to Foundry](../../README.md#deploying-the-agent-to-foundry) section of the README in the parent directory.
