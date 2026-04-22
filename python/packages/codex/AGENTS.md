# Codex Package (agent-framework-codex)

Integration with OpenAI Codex as a managed agent via the `codex-sdk-python` SDK.

## Main Classes

- **`RawCodexAgent`** - Lightweight Codex wrapper without middleware or telemetry layers
- **`CodexAgent`** - Codex agent with middleware and telemetry composition
- **`CodexAgentOptions`** - Options TypedDict for Codex-specific configuration
- **`CodexAgentSettings`** - TypedDict-based settings populated via the framework's `load_settings()` helper

## Supported Behavior

- Streaming and non-streaming runs
- Thread-backed session reuse through `AgentSession.service_session_id`
- Structured output via `response_format`

## Current Limitations

- Agent Framework custom `tools=` are not yet supported
- Background execution is not currently supported

## Usage

```python
from agent_framework_codex import CodexAgent

async with CodexAgent(instructions="You are a helpful coding assistant.") as agent:
    response = await agent.run("Explain Python context managers in two sentences.")
    print(response.text)
```
