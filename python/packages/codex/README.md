# Get Started with Microsoft Agent Framework Codex

Install the provider package:

```bash
pip install agent-framework-codex --pre
```

## Codex Integration

The Codex integration adds a managed-agent wrapper around OpenAI Codex via the
`codex-sdk-python` package. It supports:

- Non-streaming and streaming responses
- Thread-backed session reuse through `AgentSession.service_session_id`
- Structured output through `response_format`

## Authentication

Authenticate with the Codex CLI or provide an API key for the underlying `codex` process:

```bash
codex login
# or:
export CODEX_API_KEY="your-api-key"
```

## Package Settings

The package reads configuration from `CODEX_AGENT_*` environment variables:

```bash
export CODEX_AGENT_MODEL="gpt-5.4"
export CODEX_AGENT_CWD="/path/to/project"
export CODEX_AGENT_APPROVAL_POLICY="on-request"
```

## Current Limitations

- Agent Framework custom `tools=` are not yet supported by `CodexAgent`
- Background execution is not currently supported

## Examples

See the [package-local samples](samples/) for runnable examples.
