# Codex Examples

This folder contains runnable examples for the alpha `agent-framework-codex` package.

## Examples

| File | Description |
|------|-------------|
| [`step1_getting_started.py`](step1_getting_started.py) | Basic Codex agent usage with a one-off request, streaming, and thread-backed session reuse. |

## Requirements

- A working Codex CLI login (`codex login`) or `CODEX_API_KEY`
- Optional `CODEX_AGENT_MODEL` if you want to pin the model explicitly

## Current Limitations

- Agent Framework custom `tools=` are not yet supported by `CodexAgent`
- Background execution is not currently supported
