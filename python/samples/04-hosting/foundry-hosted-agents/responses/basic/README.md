# Responses: agent history and caller options

The **outer** Responses request controls whether the hosted response is stored. The developer independently selects
how the **inner** agent receives history:

| Entry point | `inner_history` | Model history |
|---|---|---|
| [`main.py`](main.py) | `"host"` | AgentServer supplies the prior Responses transcript; the inner client runs with `store=False`. |
| [`service_history.py`](service_history.py) | `"service"` | The inner service stores history; its `AgentSession.service_session_id` stays private in Foundry-backed state. The model receives only new input. |
| [`agent_history.py`](agent_history.py) | `"agent"` | The agent's `InMemoryHistoryProvider` loads history from `AgentSession.state`, which the host persists in Foundry. The inner client runs with `store=False`. |
| [`options.py`](options.py) | `"host"` | The hook removes the caller's output-token override, exposing the agent's `default_options["max_tokens"]`. |
| [`provider_background.py`](provider_background.py) | `"service"` | Opt-in to provider background execution. The host keeps the provider token private, persists it for recovery, and exposes only the outer response ID. |

Only one entry point runs at a time. The deployment manifests in this directory target `main.py`; to deploy another
entry point, select that script when configuring the agent service.

## Outer Responses state

- `store=True` stores the caller-facing response and its input, enabling `GET /responses/{response.id}`,
  `previous_response_id`, or continuation under a `conversation` ID. It does not prescribe the inner history mode.
- `store=False` returns the complete one-shot response but does not durably save framework-managed history,
  `AgentSession` changes, checkpoints, or approvals. Its protocol response ID is transient and cannot be retrieved.
  Application-owned tools or external memory providers can still have their own side effects.
- `background=True` requires `store=True`. AgentServer immediately returns a queued/in-progress response with
  `response.id`; polling retrieves that **outer** ID. It does not automatically enable provider background mode.
  See [`client.py`](client.py) for a stored conversation, an unstored request, and background polling.
- A regular agent using `main.py` can run in the outer background even if its chat client has **no** provider
  background API. Without a durable inner continuation or an explicitly safe rerun strategy, a process crash can
  leave the response in progress. [`provider_background.py`](provider_background.py) shows the opt-in alternative
  for a client that returns a MAF continuation token: the host polls it until completion and records it under the
  outer response ID for recovery. A crash before the provider token is recorded cannot be assumed safe to retry.
  The deployed agent's identity needs **Foundry User** on the project for the private provider-response poll.

The host's `response_store=` constructor argument selects the **backend** for outer response persistence; it is not
the request's `store` flag. `AgentSession.session_id`, a downstream `service_session_id`, the caller's
`response.id`/`conversation.id`, and the Foundry `agent_session_id` are distinct. The last of these identifies the
isolated sandbox and uploaded files. A `conversation` automatically binds that sandbox; a bare `previous_response_id`
does not. When continuing by response ID and needing the same sandbox, also pass its `agent_session_id`.

## Option precedence

The host translates supported Responses generation options to MAF runtime options. The OpenAI client's `extra_body`
fields are flattened into the request; when one maps to the same MAF option, the extra value wins. The developer's
`prepare_options` hook can then remove or replace a caller runtime option. A removed option falls back to the
developer-owned `Agent.default_options`; the host does not modify those defaults. `unsupported_options` is
configurable as `"ignore"`, `"warn"`, or `"error"`.

For example, invoke the deployed `options.py` host with `max_output_tokens=300` and
`extra_body={"max_tokens": 150}`. The extra-body value wins the translated-key collision first; the hook
then removes the caller's `max_tokens`, so the inner agent uses its developer-owned default of `256`.
Same-wire-key collisions are resolved by the client before the request reaches the host.

## Run and deploy

Set `FOUNDRY_PROJECT_ENDPOINT` and `AZURE_AI_MODEL_DEPLOYMENT_NAME` in `.env`, then run `python main.py`. The optional
[`client.py`](client.py) also needs `FOUNDRY_AGENT_NAME` and an authenticated Azure CLI session. Follow the
[parent hosting guide](../../README.md) for local and Foundry deployment instructions.
