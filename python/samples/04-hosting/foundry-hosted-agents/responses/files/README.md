# Responses: files in a hosted session

[`main.py`](main.py) hosts an agent with two local tools and a Foundry Toolbox code interpreter. `list_files()` lists
regular files in the sample's dedicated directory; `read_file(filename)` accepts only a single file name, rejects
traversal and symlinks, and reads at most 1 MB of UTF-8 text. Neither tool accepts an arbitrary filesystem path.
The Toolbox's own container files and citations are **not** the same resource as this agent's session files.

The tools use `$HOME/sample_files/`: on Foundry this is inside the current hosted session's sandbox. They do not
switch to the repository directory when a request lacks an environment variable. Locally, `$HOME` is shared by
your own runs rather than isolated by the platform. File and directory opens refuse symlinks without a
time-of-check/time-of-use gap; the sample requires POSIX `O_NOFOLLOW` and `O_DIRECTORY` support and fails closed
without them.
The Foundry `agent_session_id` owns that sandbox and its files; `AgentSession.session_id`, `conversation.id`, and
`response.id` serve other purposes. Local runs do not prove the platform's user-isolation guarantee.

The host creates an agent and Toolbox connection for each request so the connection cannot retain a previous
request's Foundry call ID. It uses `inner_history="host"`: stored outer Responses history is supplied to the
inner agent, which does not also store its model-side history.

## Running locally

Follow the [parent hosting guide](../../README.md) to configure `FOUNDRY_PROJECT_ENDPOINT` and
`AZURE_AI_MODEL_DEPLOYMENT_NAME`, then run `python main.py`. If the Toolbox is configured with a named endpoint,
also set `TOOLBOX_NAME` or `FOUNDRY_TOOLBOX_ENDPOINT`. The local `resources/` directory includes a sample report;
copy it into your local sample folder before invoking the file tools:

```bash
mkdir -p "$HOME/sample_files"
cp resources/contoso_q1_2026_report.txt "$HOME/sample_files/"
```

```bash
curl -X POST http://localhost:8088/responses -H "Content-Type: application/json" \
  -d '{"input": "List the sample files and summarize the quarterly report.", "store": true}'
```

## Uploading to a deployed Foundry session

A file must be uploaded to the **same** Foundry `agent_session_id` that subsequent invocations use. Run
[`upload_file.py`](upload_file.py) after creating a hosted session or receiving an ID from the first response:

```bash
python upload_file.py REPLACE_WITH_AGENT_SESSION_ID resources/contoso_q1_2026_report.txt
```

The script uses the [Session Files API](https://learn.microsoft.com/en-us/azure/foundry/agents/how-to/manage-hosted-sessions#session-file-operations)
to place the file at `sample_files/contoso_q1_2026_report.txt`. It needs `FOUNDRY_PROJECT_ENDPOINT`,
`FOUNDRY_AGENT_NAME`, and an authenticated Azure CLI session. When invoking the hosted agent again, pass the
returned `agent_session_id` in the Responses body (or use a `conversation` that Foundry has already bound to it):

A deployed Foundry test confirmed that an upload at `sample_files/hosting-canary.txt` is readable through
`$HOME/sample_files/` in its hosted session, and is absent from a second hosted session.

```bash
curl -X POST http://localhost:8088/responses -H "Content-Type: application/json" \
  -d '{"input": "Read contoso_q1_2026_report.txt.", "agent_session_id": "REPLACE_WITH_AGENT_SESSION_ID", "store": true}'
```

`previous_response_id` alone continues conversation history but does **not** route to the original file sandbox.
Other callers must not gain access to files by supplying someone else's session ID. See the [Foundry session
isolation guide](https://learn.microsoft.com/en-us/azure/foundry/agents/how-to/isolate-sessions-per-user) for the
platform boundary; application-owned external stores and tools must enforce their own scope.
