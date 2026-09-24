# Copyright (c) Microsoft. All rights reserved.

"""Host an agent with an explicit Invocations request parser and durable session state."""

import os
from typing import Any

from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from agent_framework_foundry_hosting import InvocationRun, InvocationsHostServer
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from starlette.requests import Request

load_dotenv()


async def parse_request(request: Request) -> InvocationRun:
    """Validate the application's JSON shape before handing it to the agent."""
    payload: Any = await request.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("message"), str):
        raise ValueError("message must be a string")
    options = payload.get("options", {})
    if not isinstance(options, dict):
        raise ValueError("options must be an object")
    stream = payload.get("stream", False)
    if not isinstance(stream, bool):
        raise ValueError("stream must be a boolean")
    return InvocationRun(messages=payload["message"], options=options, stream=stream)


def prepare_options(request: Request, options: dict[str, Any]) -> dict[str, Any]:
    """Keep caller JSON from changing the agent's storage and conversation identity."""
    options.pop("store", None)
    options.pop("conversation_id", None)
    return options


def main() -> None:
    agent = Agent(
        client=FoundryChatClient(
            project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
            model=os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"],
            credential=DefaultAzureCredential(),
        ),
        instructions="You are a friendly assistant. Keep your answers brief.",
        default_options={"store": False},
    )
    InvocationsHostServer(
        agent=agent,
        parse_request=parse_request,
        prepare_options=prepare_options,
        unsupported_options="warn",
    ).run()


if __name__ == "__main__":
    main()
