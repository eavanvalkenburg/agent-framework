# Copyright (c) Microsoft. All rights reserved.

"""Show how hosted request options interact with developer-owned agent defaults."""

import os
from typing import Any

from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from agent_framework.openai import OpenAIChatOptions
from agent_framework_foundry_hosting import HostedResponseRequest, ResponsesHostServer
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()


def prepare_options(_request: HostedResponseRequest, options: dict[str, Any]) -> dict[str, Any]:
    """Use the agent's output-token default rather than the caller's value."""
    options.pop("max_tokens", None)
    return options


def main() -> None:
    agent = Agent(
        client=FoundryChatClient(
            project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
            model=os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"],
            credential=DefaultAzureCredential(),
        ),
        instructions="Be concise.",
        default_options=OpenAIChatOptions(max_tokens=256),
    )
    ResponsesHostServer(
        agent=agent,
        inner_history="host",
        unsupported_options="warn",
        prepare_options=prepare_options,
    ).run()


if __name__ == "__main__":
    main()
