# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os

from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient, FoundryToolbox
from agent_framework_foundry_hosting import ResponsesHostServer
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def create_agent() -> Agent:
    """Own the Toolbox connection for the request that created this agent."""
    credential = DefaultAzureCredential()

    # FoundryToolbox resolves the toolbox endpoint from the environment
    # (TOOLBOX_ENDPOINT, or FOUNDRY_PROJECT_ENDPOINT + TOOLBOX_NAME), authenticates
    # every request with the credential, and forwards the platform per-request
    # call-id. ``load_tools=False`` keeps the toolbox's tools hidden so only its
    # Agent Skills (SEP-2640) are surfaced; passing it via ``tools=`` connects the
    # MCP session that ``as_skills_provider()`` reads from.
    toolbox = FoundryToolbox(credential, load_tools=False)

    # as_skills_provider() discovers skills from skill://index.json on the toolbox
    # MCP session and exposes them as an agent context provider; SKILL.md bodies are
    # fetched on demand via resources/read. disable_load_skill_approval=True registers
    # the load_skill tool with approval_mode="never_require" so this unattended agent
    # can load skills without an approval round-trip -- the Responses host runs the
    # agent without an interactive approval step in this sample.
    skills_provider = toolbox.as_skills_provider(disable_load_skill_approval=True)

    client = FoundryChatClient(
        project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
        model=os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"],
        credential=credential,
    )

    return Agent(
        client=client,
        name=os.environ.get("AGENT_NAME", "hosted-toolbox-mcp-skills"),
        instructions="You are a helpful assistant.",
        tools=toolbox,
        context_providers=[skills_provider],
    )


async def main() -> None:
    server = ResponsesHostServer(agent=create_agent, inner_history="host")
    await server.run_async()


if __name__ == "__main__":
    asyncio.run(main())
