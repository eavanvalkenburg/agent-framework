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
    """Keep the Toolbox MCP connection within one request's caller context."""
    credential = DefaultAzureCredential()

    # FoundryToolbox resolves the toolbox endpoint from the environment
    # (TOOLBOX_ENDPOINT, or FOUNDRY_PROJECT_ENDPOINT + TOOLBOX_NAME), authenticates
    # every request with the credential, and transparently forwards the platform
    # call-id to the toolbox. A request-scoped agent prevents the MCP connection
    # from retaining a previous request's identity in a long-lived writer task.
    toolbox = FoundryToolbox(credential)

    # Create the chat client
    client = FoundryChatClient(
        project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
        model=os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"],
        credential=credential,
    )

    return Agent(
        client=client,
        instructions="You are a friendly assistant. Keep your answers brief.",
        tools=toolbox,
    )


async def main() -> None:
    server = ResponsesHostServer(agent=create_agent, inner_history="host")
    await server.run_async()


if __name__ == "__main__":
    asyncio.run(main())
