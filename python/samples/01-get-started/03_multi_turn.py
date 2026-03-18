# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os

from agent_framework import Agent
from agent_framework.openai import OpenAIResponsesClient
from azure.identity import AzureCliCredential
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

"""
Multi-Turn Conversations — Use AgentSession to maintain context

This sample shows how to keep conversation history across multiple calls
by reusing the same session object.

Environment variables:
  AZURE_AI_PROJECT_ENDPOINT        — Your Azure AI Foundry project endpoint
  AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME — Model deployment name (e.g. gpt-4o)
"""


async def main() -> None:
    # <create_agent>
    credential = AzureCliCredential()
    client = OpenAIResponsesClient(
        backend="foundry",
        project_endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"],
        model_id=os.environ["AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME"],
        credential=credential,
    )

    agent = Agent(
        client=client,
        name="ConversationAgent",
        instructions="You are a friendly assistant. Keep your answers brief.",
    )
    # </create_agent>

    # <multi_turn>
    # Create a session to maintain conversation history
    session = agent.create_session()

    # First turn
    result = await agent.run("My name is Alice and I love hiking.", session=session)
    print(f"Agent: {result}\n")

    # Second turn — the agent should remember the user's name and hobby
    result = await agent.run("What do you remember about me?", session=session)
    print(f"Agent: {result}")
    # </multi_turn>


if __name__ == "__main__":
    asyncio.run(main())
