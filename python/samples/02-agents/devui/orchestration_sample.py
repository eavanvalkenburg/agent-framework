# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "agent-framework-core",
#     "agent-framework-devui",
#     "agent-framework-orchestrations",
#     "azure-identity",
#     "python-dotenv",
# ]
# ///
# Run with: uv run samples/02-agents/devui/orchestration_sample.py

# Copyright (c) Microsoft. All rights reserved.

"""DevUI sample with a sequential orchestration workflow using two agents.

Demonstrates:
- Writer agent that drafts content
- Reviewer agent that critiques and improves it
- Sequential workflow chaining writer → reviewer
- All three entities served together for comparison
"""

import logging
import os

from agent_framework import Agent
from agent_framework.azure import AzureOpenAIResponsesClient
from agent_framework.devui import serve
from agent_framework.orchestrations import SequentialBuilder
from azure.identity import AzureCliCredential
from dotenv import load_dotenv

load_dotenv()
load_dotenv("../../../.env")


def main() -> None:
    """Launch DevUI with writer, reviewer, and a sequential workflow."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    client = AzureOpenAIResponsesClient(
        credential=AzureCliCredential(),
        project_endpoint=os.getenv("AZURE_AI_PROJECT_ENDPOINT"),
        deployment_name=os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME"),
    )

    writer = Agent(
        name="writer",
        description="Writes a short paragraph on a given topic",
        instructions=(
            "You are a concise writer. Write exactly one short paragraph "
            "(2-3 sentences) on the given topic. Be creative but brief."
        ),
        client=client,
    )

    reviewer = Agent(
        name="reviewer",
        description="Reviews and improves written content",
        instructions=(
            "You are a content reviewer. Read the text provided and give "
            "a brief critique (1-2 sentences) followed by an improved version. "
            "Keep it concise."
        ),
        client=client,
    )

    workflow = SequentialBuilder(participants=[writer, reviewer]).build()

    logger.info("Starting DevUI with orchestration workflow")
    logger.info("  Entities:")
    logger.info("  - writer (agent)")
    logger.info("  - reviewer (agent)")
    logger.info("  - Writer-Reviewer Pipeline (workflow)")

    serve(entities=[writer, reviewer, workflow], port=8090, auto_open=False)


if __name__ == "__main__":
    main()
