# Copyright (c) Microsoft. All rights reserved.

"""Resume a native workflow after a caller approves a nested agent tool call."""

import os

from agent_framework import Agent, AgentExecutor, Workflow, WorkflowBuilder, tool
from agent_framework.foundry import FoundryChatClient
from agent_framework_foundry_hosting import HostedResponseRequest, ResponsesHostServer, WorkflowTurn
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()


@tool(approval_mode="always_require")
def publish_summary(summary: str) -> str:
    """Simulate publishing an approved summary without making external changes."""
    return f"Approved summary: {summary}"


def build_workflow(request: HostedResponseRequest) -> Workflow:
    """Build a fresh agent and workflow for each request or checkpoint restoration."""
    client = FoundryChatClient(
        project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
        model=os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"],
        credential=DefaultAzureCredential(),
    )
    writer = Agent(
        client=client,
        name="summary_writer",
        instructions="Write a short summary, then always call publish_summary with it.",
        tools=[publish_summary],
        default_options={"store": False},
    )
    start = AgentExecutor(writer, id="summary_writer")
    return WorkflowBuilder(name="approved-summary", start_executor=start, output_from=[start]).build()


async def parse_response(request: HostedResponseRequest) -> WorkflowTurn[str]:
    """Map new input or an approved/denied reply to the appropriate workflow turn."""
    pending_responses = await request.get_workflow_responses()
    if pending_responses:
        return WorkflowTurn(responses=pending_responses)
    prompt = await request.get_input_text()
    if not prompt:
        raise ValueError("A summary topic or an approval response is required.")
    return WorkflowTurn(input=prompt)


def main() -> None:
    ResponsesHostServer(workflow=build_workflow, parse_response=parse_response).run()


if __name__ == "__main__":
    main()
