# Copyright (c) Microsoft. All rights reserved.

"""Host a native workflow with a typed Responses parser and request-aware agent defaults."""

import os
from dataclasses import dataclass

from agent_framework import (
    Agent,
    AgentExecutor,
    AgentExecutorRequest,
    Content,
    Executor,
    Message,
    Workflow,
    WorkflowBuilder,
    WorkflowContext,
    handler,
)
from agent_framework.foundry import FoundryChatClient
from agent_framework.openai import OpenAIChatOptions
from agent_framework_foundry_hosting import HostedResponseRequest, ResponsesHostServer, WorkflowTurn
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class SloganRequest:
    topic: str
    style: str


class StartExecutor(Executor):
    def __init__(self) -> None:
        super().__init__(id="start")

    @handler
    async def run(self, request: SloganRequest, ctx: WorkflowContext[AgentExecutorRequest, str]) -> None:
        """Record application state and send the typed request to the writer."""
        ctx.set_state("slogan_style", request.style)
        await ctx.send_message(
            AgentExecutorRequest(
                messages=[
                    Message(
                        role="user",
                        contents=[Content.from_text(f"Create a {request.style} slogan for {request.topic}")],
                    )
                ],
            )
        )


def build_workflow(request: HostedResponseRequest) -> Workflow:
    """Build new executors and agent defaults for this hosted request."""
    client = FoundryChatClient(
        project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
        model=os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"],
        credential=DefaultAzureCredential(),
    )

    writer_options = OpenAIChatOptions(store=False)
    if (max_tokens := request.options.get("max_tokens")) is not None:
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer.")
        writer_options["max_tokens"] = max_tokens

    writer = Agent(
        client=client,
        name="writer",
        instructions="Write one short slogan for the topic and style you receive.",
        default_options=writer_options,
    )
    legal = Agent(
        client=client,
        name="legal_reviewer",
        instructions="Check the proposed slogan for misleading legal claims.",
        default_options={"store": False},
    )
    formatter = Agent(
        client=client,
        name="formatter",
        instructions="Format the final slogan in a playful retro style.",
        default_options={"store": False},
    )

    writer_executor = AgentExecutor(writer, id="writer", context_mode="last_agent")
    legal_executor = AgentExecutor(legal, id="legal_reviewer", context_mode="last_agent")
    formatter_executor = AgentExecutor(formatter, id="formatter", context_mode="last_agent")
    start = StartExecutor()
    return (
        WorkflowBuilder(
            name="slogan-workflow",
            start_executor=start,
            output_from=[formatter_executor],
        )
        .add_edge(start, writer_executor)
        .add_edge(writer_executor, legal_executor)
        .add_edge(legal_executor, formatter_executor)
        .build()
    )


async def parse_response(request: HostedResponseRequest) -> WorkflowTurn[SloganRequest]:
    """Map only this turn's Responses input into the workflow's start type."""
    topic = await request.get_input_text()
    if not topic:
        raise ValueError("A slogan topic is required.")
    style = request.options.get("slogan_style", "retro")
    if not isinstance(style, str) or not style:
        raise ValueError("slogan_style must be a non-empty string.")
    return WorkflowTurn(input=SloganRequest(topic=topic, style=style))


def main() -> None:
    ResponsesHostServer(workflow=build_workflow, parse_response=parse_response).run()


if __name__ == "__main__":
    main()
