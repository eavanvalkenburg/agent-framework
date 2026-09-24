# Copyright (c) Microsoft. All rights reserved.

"""Host a native workflow over Invocations with application-defined JSON input."""

import os
from dataclasses import dataclass
from typing import Any

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
from agent_framework_foundry_hosting import InvocationsHostServer, WorkflowTurn
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from starlette.requests import Request

load_dotenv()


@dataclass(frozen=True)
class Ticket:
    ticket_id: str
    question: str


class TicketStart(Executor):
    def __init__(self) -> None:
        super().__init__(id="ticket_start")

    @handler
    async def run(self, ticket: Ticket, ctx: WorkflowContext[AgentExecutorRequest, str]) -> None:
        """Persist application state before invoking the responder."""
        ctx.set_state("last_ticket_id", ticket.ticket_id)
        await ctx.send_message(
            AgentExecutorRequest(
                messages=[
                    Message(role="user", contents=[Content.from_text(f"Ticket {ticket.ticket_id}: {ticket.question}")])
                ]
            )
        )


def build_workflow(request: Request) -> Workflow:
    """Build a new graph for this Foundry Invocations session turn."""
    agent = Agent(
        client=FoundryChatClient(
            project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
            model=os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"],
            credential=DefaultAzureCredential(),
        ),
        name="ticket_responder",
        instructions="Answer the ticket concisely.",
        default_options={"store": False},
    )
    start = TicketStart()
    responder = AgentExecutor(agent, id="ticket_responder")
    return (
        WorkflowBuilder(name="ticket-workflow", start_executor=start, output_from=[responder])
        .add_edge(start, responder)
        .build()
    )


async def parse_request(request: Request) -> WorkflowTurn[Ticket]:
    """Validate the webhook's payload and turn it into a typed workflow input."""
    payload: Any = await request.json()
    if not isinstance(payload, dict):
        raise ValueError("The request must be a JSON object.")
    ticket_id = payload.get("ticket_id")
    question = payload.get("question")
    if not isinstance(ticket_id, str) or not ticket_id or not isinstance(question, str) or not question:
        raise ValueError("ticket_id and question must be non-empty strings.")
    stream = payload.get("stream", False)
    if not isinstance(stream, bool):
        raise ValueError("stream must be a boolean.")
    return WorkflowTurn(input=Ticket(ticket_id=ticket_id, question=question), stream=stream)


def main() -> None:
    InvocationsHostServer(workflow=build_workflow, parse_request=parse_request).run()


if __name__ == "__main__":
    main()
