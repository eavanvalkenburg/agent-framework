# Copyright (c) Microsoft. All rights reserved.

import asyncio
from typing import TYPE_CHECKING, Any

from agent_framework import (
    Agent,
    AgentResponse,
    DynamicToolPolicy,
    Message,
    make_define_tool,
)
from agent_framework.openai import OpenAIChatClient
from agent_framework_monty import MontySandboxToolCompiler
from dotenv import load_dotenv

if TYPE_CHECKING:
    from agent_framework import SupportsAgentRun

# Load environment variables from .env file
load_dotenv()

"""
LLM-defined Dynamic Tools (self-extending agent) Example

This sample builds on progressive tool exposure (see ``dynamic_tool_exposure.py``)
and takes the next step: it lets the *model define a brand-new tool at runtime* —
a name, a description, a JSON-schema for the parameters, and a Python body — and
then exposes that tool to itself through ``FunctionInvocationContext.add_tools``.

Security model (see ``docs/decisions/0027-llm-defined-dynamic-tools.md``):

- The feature is **default-deny**. Nothing happens unless you pass a
  ``DynamicToolPolicy(enabled=True)``.
- Core never executes model-authored code itself. The body runs only inside a
  sandbox supplied by a ``ToolCodeCompiler``. Here we use
  ``MontySandboxToolCompiler`` from ``agent-framework-monty``, which blocks
  filesystem/network access and applies resource limits.
- Both *defining* a tool and *invoking* a defined tool default to
  ``approval_mode="always_require"``, so a human approves each step. This sample
  keeps those defaults and approves interactively.

This is an *extra* pattern. If your goal is general code execution, prefer the
provider-driven CodeAct setup for Monty / Hyperlight in
``../context_providers/code_act/`` instead.
"""


async def handle_approvals(query: str, agent: "SupportsAgentRun") -> AgentResponse:
    """Drive the agent, approving each ``define_tool`` / dynamic-tool call."""
    result = await agent.run(query)
    while len(result.user_input_requests) > 0:
        new_inputs: list[Any] = [query]
        for user_input_needed in result.user_input_requests:
            print(
                f"\nApproval request from {agent.name}:"
                f"\n  Tool: {user_input_needed.function_call.name}"
                f"\n  Arguments: {user_input_needed.function_call.arguments}"
            )
            new_inputs.append(Message("assistant", [user_input_needed]))
            user_approval = await asyncio.to_thread(input, "\nApprove? (y/n): ")
            new_inputs.append(
                Message("user", [user_input_needed.to_function_approval_response(user_approval.lower() == "y")])
            )
        result = await agent.run(new_inputs)
    return result


async def main() -> None:
    # The compiler runs every model-authored body in the Monty sandbox. By default
    # it exposes no host tools to generated bodies (no ambient authority).
    compiler = MontySandboxToolCompiler()

    # Default-deny: you must opt in. Approvals on define + invoke stay on by default.
    define_tool = make_define_tool(compiler=compiler, policy=DynamicToolPolicy(enabled=True))

    agent = Agent(
        client=OpenAIChatClient(),
        name="SelfExtendingAgent",
        instructions=(
            "You can extend yourself. If you lack a capability, call define_tool to create "
            "a new tool (name, description, JSON-schema parameters, and a Python body that "
            "returns a JSON-serializable value), then call that new tool to answer. The body "
            "runs in a sandbox: no imports, no file or network access. Keep bodies small."
        ),
        tools=[define_tool],
    )

    # The agent starts with only ``define_tool``. To answer, it must first define a tool
    # (e.g. one that reverses a string) and then call it on the next iteration.
    result = await handle_approvals("Reverse the string 'agentframework' for me.", agent)
    print(f"\nAgent: {result}")


if __name__ == "__main__":
    asyncio.run(main())
