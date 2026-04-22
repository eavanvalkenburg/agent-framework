# Copyright (c) Microsoft. All rights reserved.

"""Shows how to run CodexAgent with streaming and session reuse.

Requires a working Codex CLI login (`codex login`) or `CODEX_API_KEY`.
Optionally set `CODEX_AGENT_MODEL` to choose a specific model.
"""

import asyncio

from agent_framework_codex import CodexAgent


async def main() -> None:
    """Run the getting-started Codex sample."""
    async with CodexAgent(instructions="You are a concise coding assistant.") as agent:
        # 1. Ask a one-off question and print the complete response.
        print("=== One-off response ===")
        first_prompt = "Explain what a Python context manager does in two sentences."
        print(f"User: {first_prompt}")
        first_response = await agent.run(first_prompt)
        print(f"Assistant: {first_response.text}\n")

        # 2. Reuse a framework session so Codex continues the same thread.
        print("=== Session reuse ===")
        session = agent.create_session()
        await agent.run("Remember that my project uses pytest for unit tests.", session=session)

        # 3. Stream the follow-up response from the same Codex thread.
        follow_up = "What test framework am I using?"
        print(f"User: {follow_up}")
        print("Assistant: ", end="", flush=True)
        async for update in agent.run(follow_up, session=session, stream=True):
            if update.text:
                print(update.text, end="", flush=True)
        print()


if __name__ == "__main__":
    asyncio.run(main())

"""
Sample output:
=== One-off response ===
User: Explain what a Python context manager does in two sentences.
Assistant: A context manager wraps setup and cleanup logic around a block of code.
It is commonly used with `with` so resources such as files or locks are released automatically.

=== Session reuse ===
User: What test framework am I using?
Assistant: You're using pytest for unit tests.
"""
