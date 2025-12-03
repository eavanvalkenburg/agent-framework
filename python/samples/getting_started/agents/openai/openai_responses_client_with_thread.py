# Copyright (c) Microsoft. All rights reserved.

import asyncio
from random import randint
from typing import Annotated

from agent_framework import ChatAgent
from agent_framework.openai import OpenAIResponsesClient
from pydantic import Field

"""
OpenAI Responses Client with Thread Management Example

This sample demonstrates thread management with OpenAI Responses Client, showing
persistent conversation context and simplified response handling.
"""


def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """Get the weather for a given location."""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


async def example_with_no_thread_creation() -> None:
    """Example showing automatic thread creation."""
    print("=== Automatic Thread Creation Example ===")

    agent = ChatAgent(
        chat_client=OpenAIResponsesClient(),
        instructions="You are a helpful weather agent.",
        tools=get_weather,
    )

    # First conversation - no thread provided, will be created automatically
    query1 = "What's the weather like in Seattle?"
    print(f"User: {query1}")
    result1 = await agent.run(query1)
    print(f"Agent: {result1.text}")

    # Second conversation - still no thread provided, will create another new thread
    query2 = "What was the last city I asked about?"
    print(f"\nUser: {query2}")
    result2 = await agent.run(query2)
    print(f"Agent: {result2.text}")
    print("Note: Each call creates a separate thread, so the agent doesn't remember previous context.\n")


async def example_with_thread_persistence_in_memory() -> None:
    """
    Example showing thread persistence across multiple conversations.
    In this example, messages are stored in-memory.
    """
    print("=== Thread Persistence Example (In-Memory) ===")

    agent = ChatAgent(
        chat_client=OpenAIResponsesClient(),
        instructions="You are a helpful weather agent.",
        tools=get_weather,
    )

    # Create a new thread that will be reused
    thread = await agent.get_local_thread()

    # First conversation
    query1 = "What's the weather like in Tokyo?"
    print(f"User: {query1}")
    result1 = await agent.run(query1, thread=thread)
    print(f"Agent: {result1.text}")

    # Second conversation using the same thread - maintains context
    query2 = "How about London?"
    print(f"\nUser: {query2}")
    result2 = await agent.run(query2, thread=thread)
    print(f"Agent: {result2.text}")

    # Third conversation - agent should remember both previous cities
    query3 = "Which of the cities I asked about has better weather?"
    print(f"\nUser: {query3}")
    result3 = await agent.run(query3, thread=thread)
    print(f"Agent: {result3.text}")
    print("Note: The agent remembers context from previous messages in the same thread.\n")

    print("The conversation is fully available in the thread:")
    print(thread.to_json(indent=2)[:500] + "\n...")  # Print first 500 chars for brevity


async def example_with_existing_thread_id() -> None:
    """
    Example showing how to work with an existing thread ID from the service.
    In this example, messages are stored on the server using OpenAI conversation state.
    """
    print("=== Existing Thread ID Example ===")

    # First, create a conversation and capture the thread ID
    existing_thread_id = None

    agent = ChatAgent(
        chat_client=OpenAIResponsesClient(),
        instructions="You are a helpful weather agent.",
        tools=get_weather,
    )

    # Start a conversation and get the thread ID
    thread = await agent.get_hosted_thread()

    query1 = "What's the weather in Paris?"
    print(f"User: {query1}")
    result1 = await agent.run(query1, thread=thread)
    print(f"Agent: {result1.text}")

    # The thread ID is set after the first response
    existing_thread_id = thread.hosted_thread_id
    print(f"Thread ID: {existing_thread_id}")

    if existing_thread_id:
        print("\n--- Continuing with the same thread ID in a new agent instance ---")

        agent = ChatAgent(
            chat_client=OpenAIResponsesClient(),
            instructions="You are a helpful weather agent.",
            tools=get_weather,
        )

        # Create a thread with the existing ID
        thread = await agent.get_hosted_thread(hosted_thread_id=existing_thread_id)

        query2 = "What was the last city I asked about?"
        print(f"User: {query2}")
        result2 = await agent.run(query2, thread=thread)
        print(f"Agent: {result2.text}")
        print("Note: The agent continues the conversation from the previous thread by using thread ID.\n")

        print("The conversation is fully serializable:")
        print(thread.to_json(indent=2))


async def main() -> None:
    print("=== OpenAI Response Client Agent Thread Management Examples ===\n")

    await example_with_thread_persistence_in_memory()
    await example_with_existing_thread_id()
    await example_with_no_thread_creation()


if __name__ == "__main__":
    asyncio.run(main())


"""
Example Output:
=== OpenAI Response Client Agent Thread Management Examples ===

=== Thread Persistence Example (In-Memory) ===
User: What's the weather like in Tokyo?
Agent: Sunny in Tokyo today with a high of 26°C. Want the current temperature, hourly forecast, or a 7-day outlook? I can pull those up.

User: How about London?
Agent: Stormy in London today with a high of 25°C. Want the current temperature, hourly forecast, or a 7-day outlook? I can pull those up.

User: Which of the cities I asked about has better weather?
Agent: Tokyo currently has better weather: sunny and 26°C, while London is stormy at around 25°C. Of course, "better" depends on your preferences (clear skies vs rain). Want me to pull the current temperatures and hourly forecasts for both to compare more precisely?
Note: The agent remembers context from previous messages in the same thread.

The conversation is fully available in the thread:
{
  "type": "local_thread",
  "context_states": {},
  "additional_properties": {},
  "messages": [
    {
      "type": "chat_message",
      "role": {
        "type": "role",
        "value": "user"
      },
      "contents": [
        {
          "type": "text",
          "text": "What's the weather like in Tokyo?"
        }
      ],
      "additional_properties": {}
    },
    {
      "type": "chat_message",
      "role": {
        "type": "role",
        "value": "assistant"
      },
      "c
...
=== Existing Thread ID Example ===
User: What's the weather in Paris?
Agent: Paris is currently stormy with a high of 23°C. Would you like a more detailed breakdown (wind, humidity, precipitation chances) or the forecast for the next few days? And did you mean Paris, France?
Thread ID: resp_0ef4cb128886c614006930456bd4448194876cff912c9ea9c7

--- Continuing with the same thread ID in a new agent instance ---
User: What was the last city I asked about?
Agent: Paris.
Note: The agent continues the conversation from the previous thread by using thread ID.

The conversation is fully serializable:
{
  "type": "hosted_thread",
  "context_states": {},
  "additional_properties": {},
  "hosted_thread_id": "resp_0ef4cb128886c614006930457451cc8194ac2463b95bb94542"
}
=== Automatic Thread Creation Example ===
User: What's the weather like in Seattle?
Agent: The weather in Seattle is cloudy with a high around 16°C (about 62°F). Want the current conditions, wind/humidity details, or an hourly/daily forecast (in Fahrenheit if you prefer)?

User: What was the last city I asked about?
Agent: I don’t have a record of any city in this chat yet. If you tell me the city you’re referring to (or remind me from a previous session), I can help. Want me to fetch the current weather for a city now?
Note: Each call creates a separate thread, so the agent doesn't remember previous context.

"""
