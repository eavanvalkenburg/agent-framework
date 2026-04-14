# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "agent-framework-core",
#     "agent-framework-devui",
#     "tiktoken>=0.11.0",
#     "azure-identity",
#     "python-dotenv",
# ]
# ///
# Run with: uv run samples/02-agents/devui/advanced_agent.py

# Copyright (c) Microsoft. All rights reserved.

"""Advanced DevUI sample with tiktoken tokenizer, tool result compaction,
and dynamic tools that return realistic data.

Demonstrates:
- TiktokenTokenizer for accurate token counting
- ToolResultCompactionStrategy to keep context lean across turns
- Dynamic weather/time tools with randomized data
"""

import logging
import os
import random
from datetime import datetime, timezone
from typing import Annotated
from zoneinfo import ZoneInfo

import tiktoken
from agent_framework import (
    Agent,
    TokenBudgetComposedStrategy,
    TokenizerProtocol,
    ToolResultCompactionStrategy,
    tool,
)
from agent_framework.azure import AzureOpenAIResponsesClient
from agent_framework.devui import serve
from azure.identity import AzureCliCredential
from dotenv import load_dotenv
from typing_extensions import Any

load_dotenv()
load_dotenv("../../../.env")


# region Tiktoken tokenizer


class TiktokenTokenizer(TokenizerProtocol):
    """Accurate token counter using tiktoken's o200k_base encoding (gpt-4.1+)."""

    def __init__(
        self,
        *,
        encoding_name: str = "o200k_base",
        model_name: str | None = None,
    ) -> None:
        if model_name is not None:
            self._encoding = tiktoken.encoding_for_model(model_name)
        else:
            self._encoding: Any = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text))


# region Tools

_CONDITIONS = ["sunny", "partly cloudy", "cloudy", "rainy", "stormy", "foggy", "snowy", "windy"]

# Common city → timezone mappings for realistic responses
_CITY_TIMEZONES: dict[str, str] = {
    "new york": "America/New_York",
    "los angeles": "America/Los_Angeles",
    "chicago": "America/Chicago",
    "seattle": "America/Los_Angeles",
    "san francisco": "America/Los_Angeles",
    "london": "Europe/London",
    "paris": "Europe/Paris",
    "berlin": "Europe/Berlin",
    "tokyo": "Asia/Tokyo",
    "sydney": "Australia/Sydney",
    "mumbai": "Asia/Kolkata",
    "dubai": "Asia/Dubai",
    "singapore": "Asia/Singapore",
    "amsterdam": "Europe/Amsterdam",
    "rome": "Europe/Rome",
    "madrid": "Europe/Madrid",
}


# NOTE: approval_mode="never_require" is for sample brevity.
@tool(approval_mode="never_require")
def get_weather(
    location: Annotated[str, "The city or location to get weather for."],
) -> str:
    """Get current weather conditions for a location."""
    condition = random.choice(_CONDITIONS)
    temp_c = random.randint(-5, 38)
    temp_f = round(temp_c * 9 / 5 + 32)
    humidity = random.randint(20, 95)
    wind_kmh = random.randint(0, 50)

    return f"Weather in {location}: {condition}, {temp_c}°C ({temp_f}°F), humidity {humidity}%, wind {wind_kmh} km/h."


@tool(approval_mode="never_require")
def get_time(
    location: Annotated[str, "The city or timezone to get time for."],
) -> str:
    """Get the current local time for a city or timezone."""
    # Try to resolve city to timezone
    tz_name = _CITY_TIMEZONES.get(location.lower())
    if not tz_name:
        # Try as direct timezone string
        tz_name = location

    try:
        tz = ZoneInfo(tz_name)
        now = datetime.now(tz=tz)
        return (
            f"Current time in {location}: {now.strftime('%I:%M %p')} ({now.strftime('%Z')}, {now.strftime('%Y-%m-%d')})"
        )
    except (KeyError, ValueError):
        # Fallback to UTC
        now = datetime.now(tz=timezone.utc)
        return (
            f"Could not resolve timezone for '{location}'. "
            f"UTC time: {now.strftime('%I:%M %p')} ({now.strftime('%Y-%m-%d')})"
        )


# region Main


def main() -> None:
    """Launch DevUI with a single advanced weather agent."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    # Create tokenizer
    tokenizer = TiktokenTokenizer()

    # Create compaction: token-aware composite with tool result compaction
    compaction = TokenBudgetComposedStrategy(
        token_budget=1024,
        tokenizer=tokenizer,
        strategies=[
            ToolResultCompactionStrategy(keep_last_tool_call_groups=1),
        ],
    )

    # Create Azure OpenAI client
    client = AzureOpenAIResponsesClient(
        credential=AzureCliCredential(),
        project_endpoint=os.getenv("AZURE_AI_PROJECT_ENDPOINT"),
        deployment_name=os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME"),
    )

    # Create agent with tokenizer and compaction
    weather_agent = Agent(
        name="weather-assistant",
        description="Weather and time assistant with token-aware compaction",
        instructions=(
            "You are a helpful weather and time assistant. Use the available tools to "
            "provide accurate weather information and current local time for any location. "
            "Always be friendly and include relevant details like humidity and wind speed."
        ),
        client=client,
        tools=[get_weather, get_time],
        tokenizer=tokenizer,
        compaction=compaction,
        default_options={"store": False},
    )

    logger.info("Starting DevUI with advanced weather agent")
    logger.info("  - tiktoken tokenizer (o200k_base)")
    logger.info("  - TokenBudgetComposedStrategy (4096 budget)")
    logger.info("    - ToolResultCompactionStrategy (keep_last=1)")
    logger.info("  - Dynamic weather/time tools")

    serve(entities=[weather_agent], port=8090, auto_open=False)


if __name__ == "__main__":
    main()
