"""
Tools lets agent to perform actions that are not part of the LLM's capabilities.

Tools can be of:

- Custom function tools: Python functions
- Hosted tools: Provided by OpenAI. Web Search, File Search, Computer Use are the three supported hosted tools.
- Agent tools: Agents as tools
"""

import os
import asyncio
from dotenv import load_dotenv

from agents import (
    Agent,
    Runner,
    FunctionTool,
    function_tool,
    WebSearchTool,
    RunContextWrapper,
)
from pydantic import BaseModel
from agents import set_default_openai_key
from typing import Any

load_dotenv()
set_default_openai_key(os.getenv("OPENAI_API_KEY"))


class Weather(BaseModel):
    city: str
    temperature_range: str
    conditions: str


# Function as tool.
@function_tool
def get_weather(ctx: RunContextWrapper[Any], city: str) -> Weather:
    print("[debug] get_weather called")
    print(ctx)
    return Weather(city=city, temperature_range="20-25", conditions="sunny")


# agent as tool
spanish_agent = Agent(
    name="Spanish agent",
    instructions="You translate the user's message to Spanish",
)

agent = Agent(
    name="Assitant",
    instructions="""You are a helpful assistant. You have access to the following tools:
    - get_weather: Get the weather in a given city
    - web_search: Search the web for information
    - translate_to_spanish: Translate the user's message to Spanish

    Depending on the user's question, you should use the appropriate tool to answer the question.
    """,
    model="gpt-4o-mini",
    tools=[
        get_weather,
        WebSearchTool(),  # Hosted tool
        spanish_agent.as_tool(
            tool_name="translate_to_spanish",
            tool_description="Translate the user's message to Spanish",
        ),
    ],
)

for tool in agent.tools:
    if isinstance(tool, FunctionTool):
        print(tool.name)
        print(tool.description)
        print("----")
    else:
        print(tool)
        print(type(tool))
        print("----")


async def main():
    input_text = input("Ask you question: ")
    runner = await Runner.run(starting_agent=agent, input=input_text)
    print(runner.final_output)


if __name__ == "__main__":
    asyncio.run(main())
