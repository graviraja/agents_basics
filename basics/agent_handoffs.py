"""
Handoffs are a way to pass the baton from one agent to another.

"""

from agents import (
    Agent,
    Runner,
    set_default_openai_key,
    function_tool,
    RunContextWrapper,
    handoff,
)
from dotenv import load_dotenv
import os
from pydantic import BaseModel
import random
import asyncio

load_dotenv()

set_default_openai_key(os.getenv("OPENAI_API_KEY"))


async def handoff_callback(ctx: RunContextWrapper):
    print("\n🔄 Handoff just happened")


@function_tool
def random_number(max_number: int) -> int:
    """Generate a random number"""
    return random.randint(0, max_number)


@function_tool
def add_one(x: int) -> int:
    """Add one to the user's number"""
    return x + 1


@function_tool
def multiply_by_two(x: int) -> int:
    """Simple multiplication by two"""
    return x * 2


class FinalResult(BaseModel):
    result: int


multiply_agent = Agent(
    name="Multiply Agent",
    instructions="You multiply the user's message by 2",
    tools=[multiply_by_two],
    output_type=FinalResult,
)

addition_agent = Agent(
    name="Addition Agent",
    instructions="You add 1 to the user's number",
    tools=[add_one],
    output_type=FinalResult,
)

# When the other agents are configured as handoffs, the main agent will not have access to the outputs and formats it before returning it.
agents_as_handoffs = Agent(
    name="Agent Handoffs",
    instructions="Generate a random number. If it's even, hand off to the addition agent. If it's odd, hand off to the multiply agent.",
    tools=[random_number],
    handoffs=[multiply_agent, handoff(addition_agent, on_handoff=handoff_callback)],
)
# For the input, 43 we get the output as {"result": 86}

# When the other agents are configured as tools, the main agent will still have the access to the outputs and formats it before returning it.
agents_as_tools = Agent(
    name="Agent Tools",
    instructions="Generate a random number. If it's even, hand off to the addition agent. If it's odd, hand off to the multiply agent.",
    tools=[
        random_number,
        multiply_agent.as_tool(
            tool_name="multiply_agent", tool_description="Multiply the user's number"
        ),
        addition_agent.as_tool(
            tool_name="addition_agent",
            tool_description="Add the number to user's number",
        ),
    ],
)
# For the input 43, we get the output as: The random number generated was 43, which is odd. The result after processing with the multiply agent is 86.
# The extra summary is due to the fact that the main agent has access to the outputs and formats it before returning it.


# The handoff agents can be configured to return the control back to the main agent after the execution of the handoff agent.
multiply_agent_with_handoff = Agent(
    name="Multiply Agent",
    instructions="You multiply the user's message by 2. Once you have the result, hand off the result to the main agent.",
    tools=[multiply_by_two],
    output_type=FinalResult,
)

addition_agent_with_handoff = Agent(
    name="Addition Agent",
    instructions="You add 1 to the user's number. Once you have the result, hand off the result to the main agent.",
    tools=[add_one],
    output_type=FinalResult,
)

main_agent = Agent(
    name="Main Agent",
    instructions="Generate a random number. If it's even, hand off to the addition agent. If it's odd, hand off to the multiply agent.",
    tools=[random_number],
)
main_agent.handoffs = [
    handoff(multiply_agent_with_handoff, on_handoff=handoff_callback),
    handoff(addition_agent_with_handoff, on_handoff=handoff_callback),
]
# Each handoff agent is configured to return the control back to the main agent after the execution of the handoff agent.
multiply_agent_with_handoff.handoffs = [
    handoff(main_agent, on_handoff=handoff_callback),
]
addition_agent_with_handoff.handoffs = [
    handoff(main_agent, on_handoff=handoff_callback),
]


async def main():
    user_input = input("Enter a max number: ")
    handoffs_runner = await Runner.run(
        starting_agent=agents_as_handoffs,
        input=f"Generate a random number between 0 and {user_input}",
    )
    print(handoffs_runner.final_output)

    tools_runner = await Runner.run(
        starting_agent=agents_as_tools,
        input=f"Generate a random number between 0 and {user_input}",
    )
    print(tools_runner.final_output)

    handoff_runner = await Runner.run(
        starting_agent=main_agent,
        input=f"Generate a random number between 0 and {user_input}",
    )
    print(handoff_runner.final_output)


if __name__ == "__main__":
    asyncio.run(main())
