import asyncio
from agents import Agent, Runner, set_default_openai_key, RunContextWrapper
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from typing import Literal

load_dotenv()

set_default_openai_key(os.getenv("OPENAI_API_KEY"))


class Context(BaseModel):
    style: Literal["haiku", "pirate", "robot"]


def custom_instructions(run_context: RunContextWrapper[Context], agent: Agent[Context]):
    context = run_context.context
    if context.style == "haiku":
        return "Respond in haiku style"
    elif context.style == "pirate":
        return "Respond in pirate style"
    else:
        return "Respond in robot style"


agent = Agent[Context](
    name="Style Agent",
    instructions=custom_instructions,
    model="gpt-4o-mini",
)


async def main():
    style = input("Enter a style (haiku, pirate, robot): ")
    context = Context(style=style)
    runner = await Runner.run(
        starting_agent=agent,
        input="Write a poem about carrot",
        context=context,
    )
    print(runner.final_output)


if __name__ == "__main__":
    asyncio.run(main())
