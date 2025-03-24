import os
import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner
from agents import set_default_openai_key
from openai.types.responses import ResponseTextDeltaEvent

load_dotenv()
# setting the openai key for tracing purposes
set_default_openai_key(os.getenv("OPENAI_API_KEY"))

# creating the agent
agent = Agent(name="Assistant", instructions="You are an helpful assistant")

# agent can be run in sync, async or streaming mode

# sync mode
runner = Runner.run_sync(starting_agent=agent, input="What is the capital of France?")

print(runner.final_output)


async def main():
    # async mode
    runner = await Runner.run(
        starting_agent=agent, input="What is the purpose of the universe?"
    )
    print(runner.final_output)

    # streaming mode
    result = Runner.run_streamed(agent, input="Tell me 3 jokes about a chicken")
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(
            event.data, ResponseTextDeltaEvent
        ):
            print(event.data.delta, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
