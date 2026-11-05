import asyncio

from orchestrator import call_agent


async def main():
    answer = await call_agent(
        agent_url="http://127.0.0.1:8004",
        question="Find a physiotherapist in Chicago.",
    )

    print(answer)


if __name__ == "__main__":
    asyncio.run(main())