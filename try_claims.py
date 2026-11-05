import asyncio

from orchestrator import call_agent


async def main():
    answer = await call_agent(
        agent_url="http://127.0.0.1:8003",
        question="What is the status of CLM-1001?",
    )

    print(answer)


if __name__ == "__main__":
    asyncio.run(main())