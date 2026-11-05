import asyncio

from mcp import Client
from mcp.types import TextContent


async def main():
    async with Client("http://127.0.0.1:8002/mcp") as client:
        result = await client.call_tool(
            "search_providers",
            {
                "specialty": "physical therapy",
                "city": "Chicago",
                "plan_id": "DEMO-PLAN-001",
            },
        )

        print("Tool error:", result.is_error)

        for block in result.content:
            if isinstance(block, TextContent):
                print(block.text)


if __name__ == "__main__":
    asyncio.run(main())