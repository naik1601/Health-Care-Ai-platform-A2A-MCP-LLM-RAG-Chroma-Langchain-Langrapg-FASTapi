import asyncio

from mcp import Client
from mcp.types import TextContent


async def main():
    async with Client("http://127.0.0.1:8002/mcp") as client:
        tools = await client.list_tools()

        print("Available tools:")
        for tool in tools.tools:
            print(f"- {tool.name}")

        result = await client.call_tool(
            "search_plan_documents",
            {
                "plan_id": "DEMO-PLAN-001",
                "question": (
                    "Do I need approval before physical therapy?"
                ),
            },
        )

        print("\nTool error:", result.is_error)
        print("\nTool response:")

        for block in result.content:
            if isinstance(block, TextContent):
                print(block.text)


if __name__ == "__main__":
    asyncio.run(main())