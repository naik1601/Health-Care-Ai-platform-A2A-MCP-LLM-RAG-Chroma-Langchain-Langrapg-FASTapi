import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI


# Read .env from this script's folder.
load_dotenv(Path(__file__).resolve().parent / ".env")


async def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing from .env")

    async with AsyncOpenAI(
        timeout=30.0,
        max_retries=1,
    ) as client:
        response = await client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            instructions="Respond briefly.",
            input="Say: Healthcare AI connection successful.",
            max_output_tokens=100,
            store=False,
        )

        print(response.output_text)


if __name__ == "__main__":
    asyncio.run(main())