import asyncio

from a2a.client import ClientConfig, create_client
from a2a.helpers import new_text_message
from a2a.types import Role, SendMessageRequest


async def main():
    # Discover the agent and create a client.
    client = await create_client(
        agent="http://127.0.0.1:8001",
        client_config=ClientConfig(streaming=False),
    )

    try:
        question = "Does my plan cover hearing aids?"
        print(f"Question: {question}\n")

        # Create an A2A message with the sender role "user".
        message = new_text_message(
            question,
            role=Role.ROLE_USER,
        )

        request = SendMessageRequest(message=message)

        # Send the request and display the response.
        async for response in client.send_message(request):
            print("Agent response:")
            print(response)

    finally:
        # Release the client's network connections.
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())