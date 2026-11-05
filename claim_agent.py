import json
import logging
import re

from mcp import Client
from mcp.types import TextContent

from a2a.helpers import get_message_text, new_text_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    create_agent_card_routes,
    create_jsonrpc_routes,
)
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
)
from starlette.applications import Starlette


claims_skill = AgentSkill(
    id="claim_status",
    name="Claim Status",
    description="Checks the status of a synthetic insurance claim.",
    tags=["healthcare", "claims", "status"],
    examples=["What is the status of CLM-1001?"],
)

agent_card = AgentCard(
    name="Claims Agent",
    description="Retrieves synthetic claim status information.",
    version="0.1.0",
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"],
    capabilities=AgentCapabilities(streaming=False),
    supported_interfaces=[
        AgentInterface(
            url="http://127.0.0.1:8003/",
            protocol_binding="JSONRPC",
            protocol_version="1.0",
        )
    ],
    skills=[claims_skill],
)


class ClaimsExecutor(AgentExecutor):
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        question = get_message_text(context.message)

        # Extract claim IDs such as CLM-1001.
        claim_ids = set(
            re.findall(
                r"\bCLM-\d+\b",
                question.upper(),
            )
        )

        if len(claim_ids) != 1:
            answer = (
                "Please provide one demo claim ID, "
                "for example: What is the status of CLM-1001?"
            )
        else:
            claim_id = next(iter(claim_ids))

            try:
                async with Client(
                    "http://127.0.0.1:8002/mcp"
                ) as client:
                    result = await client.call_tool(
                        "get_claim_status",
                        {"claim_id": claim_id},
                    )

                if result.is_error:
                    answer = "The claim lookup failed. Please try again."
                else:
                    payload = result.structured_content

                    if payload is None:
                        tool_text = "\n".join(
                            block.text
                            for block in result.content
                            if isinstance(block, TextContent)
                        )
                        payload = json.loads(tool_text)

                    if not payload.get("found"):
                        answer = payload["message"]
                    else:
                        answer = (
                            "Synthetic demo claim\n\n"
                            f"Claim ID: {payload['claim_id']}\n"
                            f"Service: {payload['service']}\n"
                            f"Status: {payload['status']}\n"
                            f"Submitted: {payload['submitted_date']}\n"
                            f"Billed amount: {payload['currency']} "
                            f"{payload['billed_amount']:.2f}\n"
                            f"Details: {payload['status_details']}\n\n"
                            "Source: Demo claims records"
                        )

            except Exception:
                logging.exception("Claim lookup failed")
                answer = (
                    "The claims service is currently unavailable. "
                    "Please try again later."
                )

        await event_queue.enqueue_event(
            new_text_message(answer)
        )

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        raise NotImplementedError(
            "Cancellation is not supported by this prototype."
        )


request_handler = DefaultRequestHandler(
    agent_executor=ClaimsExecutor(),
    task_store=InMemoryTaskStore(),
    agent_card=agent_card,
)

routes = [
    *create_agent_card_routes(agent_card),
    *create_jsonrpc_routes(request_handler, "/"),
]

app = Starlette(routes=routes)