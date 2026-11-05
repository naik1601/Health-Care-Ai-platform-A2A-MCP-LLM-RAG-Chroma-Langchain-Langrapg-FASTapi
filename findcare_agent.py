import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from mcp import Client
from mcp.types import TextContent
from openai import AsyncOpenAI
from pydantic import BaseModel

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


load_dotenv(Path(__file__).resolve().parent / ".env")


class ProviderSearchInput(BaseModel):
    specialty: str
    city: str


async def extract_search_input(question: str) -> ProviderSearchInput:
    async with AsyncOpenAI(
        timeout=30.0,
        max_retries=1,
    ) as client:
        response = await client.responses.parse(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            instructions=(
                "Extract the provider specialty and city explicitly "
                "requested by the user. "
                "Do not invent missing values or infer the user's location. "
                "Use an empty string for a missing value. "
                "Normalize physical therapist, physiotherapist, "
                "and physical therapy to physiotherapy. "
                "Normalize general practitioner to primary care. "
                "Do not answer the user's question."
            ),
            input=question,
            text_format=ProviderSearchInput,
            max_output_tokens=200,
            store=False,
        )

    if response.output_parsed is None:
        raise RuntimeError("Could not extract provider search inputs.")

    return response.output_parsed


async def find_providers(question: str) -> str:
    search = await extract_search_input(question)

    missing = []
    if not search.specialty.strip():
        missing.append("specialty")
    if not search.city.strip():
        missing.append("city")

    if missing:
        return (
            f"Please specify the {' and '.join(missing)}. "
            "For example: Find a physiotherapist in Chicago."
        )

    async with Client("http://127.0.0.1:8002/mcp") as client:
        result = await client.call_tool(
            "search_providers",
            {
                "specialty": search.specialty,
                "city": search.city,
                "plan_id": "DEMO-PLAN-001",
            },
        )

    if result.is_error:
        return "Provider search failed. Please try again."

    payload = result.structured_content

    if payload is None:
        tool_text = "\n".join(
            block.text
            for block in result.content
            if isinstance(block, TextContent)
        )
        payload = json.loads(tool_text)

    providers = payload.get("providers", [])

    if not providers:
        return (
            "No matching providers were found in the synthetic "
            "directory. This does not mean none exist in real life."
        )

    lines = [
        "Synthetic provider search",
        f"Specialty: {payload['specialty']}",
        f"City: {payload['city']}",
        f"Demo plan: {payload['plan_id']}",
        "",
    ]

    for provider in providers:
        accepting = (
            "Yes"
            if provider["accepting_new_patients"]
            else "No"
        )

        lines.extend(
            [
                f"{provider['name']} ({provider['provider_id']})",
                f"Accepting new patients: {accepting}",
                "",
            ]
        )

    lines.extend(
        [
            "Source: Synthetic provider directory",
            "Fictional records for testing only.",
            "Filters applied: specialty, city, and demo plan.",
            "Appointment availability is not provided.",
        ]
    )

    return "\n".join(lines)


class FindCareExecutor(AgentExecutor):
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        question = get_message_text(context.message)

        try:
            answer = await find_providers(question)
        except Exception:
            logging.exception("FindCare request failed")
            answer = (
                "The provider search service is currently unavailable. "
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


agent_card = AgentCard(
    name="FindCare Agent",
    description="Searches a fictional provider directory.",
    version="0.1.0",
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"],
    capabilities=AgentCapabilities(streaming=False),
    supported_interfaces=[
        AgentInterface(
            url="http://127.0.0.1:8004/",
            protocol_binding="JSONRPC",
            protocol_version="1.0",
        )
    ],
    skills=[
        AgentSkill(
            id="find_providers",
            name="Find Providers",
            description="Finds fictional providers by specialty and city.",
            tags=["healthcare", "providers", "directory"],
            examples=["Find a physiotherapist in Chicago."],
        )
    ],
)

request_handler = DefaultRequestHandler(
    agent_executor=FindCareExecutor(),
    task_store=InMemoryTaskStore(),
    agent_card=agent_card,
)

app = Starlette(
    routes=[
        *create_agent_card_routes(agent_card),
        *create_jsonrpc_routes(request_handler, "/"),
    ]
)