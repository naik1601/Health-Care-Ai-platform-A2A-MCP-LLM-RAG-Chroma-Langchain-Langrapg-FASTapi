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
from mcp import Client
from mcp.types import TextContent
import logging
import json
from answer_generator import generate_answer

# Describe one capability offered by this agent.
benefits_skill = AgentSkill(
    id="benefits_questions",
    name="Benefits Questions",
    description="Handles questions about insurance plan benefits.",
    tags=["healthcare", "insurance", "benefits"],
    examples=["Does my plan cover physiotherapy?"],
)


# Publish the agent's identity and connection details.
agent_card = AgentCard(
    name="Benefits Agent",
    description="A local prototype for insurance benefits questions.",
    version="0.1.0",
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"],
    capabilities=AgentCapabilities(streaming=False),
    supported_interfaces=[
        AgentInterface(
            url="http://127.0.0.1:8001/",
            protocol_binding="JSONRPC",
            protocol_version="1.0",
        )
    ],
    skills=[benefits_skill],
)


# Handle messages delivered through A2A.
class BenefitsExecutor(AgentExecutor):
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        question = get_message_text(context.message)

        if not question.strip():
            answer = "Please enter a benefits question."
        else:
            try:
                async with Client(
                    "http://127.0.0.1:8002/mcp"
                ) as client:
                    result = await client.call_tool(
                        "search_plan_documents",
                        {
                            "plan_id": "DEMO-PLAN-001",
                            "question": question,
                        },
                    )

                if result.is_error:
                    answer = "Document search failed. Please try again."
                else:
                    # Prefer MCP's structured result.
                    payload = result.structured_content

                    # Fall back to parsing the tool's JSON text.
                    if payload is None:
                        tool_text = "\n".join(
                            block.text
                            for block in result.content
                            if isinstance(block, TextContent)
                        )
                        payload = json.loads(tool_text)

                    matches = payload.get("matches", [])

                    # Use retrieved evidence to generate an answer.
                    answer = await generate_answer(
                        question=question,
                        matches=matches,
                    )

            except Exception:
                logging.exception("Benefits request failed")
                answer = (
                    "The benefits assistant is currently unavailable. "
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


# Connect our executor to the SDK's protocol handler.
request_handler = DefaultRequestHandler(
    agent_executor=BenefitsExecutor(),
    task_store=InMemoryTaskStore(),
    agent_card=agent_card,
)


# Create discovery and messaging endpoints.
routes = [
    *create_agent_card_routes(agent_card),
    *create_jsonrpc_routes(request_handler, "/"),
]

app = Starlette(routes=routes)