from typing import TypedDict
import re
import httpx
from langgraph.graph import END, START, StateGraph

from a2a.client import ClientConfig, create_client
from a2a.helpers import get_message_text, new_text_message
from a2a.types import Role, SendMessageRequest

import asyncio
import logging

from intent_router import detect_intents
# Information passed through the graph.
class HealthcareState(TypedDict):
    question: str
    answer: str
    benefits_question: str
    claims_question: str
    findcare_question: str
    clarification: str

# Reusable function for contacting an A2A agent.
async def call_agent(agent_url: str, question: str) -> str:
    async with httpx.AsyncClient(timeout=90.0) as http_client:
        client = await create_client(
            agent=agent_url,
            client_config=ClientConfig(
                streaming=False,
                httpx_client=http_client,
            ),
        )

        try:
            message = new_text_message(
                question,
                role=Role.ROLE_USER,
            )

            request = SendMessageRequest(message=message)
            answers = []

            async for response in client.send_message(request):
                if response.HasField("message"):
                    text = get_message_text(response.message)

                    if text.strip():
                        answers.append(text)

            if not answers:
                raise RuntimeError("The agent returned no text answer.")

            return "\n\n".join(answers)

        finally:
            await client.close()


async def detect_node(state: HealthcareState) -> dict:
    decision = await detect_intents(state["question"])

    print(
        "[Router] "
        f"Benefits: {bool(decision.benefits_question)} | "
        f"Claims: {bool(decision.claims_question)} | "
        f"FindCare: {bool(decision.findcare_question)}"
    )

    return decision.model_dump()


async def dispatch_node(state: HealthcareState) -> dict:
    clarification = state["clarification"].strip()

    if clarification:
        return {"answer": clarification}

    requests = []

    if state["benefits_question"].strip():
        requests.append(
            (
                "Benefits",
                "http://127.0.0.1:8001",
                state["benefits_question"],
            )
        )

    if state["claims_question"].strip():
        requests.append(
            (
                "Claims",
                "http://127.0.0.1:8003",
                state["claims_question"],
            )
        )
    if state["findcare_question"].strip():
        requests.append(
            (
                "FindCare",
                "http://127.0.0.1:8004",
                state["findcare_question"],
            )
        )

    if not requests:
        return {
            "answer": (
                "Please ask about demo plan benefits, "
                "claim status, or finding a provider."
            )
        }

    async def run_request(label, url, question):
        print(f"[LangGraph] Calling {label} agent")

        try:
            answer = await call_agent(url, question)
        except Exception:
            logging.exception("%s agent request failed", label)
            answer = (
                f"The {label} service could not be reached. "
                "Please try again."
            )

        return f"{label}\n{answer}"

    # Start independent agent calls concurrently.
    answers = await asyncio.gather(
        *[
            run_request(label, url, question)
            for label, url, question in requests
        ]
    )

    # Preserve each agent's answer and citations.
    return {"answer": "\n\n".join(answers)}


builder = StateGraph(HealthcareState)

builder.add_node("detect_intents", detect_node)
builder.add_node("dispatch", dispatch_node)

builder.add_edge(START, "detect_intents")
builder.add_edge("detect_intents", "dispatch")
builder.add_edge("dispatch", END)

healthcare_graph = builder.compile()