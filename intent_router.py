import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel


load_dotenv(Path(__file__).resolve().parent / ".env")


class RoutingDecision(BaseModel):
    benefits_question: str
    claims_question: str
    findcare_question: str
    clarification: str


async def detect_intents(question: str) -> RoutingDecision:
    async with AsyncOpenAI(
        timeout=30.0,
        max_retries=1,
    ) as client:
        response = await client.responses.parse(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
                       instructions=(
                "Route questions for a synthetic healthcare assistant. "
                "Do not answer questions or claim to have searched records. "
                "benefits_question: a standalone question about coverage, "
                "exclusions, copayments, visit limits, or authorization. "
                "claims_question: a standalone question about a submitted "
                "claim's status. Preserve claim IDs exactly. "
                "findcare_question: a standalone request to find providers "
                "by specialty and city. Preserve the requested location "
                "and specialty. Do not invent missing details. "
                "Use an empty string for each category not requested. "
                "If multiple independent intents are present, populate "
                "all applicable question fields. "
                "Carry explicitly stated shared context into the relevant "
                "questions. For example, physiotherapy coverage and finding "
                "a provider in Chicago means finding a physiotherapist "
                "in Chicago. "
                "Route provider searches even when the city or specialty "
                "is missing; FindCare will request missing information. "
                "Route claims without IDs; Claims will request the ID. "
                "Do not route a simple provider search to Benefits unless "
                "the user also asks about coverage or plan rules. "
                "clarification: leave empty for routable requests. "
                "For unrelated or ambiguous requests, or requests where "
                "one agent's result is needed before another can act, "
                "leave all question fields empty and ask a brief "
                "clarification. "
                "Only Benefits, Claims, and FindCare are supported."
            ),
            input=question,
            text_format=RoutingDecision,
            max_output_tokens=500,
            store=False,
        )

    decision = response.output_parsed

    if (
        decision.benefits_question.strip()
        or decision.claims_question.strip()
        or decision.findcare_question.strip()
    ):
        decision.clarification = ""

    return decision

    return decision