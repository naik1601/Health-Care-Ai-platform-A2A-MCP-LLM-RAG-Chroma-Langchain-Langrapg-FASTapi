import asyncio
import logging

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from a2a.client import ClientConfig, create_client
from a2a.helpers import get_message_text, new_text_message
from a2a.types import Role, SendMessageRequest
from orchestrator import healthcare_graph

app = FastAPI(
    title="Healthcare AI Platform",
    version="0.1.0",
)


# Define the JSON our endpoint accepts.
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


# Define the JSON our endpoint returns.
class ChatResponse(BaseModel):
    answer: str


@app.get("/")
async def home():
    return {"message": "Healthcare AI backend is running"}


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    question = request.message.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Please enter a non-empty message.",
        )

    try:
        async with asyncio.timeout(90):
                         result = await healthcare_graph.ainvoke(
                {
                    "question": question,
                    "answer": "",
                    "benefits_question": "",
                    "claims_question": "",
                    "findcare_question": "",
                    "clarification": "",
                }
            )
            

        return ChatResponse(answer=result["answer"])

    except (TimeoutError, httpx.TimeoutException):
        raise HTTPException(
            status_code=504,
            detail="The benefits assistant took too long to respond.",
        )

    except Exception:
        logging.exception("Healthcare workflow failed")

        raise HTTPException(
            status_code=502,
            detail="Unable to complete the healthcare request.",
        )