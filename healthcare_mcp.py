from mcp.server import MCPServer
from pathlib import Path

import chromadb

from dotenv import load_dotenv
from openai import OpenAI

mcp = MCPServer("Healthcare Tools")


# Synthetic data for learning and testing.
DEMO_BENEFITS = {
    "DEMO-PLAN-001": {
        "physiotherapy": (
            "Demo coverage: up to 20 physiotherapy visits per year. "
            "Prior authorization is required."
        ),
        "dental": (
            "Demo coverage: routine dental care is excluded."
        ),
    }
}


@mcp.tool()
def get_plan_benefit(plan_id: str, service: str) -> str:
    """Look up a service in a synthetic insurance plan."""
    plan = DEMO_BENEFITS.get(plan_id)

    if plan is None:
        return "Unknown demo plan. No coverage information available."

    benefit = plan.get(service.strip().lower())

    if benefit is None:
        return (
            "This service is not listed in the demo data. "
            "Its coverage cannot be determined."
        )

    return benefit

PROJECT_PATH = Path(__file__).resolve().parent
load_dotenv(PROJECT_PATH / ".env")

EMBEDDING_MODEL = "text-embedding-3-small"

openai_client = OpenAI(timeout=30.0, max_retries=2)

chroma_client = chromadb.PersistentClient(
    path=str(PROJECT_PATH / "chroma_data")
)

plan_documents = chroma_client.get_collection(
    name="plan_documents_openai_v1",
    embedding_function=None,
)


@mcp.tool()
def search_plan_documents(
    plan_id: str,
    question: str,
) -> dict:
    """Retrieve candidate passages from a synthetic insurance plan.

    Results are search matches, not confirmed coverage decisions.
    """
    if not plan_id.strip() or not question.strip():
        return {
            "matches": [],
            "message": "A plan ID and question are required.",
        }

    embedding_response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[question],
    )

    question_vector = embedding_response.data[0].embedding

    results = plan_documents.query(
        query_embeddings=[question_vector],
        n_results=2,
        where={"plan_id": plan_id},
        include=["documents", "metadatas", "distances"],
    )

    matches = []

    for index, chunk_id in enumerate(results["ids"][0]):
        metadata = results["metadatas"][0][index] or {}

        matches.append(
            {
                "chunk_id": chunk_id,
                "text": results["documents"][0][index],
                "source": metadata.get("source", "Unknown"),
                "section": metadata.get("section", "Unknown"),
                "distance": results["distances"][0][index],
            }
        )

    return {
        "plan_id": plan_id,
        "question": question,
        "matches": matches,
        "message": (
            "Retrieved candidate passages."
            if matches
            else "No passages found for this plan."
        ),
    }

if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=8002,
    )