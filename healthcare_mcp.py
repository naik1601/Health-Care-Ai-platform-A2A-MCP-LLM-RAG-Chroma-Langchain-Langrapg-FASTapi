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
    name="plan_pdf_openai_v1",
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
        n_results=4,
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
                "page": metadata.get("page"),
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

# Invented records for our demo member.
DEMO_CLAIMS = {
    "CLM-1001": {
        "member_id": "DEMO-MEMBER-001",
        "service": "Physiotherapy",
        "status": "Pending",
        "submitted_date": "2026-10-28",
        "billed_amount": 250.00,
        "currency": "USD",
        "status_details": "Awaiting supporting documents.",
    },
    "CLM-1002": {
        "member_id": "DEMO-MEMBER-001",
        "service": "Annual preventive checkup",
        "status": "Approved",
        "submitted_date": "2026-10-20",
        "billed_amount": 180.00,
        "currency": "USD",
        "status_details": (
            "Claim approved. Payment status is not available "
            "in this demo record."
        ),
    },
}


@mcp.tool()
def get_claim_status(claim_id: str) -> dict:
    """Retrieve a synthetic claim belonging to the fixed demo member."""
    normalized_id = claim_id.strip().upper()
    claim = DEMO_CLAIMS.get(normalized_id)

    if claim is None:
        return {
            "found": False,
            "claim_id": normalized_id,
            "message": "No matching claim found in the demo records.",
        }

    return {
        "found": True,
        "claim_id": normalized_id,
        **claim,
    }

# Fictional providers for testing only.
DEMO_PROVIDERS = [
    {
        "provider_id": "PROV-001",
        "name": "Demo Lakeside Physical Therapy",
        "specialty": "physiotherapy",
        "city": "Chicago",
        "plan_ids": ["DEMO-PLAN-001"],
        "accepting_new_patients": True,
    },
    {
        "provider_id": "PROV-002",
        "name": "Demo Central Rehabilitation",
        "specialty": "physiotherapy",
        "city": "Chicago",
        "plan_ids": ["DEMO-PLAN-001"],
        "accepting_new_patients": False,
    },
    {
        "provider_id": "PROV-003",
        "name": "Demo Springfield Physical Therapy",
        "specialty": "physiotherapy",
        "city": "Springfield",
        "plan_ids": ["DEMO-PLAN-001"],
        "accepting_new_patients": True,
    },
    {
        "provider_id": "PROV-004",
        "name": "Demo Chicago Primary Care",
        "specialty": "primary care",
        "city": "Chicago",
        "plan_ids": ["DEMO-PLAN-001"],
        "accepting_new_patients": True,
    },
]


@mcp.tool()
def search_providers(
    specialty: str,
    city: str,
    plan_id: str,
) -> dict:
    """Search fictional providers by specialty, city, and demo plan."""
    normalized_specialty = specialty.strip().lower()
    normalized_city = city.strip().lower()
    normalized_plan = plan_id.strip().upper()

    # Recognize a few common names for the same specialty.
    specialty_aliases = {
        "physical therapy": "physiotherapy",
        "physical therapist": "physiotherapy",
        "physiotherapist": "physiotherapy",
        "primary care physician": "primary care",
    }

    normalized_specialty = specialty_aliases.get(
        normalized_specialty,
        normalized_specialty,
    )

    if not all(
        [normalized_specialty, normalized_city, normalized_plan]
    ):
        return {
            "providers": [],
            "message": "Specialty, city, and plan ID are required.",
        }

    matches = [
        {
            "provider_id": provider["provider_id"],
            "name": provider["name"],
            "specialty": provider["specialty"],
            "city": provider["city"],
            "accepting_new_patients": (
                provider["accepting_new_patients"]
            ),
        }
        for provider in DEMO_PROVIDERS
        if provider["specialty"] == normalized_specialty
        and provider["city"].lower() == normalized_city
        and normalized_plan in provider["plan_ids"]
    ]

    return {
        "plan_id": normalized_plan,
        "specialty": normalized_specialty,
        "city": city.strip(),
        "providers": matches,
        "message": (
            f"Found {len(matches)} fictional provider(s)."
            if matches
            else "No matching providers found in the demo directory."
        ),
        "source": "Synthetic provider directory",
        "notice": (
            "Fictional records, not real recommendations. "
            "Appointment availability is not provided."
        ),
    }

if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=8002,
    )