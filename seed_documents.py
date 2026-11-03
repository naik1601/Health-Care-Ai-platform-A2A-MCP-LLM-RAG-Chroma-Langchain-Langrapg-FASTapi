from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI


PROJECT_PATH = Path(__file__).resolve().parent
load_dotenv(PROJECT_PATH / ".env")

EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "plan_documents_openai_v1"

openai_client = OpenAI(
    timeout=30.0,
    max_retries=2,
)

client = chromadb.PersistentClient(
    path=str(PROJECT_PATH / "chroma_data")
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=None,
    metadata={"embedding_model": EMBEDDING_MODEL},
)


# These are invented document passages for our demo.
documents = [
    {
        "id": "demo-plan-001-physio",
        "text": (
            "Under DEMO-PLAN-001, outpatient physiotherapy "
            "is covered for up to 20 visits per year. "
            "Prior authorization is required."
        ),
        "section": "Physiotherapy",
    },
    {
        "id": "demo-plan-001-dental",
        "text": (
            "Under DEMO-PLAN-001, routine dental checkups "
            "and teeth cleaning are excluded."
        ),
        "section": "Dental",
    },
    {
        "id": "demo-plan-001-preventive",
        "text": (
            "Under DEMO-PLAN-001, one annual preventive "
            "health checkup is covered when performed "
            "by an in-network provider."
        ),
        "section": "Preventive Care",
    },
]


# Extract the text from each document.
texts = [document["text"] for document in documents]

# Send all three passages in one embedding request.
response = openai_client.embeddings.create(
    model=EMBEDDING_MODEL,
    input=texts,
)

# Match returned vectors to the original input order.
embeddings = [
    item.embedding
    for item in sorted(response.data, key=lambda item: item.index)
]

# Store vectors, original text, and metadata together.
collection.upsert(
    ids=[document["id"] for document in documents],
    documents=texts,
    embeddings=embeddings,
    metadatas=[
        {
            "plan_id": "DEMO-PLAN-001",
            "source": "Synthetic Demo Plan",
            "section": document["section"],
            "embedding_model": EMBEDDING_MODEL,
        }
        for document in documents
    ],
)

print(f"Stored document chunks: {collection.count()}")
print(f"Embedding dimensions: {len(embeddings[0])}")

# Embed the search question using the SAME model.
question = "Do I need approval before physical therapy?"

question_response = openai_client.embeddings.create(
    model=EMBEDDING_MODEL,
    input=[question],
)

question_vector = question_response.data[0].embedding

results = collection.query(
    query_embeddings=[question_vector],
    n_results=2,
    where={"plan_id": "DEMO-PLAN-001"},
    include=["documents", "metadatas", "distances"],
)

print(f"\nQuestion: {question}")

for index, text in enumerate(results["documents"][0]):
    metadata = results["metadatas"][0][index]
    distance = results["distances"][0][index]

    print(f"\nMatch {index + 1}:")
    print(text)
    print(f"Source: {metadata['source']}")
    print(f"Section: {metadata['section']}")
    print(f"Distance: {distance:.4f}")

openai_client.close()