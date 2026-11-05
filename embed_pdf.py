import json
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI


PROJECT_PATH = Path(__file__).resolve().parent
CHUNKS_PATH = PROJECT_PATH / "data" / "pdf_chunks.json"

EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "plan_pdf_openai_v1"

load_dotenv(PROJECT_PATH / ".env")


def main():
    # Read the chunks created by chunk_pdf.py.
    records = json.loads(
        CHUNKS_PATH.read_text(encoding="utf-8")
    )

    if not records:
        raise ValueError("No chunks found in pdf_chunks.json")

    texts = [record["text"] for record in records]

    if any(not text.strip() for text in texts):
        raise ValueError("An empty chunk was found.")

    # Generate document embeddings in small batches.
    embeddings = []

    with OpenAI(timeout=60.0, max_retries=2) as client:
        batch_size = 16

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]

            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch,
            )

            batch_vectors = [
                item.embedding
                for item in sorted(
                    response.data,
                    key=lambda item: item.index,
                )
            ]

            embeddings.extend(batch_vectors)

            print(
                f"Embedded {len(embeddings)} "
                f"of {len(texts)} chunks"
            )

    # Open a separate collection for PDF content.
    chroma_client = chromadb.PersistentClient(
        path=str(PROJECT_PATH / "chroma_data")
    )

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
        metadata={"embedding_model": EMBEDDING_MODEL},
    )

    collection.upsert(
        ids=[record["id"] for record in records],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {
                **record["metadata"],
                "embedding_model": EMBEDDING_MODEL,
            }
            for record in records
        ],
    )

    print("\nPDF embeddings saved successfully!")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Stored chunks: {collection.count()}")
    print(f"Vector dimensions: {len(embeddings[0])}")


if __name__ == "__main__":
    main()