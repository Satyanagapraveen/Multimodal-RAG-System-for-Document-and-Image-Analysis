import os
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from pathlib import Path
from src.embeddings.model_loader import embed_text, embed_image, get_embedding_dimension

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")

print(f"[ChromaManager] Initializing ChromaDB at: {CHROMA_DB_PATH}")

chroma_client = chromadb.PersistentClient(
    path=CHROMA_DB_PATH,
    settings=Settings(anonymized_telemetry=False)
)

EMBEDDING_DIM = get_embedding_dimension()

text_collection = chroma_client.get_or_create_collection(
    name="text_chunks",
    metadata={"hnsw:space": "cosine"}
)

image_collection = chroma_client.get_or_create_collection(
    name="image_chunks",
    metadata={"hnsw:space": "cosine"}
)

table_collection = chroma_client.get_or_create_collection(
    name="table_chunks",
    metadata={"hnsw:space": "cosine"}
)

print(f"[ChromaManager] Collections ready. Embedding dim: {EMBEDDING_DIM}")

def add_text_chunks(chunks: list[dict]) -> int:

    if not chunks:
        print("[ChromaManager] No text chunks to add.")
        return 0

    added_count = 0

    for chunk in chunks:

        chunk_id = chunk["metadata"]["chunk_id"]

        existing = text_collection.get(ids=[chunk_id])

        if existing["ids"]:
            print(f"[ChromaManager] Skipping duplicate: {chunk_id}")
            continue

        content = chunk["content"]

        # Filter out low-quality table extractions
        meaningful_chars = len([c for c in content if c.isalnum()])
        if meaningful_chars < 20:
            print(f"[ChromaManager] Skipping low-quality table: {chunk['metadata']['chunk_id']}")
            continue

        embedding = embed_text(content)

        text_collection.add(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[chunk["metadata"]]
        )

        added_count += 1

    print(f"[ChromaManager] Added {added_count} text chunks to ChromaDB")
    return added_count

def add_image_chunks(chunks: list[dict]) -> int:

    if not chunks:
        print("[ChromaManager] No image chunks to add.")
        return 0

    added_count = 0

    for chunk in chunks:

        chunk_id = chunk["metadata"]["chunk_id"]

        existing = image_collection.get(ids=[chunk_id])

        if existing["ids"]:
            print(f"[ChromaManager] Skipping duplicate image: {chunk_id}")
            continue

        image_path = chunk["metadata"].get("image_path", "")

        if image_path and os.path.exists(image_path):
            embedding = embed_image(image_path)
        else:
            embedding = embed_text(chunk["content"])
            print(f"[ChromaManager] Image not found, using text embedding for: {chunk_id}")

                # Normalize the image path before storing
        if "image_path" in chunk["metadata"]:
            chunk["metadata"]["image_path"] = str(
                Path(chunk["metadata"]["image_path"]).resolve()
            )

        image_collection.add(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[chunk["content"]],
            metadatas=[chunk["metadata"]]
        )

        added_count += 1

    print(f"[ChromaManager] Added {added_count} image chunks to ChromaDB")
    return added_count

def add_table_chunks(chunks: list[dict]) -> int:

    if not chunks:
        print("[ChromaManager] No table chunks to add.")
        return 0

    added_count = 0

    for chunk in chunks:

        chunk_id = chunk["metadata"]["chunk_id"]

        existing = table_collection.get(ids=[chunk_id])

        if existing["ids"]:
            print(f"[ChromaManager] Skipping duplicate table: {chunk_id}")
            continue

        content = chunk["content"]

        embedding = embed_text(content)

        table_collection.add(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[chunk["metadata"]]
        )

        added_count += 1

    print(f"[ChromaManager] Added {added_count} table chunks to ChromaDB")
    return added_count

def ingest_parsed_document(parsed_result: dict) -> dict:

    print(f"[ChromaManager] Starting ingestion...")

    text_added = add_text_chunks(parsed_result["text_chunks"])
    image_added = add_image_chunks(parsed_result["image_chunks"])
    table_added = add_table_chunks(parsed_result["table_chunks"])

    summary = {
        "text_chunks_added": text_added,
        "image_chunks_added": image_added,
        "table_chunks_added": table_added,
        "total_added": text_added + image_added + table_added
    }

    print(f"[ChromaManager] Ingestion complete: {summary}")
    return summary

def search_text_collection(query_embedding: list[float], n_results: int = 5) -> dict:

    results = text_collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, text_collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    return results


def search_image_collection(query_embedding: list[float], n_results: int = 5) -> dict:

    results = image_collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, image_collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    return results


def search_table_collection(query_embedding: list[float], n_results: int = 5) -> dict:

    results = table_collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, table_collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    return results


def get_collection_stats() -> dict:

    return {
        "text_chunks": text_collection.count(),
        "image_chunks": image_collection.count(),
        "table_chunks": table_collection.count(),
        "total": text_collection.count() + image_collection.count() + table_collection.count()
    }