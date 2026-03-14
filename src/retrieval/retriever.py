import os
import numpy as np
from dotenv import load_dotenv
from src.embeddings.model_loader import embed_text
from src.vector_store.chroma_manager import (
    search_text_collection,
    search_image_collection,
    search_table_collection,
    get_collection_stats
)

load_dotenv()
TEXT_RESULTS = int(os.getenv("TEXT_RESULTS", "5"))
IMAGE_RESULTS = int(os.getenv("IMAGE_RESULTS", "3"))
TABLE_RESULTS = int(os.getenv("TABLE_RESULTS", "3"))

DISTANCE_THRESHOLD = float(os.getenv("DISTANCE_THRESHOLD", "0.75"))

def parse_chroma_results(results: dict, content_type: str) -> list[dict]:

    parsed = []

    if not results or not results.get("ids") or not results["ids"][0]:
        return parsed

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i in range(len(ids)):

        distance = distances[i]
        similarity = 1 - distance

        if distance > DISTANCE_THRESHOLD:
            continue

        item = {
            "chunk_id": ids[i],
            "content": documents[i],
            "metadata": metadatas[i],
            "similarity_score": round(similarity, 4),
            "distance": round(distance, 4),
            "content_type": content_type
        }

        parsed.append(item)

    return parsed

def retrieve(query: str, n_results: int = 5) -> dict:

    print(f"[Retriever] Query: '{query}'")

    stats = get_collection_stats()
    if stats["total"] == 0:
        print("[Retriever] WARNING: ChromaDB is empty. Run ingestion first.")
        return {"query": query, "results": [], "total_found": 0}

    query_embedding = embed_text(query)
    print(f"[Retriever] Query embedded. Searching collections...")

    raw_text = search_text_collection(query_embedding, n_results=TEXT_RESULTS)
    raw_images = search_image_collection(query_embedding, n_results=IMAGE_RESULTS)
    raw_tables = search_table_collection(query_embedding, n_results=TABLE_RESULTS)

    text_results = parse_chroma_results(raw_text, "text")
    image_results = parse_chroma_results(raw_images, "image")
    table_results = parse_chroma_results(raw_tables, "table")

    print(f"[Retriever] Found - Text: {len(text_results)}, Images: {len(image_results)}, Tables: {len(table_results)}")

    all_results = text_results + image_results + table_results

    all_results.sort(key=lambda x: x["similarity_score"], reverse=True)

    top_results = all_results[:n_results]

    return {
        "query": query,
        "results": top_results,
        "total_found": len(all_results),
        "breakdown": {
            "text": len(text_results),
            "images": len(image_results),
            "tables": len(table_results)
        }
    }

def format_context_for_generation(retrieval_result: dict) -> dict:

    text_context = []
    image_paths = []
    source_references = []

    for item in retrieval_result["results"]:

        metadata = item["metadata"]
        content_type = item["content_type"]

        source_ref = {
            "document_id": metadata.get("source_document", "unknown"),
            "page_number": metadata.get("page_number", 0),
            "content_type": content_type,
            "similarity_score": item["similarity_score"]
        }

        if content_type == "text":
            text_context.append(
                f"[Source: {metadata.get('source_document')} | "
                f"Page {metadata.get('page_number')} | Text]\n"
                f"{item['content']}"
            )
            source_ref["snippet"] = item["content"][:200]

        elif content_type == "image":
            image_path = metadata.get("image_path", "")
            if image_path and os.path.exists(image_path):
                image_paths.append(image_path)
                source_ref["snippet"] = image_path
            text_context.append(
                f"[Source: {metadata.get('source_document')} | "
                f"Page {metadata.get('page_number')} | Image]\n"
                f"An image was found on this page."
            )

        elif content_type == "table":
            text_context.append(
                f"[Source: {metadata.get('source_document')} | "
                f"Page {metadata.get('page_number')} | Table]\n"
                f"{item['content']}"
            )
            source_ref["snippet"] = item["content"][:200]

        source_references.append(source_ref)

    return {
        "text_context": "\n\n---\n\n".join(text_context),
        "image_paths": image_paths,
        "source_references": source_references,
        "query": retrieval_result["query"]
    }