import os
import base64
import time
from pathlib import Path
from PIL import Image
import google.genai as genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_IMAGES_IN_CONTEXT = int(os.getenv("MAX_IMAGES_IN_CONTEXT", "3"))
MAX_TEXT_CONTEXT_CHARS = int(os.getenv("MAX_TEXT_CONTEXT_CHARS", "8000"))

if not GEMINI_API_KEY:
    raise ValueError(
        "[Generator] GEMINI_API_KEY not found in .env file. "
        "Please add it before running."
    )

client = genai.Client(api_key=GEMINI_API_KEY)

print(f"[Generator] Gemini client initialized. Model: {GEMINI_MODEL}")

def encode_image_to_base64(image_path: str) -> tuple[str, str]:

    try:
        img = Image.open(image_path)

        if img.mode != "RGB":
            img = img.convert("RGB")

        if img.width > 1024 or img.height > 1024:
            img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

        from io import BytesIO
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        image_bytes = buffer.getvalue()

        encoded = base64.b64encode(image_bytes).decode("utf-8")

        return encoded, "image/jpeg"

    except Exception as e:
        print(f"[Generator] Failed to encode image {image_path}: {e}")
        return None, None
    
def build_prompt_parts(
    query: str,
    text_context: str,
    image_paths: list[str]
) -> list:

    parts = []

    system_instruction = """You are an expert multimodal RAG (Retrieval-Augmented Generation) assistant.
You will be given:
1. Text context retrieved from documents (with source labels)
2. Relevant images from those documents
3. A user question

Your job is to:
- Answer the question using ONLY the provided context
- When your answer references an image, explicitly mention it: "As shown in the image from page X..."
- When your answer references text, cite the source: "According to [document name], page X..."
- If the context does not contain enough information, say: "The provided documents do not contain sufficient information to answer this question."
- Be precise and factual. Do not hallucinate or add information not in the context.
- Structure your answer clearly with the most important information first."""

    parts.append(
        types.Part.from_text(text=f"INSTRUCTIONS:\n{system_instruction}")
    )

    if text_context:
        trimmed_context = text_context[:MAX_TEXT_CONTEXT_CHARS]
        if len(text_context) > MAX_TEXT_CONTEXT_CHARS:
            trimmed_context += "\n[Context trimmed for length...]"

        parts.append(
            types.Part.from_text(
                text=f"RETRIEVED CONTEXT FROM DOCUMENTS:\n\n{trimmed_context}"
            )
        )

    images_added = 0
    for image_path in image_paths:

        if images_added >= MAX_IMAGES_IN_CONTEXT:
            print(f"[Generator] Image limit reached ({MAX_IMAGES_IN_CONTEXT}), skipping remaining")
            break

        if not os.path.exists(image_path):
            print(f"[Generator] Image not found, skipping: {image_path}")
            continue

        encoded, mime_type = encode_image_to_base64(image_path)

        if encoded:
            parts.append(
                types.Part.from_bytes(
                    data=base64.b64decode(encoded),
                    mime_type=mime_type
                )
            )
            parts.append(
                types.Part.from_text(
                    text=f"[Image from: {Path(image_path).name}]"
                )
            )
            images_added += 1
            print(f"[Generator] Added image to context: {Path(image_path).name}")

    parts.append(
        types.Part.from_text(text=f"USER QUESTION:\n{query}")
    )

    return parts

def generate_answer(context: dict) -> dict:

    query = context["query"]
    text_context = context["text_context"]
    image_paths = context["image_paths"]
    source_references = context["source_references"]

    print(f"[Generator] Generating answer for: '{query}'")
    print(f"[Generator] Context: {len(text_context)} chars text, {len(image_paths)} images")

    parts = build_prompt_parts(query, text_context, image_paths)

    start_time = time.time()

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=parts,
        )

        elapsed = round(time.time() - start_time, 2)
        print(f"[Generator] Response received in {elapsed}s")

        answer_text = response.text

        return {
            "answer": answer_text,
            "sources": source_references,
            "query": query,
            "response_time_seconds": elapsed,
            "images_used": len(image_paths),
            "context_chunks_used": len(source_references)
        }

    except Exception as e:
        elapsed = round(time.time() - start_time, 2)
        print(f"[Generator] Gemini API error after {elapsed}s: {e}")

        return {
            "answer": f"Error generating answer: {str(e)}",
            "sources": source_references,
            "query": query,
            "response_time_seconds": elapsed,
            "images_used": 0,
            "context_chunks_used": 0,
            "error": str(e)
        }
    
def run_rag_pipeline(query: str, n_results: int = 5) -> dict:

    from src.retrieval.retriever import retrieve, format_context_for_generation

    print(f"\n{'='*50}")
    print(f"[RAG Pipeline] Query: {query}")
    print(f"{'='*50}")

    retrieval_result = retrieve(query, n_results=n_results)

    if retrieval_result["total_found"] == 0:
        return {
            "answer": "No relevant documents found in the database. Please ingest documents first.",
            "sources": [],
            "query": query,
            "response_time_seconds": 0,
            "images_used": 0,
            "context_chunks_used": 0
        }

    context = format_context_for_generation(retrieval_result)

    result = generate_answer(context)

    result["retrieval_breakdown"] = retrieval_result["breakdown"]

    print(f"[RAG Pipeline] Complete. Time: {result['response_time_seconds']}s")

    return result