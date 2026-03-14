import os
import shutil
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

SAMPLE_DOCUMENTS_PATH = os.getenv("SAMPLE_DOCUMENTS_PATH", "./sample_documents")
os.makedirs(SAMPLE_DOCUMENTS_PATH, exist_ok=True)


# ── Lifespan ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("[API] Server starting up...")
    from src.embeddings.model_loader import embedding_model
    from src.vector_store.chroma_manager import get_collection_stats
    stats = get_collection_stats()
    print(f"[API] ChromaDB ready: {stats}")
    print("[API] Server ready to accept requests")
    yield
    # Shutdown
    print("[API] Server shutting down...")


# ── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Multimodal RAG API",
    description="A RAG system that understands text, images, and tables from documents",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("[API] FastAPI app initialized")


# ── Request / Response Models ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    n_results: int = 5


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[dict]
    response_time_seconds: float
    images_used: int
    context_chunks_used: int
    retrieval_breakdown: dict


class IngestResponse(BaseModel):
    filename: str
    status: str
    chunks_added: dict
    message: str


class HealthResponse(BaseModel):
    status: str
    collections: dict
    models_loaded: bool
    message: str


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": "Multimodal RAG API is running",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "query": "POST /query",
            "ingest": "POST /ingest",
            "health": "GET /health"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        from src.vector_store.chroma_manager import get_collection_stats
        from src.embeddings.model_loader import embedding_model

        stats = get_collection_stats()
        models_loaded = embedding_model is not None

        return HealthResponse(
            status="healthy",
            collections=stats,
            models_loaded=models_loaded,
            message=f"System ready. {stats['total']} chunks indexed."
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}"
        )


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):

    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )

    if len(request.query) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Query too long. Maximum 1000 characters."
        )

    try:
        from src.generation.generator import run_rag_pipeline

        result = run_rag_pipeline(
            query=request.query.strip(),
            n_results=request.n_results
        )

        return QueryResponse(
            query=result["query"],
            answer=result["answer"],
            sources=result["sources"],
            response_time_seconds=result["response_time_seconds"],
            images_used=result["images_used"],
            context_chunks_used=result["context_chunks_used"],
            retrieval_breakdown=result.get("retrieval_breakdown", {})
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):

    supported_types = {
        "application/pdf": ".pdf",
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
    }

    if file.content_type not in supported_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Supported: PDF, PNG, JPEG"
        )

    file_extension = supported_types[file.content_type]
    safe_filename = Path(file.filename).stem
    safe_filename = "".join(
        c for c in safe_filename
        if c.isalnum() or c in (' ', '-', '_')
    ).strip()
    final_filename = f"{safe_filename}{file_extension}"
    save_path = os.path.join(SAMPLE_DOCUMENTS_PATH, final_filename)

    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"[API] File saved: {save_path}")

        if file_extension == ".pdf":
            from src.ingestion.document_parser import parse_document
            from src.vector_store.chroma_manager import ingest_parsed_document

            parsed = parse_document(save_path)
            summary = ingest_parsed_document(parsed)

            return IngestResponse(
                filename=final_filename,
                status="success",
                chunks_added=summary,
                message=f"Successfully ingested PDF: {summary['total_added']} chunks added"
            )

        else:
            from src.ingestion.image_processor import process_standalone_image
            from src.vector_store.chroma_manager import add_image_chunks

            chunk = process_standalone_image(save_path)
            added = add_image_chunks([chunk])

            return IngestResponse(
                filename=final_filename,
                status="success",
                chunks_added={"image_chunks_added": added, "total_added": added},
                message="Successfully ingested image with OCR"
            )

    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}"
        )