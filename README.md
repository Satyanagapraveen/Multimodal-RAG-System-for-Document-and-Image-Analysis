# Multimodal RAG System

A Retrieval-Augmented Generation system that processes PDFs and images to answer questions using both text and visual context via Google Gemini 2.5 Flash.

## What This System Does

1. **Ingests** PDF documents — extracts text, embedded images, and tables
2. **Embeds** all content using CLIP (text and images in the same vector space)
3. **Stores** embeddings in ChromaDB with full metadata
4. **Retrieves** relevant chunks across all modalities using cross-modal search
5. **Generates** grounded answers using Gemini 2.5 Flash with image context

## Project Structure

```
multimodal_rag/
├── src/
│   ├── api/
│   │   └── main.py              # FastAPI endpoints
│   ├── ingestion/
│   │   ├── document_parser.py   # PDF text/image/table extraction
│   │   └── image_processor.py  # Standalone image OCR
│   ├── embeddings/
│   │   └── model_loader.py      # CLIP embedding functions
│   ├── retrieval/
│   │   └── retriever.py         # Cross-modal retrieval + fusion
│   ├── generation/
│   │   └── generator.py         # Gemini answer generation
│   └── vector_store/
│       └── chroma_manager.py    # ChromaDB operations
├── tests/
│   ├── test_ingestion.py        # Parser and embedding tests
│   └── test_api.py              # API endpoint tests
├── sample_documents/            # Input PDFs (10 papers)
├── extracted_images/            # Images extracted from PDFs
├── .env                         # API keys (not in git)
├── .env.example                 # Template for .env
├── requirements.txt
└── ARCHITECTURE.md
```

## Setup

### 1. Clone and create virtual environment
```bash
git clone https://github.com/Satyanagapraveen/Multimodal-RAG-System-for-Document-and-Image-Analysis
cd multimodal_rag
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 4. Add documents and ingest
```bash
# Place PDF files in sample_documents/
python -c "
import os
from src.ingestion.document_parser import parse_document
from src.vector_store.chroma_manager import ingest_parsed_document
for f in os.listdir('sample_documents'):
    if f.endswith('.pdf'):
        result = ingest_parsed_document(parse_document(f'sample_documents/{f}'))
        print(f'{f}: {result}')
"
```

### 5. Start the API
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### POST /query
Ask a question against the document database.
```json
{
  "query": "Describe the transformer architecture diagram",
  "n_results": 5
}
```

Response:
```json
{
  "query": "...",
  "answer": "As shown in the image from page 4...",
  "sources": [{"document_id": "...", "page_number": 4, "content_type": "image"}],
  "response_time_seconds": 4.91,
  "images_used": 2,
  "context_chunks_used": 5,
  "retrieval_breakdown": {"text": 2, "images": 2, "tables": 1}
}
```

### POST /ingest
Upload a PDF or image file.
```bash
curl -X POST "http://localhost:8000/ingest" \
  -F "file=@document.pdf"
```

### GET /health
Check system status and chunk counts.

## Running Tests
```bash
pytest tests/ -v
```

Expected: **38 passed**

## Technology Stack

| Component | Technology | Why |
|---|---|---|
| PDF Parsing | PyMuPDF + pdfplumber | Best text/image/table extraction |
| OCR | EasyOCR | No binary dependencies, 80+ languages |
| Embeddings | CLIP ViT-B-32 | Shared text+image vector space |
| Vector DB | ChromaDB | Local persistent, no server needed |
| LLM | Gemini 2.5 Flash | Fast multimodal responses |
| API | FastAPI | Auto-docs, async, Pydantic validation |

## Design Decisions

**WHY CLIP for embeddings:**
CLIP places text and images in the same 512-dimensional vector space. This means a text query like "show me the architecture diagram" can retrieve the actual image — not just text that mentions diagrams.

**WHY guaranteed modality slots in retrieval:**
Pure score fusion causes text results to always outscore images (text-to-text similarity is naturally higher than cross-modal similarity). Guaranteed slots ensure images always appear in context.

**WHY ChromaDB over Pinecone/Weaviate:**
Local deployment, zero infrastructure, data persists on disk. Suitable for a self-contained submission that works offline.
