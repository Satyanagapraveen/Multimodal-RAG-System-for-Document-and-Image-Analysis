import pytest
import os
import shutil
from pathlib import Path


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_pdf_path():
    """Returns path to first available PDF in sample_documents."""
    sample_dir = "sample_documents"
    pdfs = [f for f in os.listdir(sample_dir) if f.endswith(".pdf")]
    assert len(pdfs) > 0, "No PDF files found in sample_documents/"
    return os.path.join(sample_dir, pdfs[0])


@pytest.fixture(scope="module")
def parsed_document(sample_pdf_path):
    """Parses one PDF once and shares result across all tests."""
    from src.ingestion.document_parser import parse_document
    return parse_document(sample_pdf_path)


# ── Document Parser Tests ────────────────────────────────────────────────────

class TestDocumentParser:

    def test_parse_document_returns_dict(self, parsed_document):
        """parse_document must return a dictionary."""
        assert isinstance(parsed_document, dict)

    def test_parse_document_has_required_keys(self, parsed_document):
        """Result must contain all required keys."""
        required_keys = ["text_chunks", "image_chunks", "table_chunks", "total_chunks"]
        for key in required_keys:
            assert key in parsed_document, f"Missing key: {key}"

    def test_text_chunks_not_empty(self, parsed_document):
        """Every PDF must produce at least one text chunk."""
        assert len(parsed_document["text_chunks"]) > 0

    def test_text_chunk_structure(self, parsed_document):
        """Each text chunk must have content and metadata."""
        for chunk in parsed_document["text_chunks"]:
            assert "content" in chunk
            assert "metadata" in chunk
            assert isinstance(chunk["content"], str)
            assert len(chunk["content"]) > 0

    def test_text_chunk_metadata_fields(self, parsed_document):
        """Text chunk metadata must have all required fields."""
        required_fields = ["source_document", "page_number", "content_type", "chunk_id"]
        for chunk in parsed_document["text_chunks"]:
            for field in required_fields:
                assert field in chunk["metadata"], \
                    f"Missing metadata field: {field}"

    def test_text_chunk_content_type(self, parsed_document):
        """Text chunks must have content_type = 'text'."""
        for chunk in parsed_document["text_chunks"]:
            assert chunk["metadata"]["content_type"] == "text"

    def test_image_chunk_structure(self, parsed_document):
        """Image chunks must have image_path in metadata."""
        for chunk in parsed_document["image_chunks"]:
            assert "content" in chunk
            assert "metadata" in chunk
            assert "image_path" in chunk["metadata"]

    def test_image_files_saved_to_disk(self, parsed_document):
        """Extracted images must actually exist on disk."""
        for chunk in parsed_document["image_chunks"]:
            image_path = chunk["metadata"]["image_path"]
            assert os.path.exists(image_path), \
                f"Image file not found: {image_path}"

    def test_total_chunks_is_sum(self, parsed_document):
        """total_chunks must equal sum of all chunk types."""
        expected = (
            len(parsed_document["text_chunks"]) +
            len(parsed_document["image_chunks"]) +
            len(parsed_document["table_chunks"])
        )
        assert parsed_document["total_chunks"] == expected

    def test_chunk_ids_are_unique(self, parsed_document):
        """All chunk IDs across all types must be unique."""
        all_ids = []
        for chunk in parsed_document["text_chunks"]:
            all_ids.append(chunk["metadata"]["chunk_id"])
        for chunk in parsed_document["image_chunks"]:
            all_ids.append(chunk["metadata"]["chunk_id"])
        for chunk in parsed_document["table_chunks"]:
            all_ids.append(chunk["metadata"]["chunk_id"])

        assert len(all_ids) == len(set(all_ids)), \
            "Duplicate chunk IDs found"

    def test_page_numbers_are_positive(self, parsed_document):
        """Page numbers must be >= 1."""
        for chunk in parsed_document["text_chunks"]:
            assert chunk["metadata"]["page_number"] >= 1


# ── Embedding Tests ──────────────────────────────────────────────────────────

class TestEmbeddings:

    def test_embed_text_returns_list(self):
        """embed_text must return a list."""
        from src.embeddings.model_loader import embed_text
        result = embed_text("test query")
        assert isinstance(result, list)

    def test_embed_text_correct_dimension(self):
        """embed_text must return 512-dimensional vector."""
        from src.embeddings.model_loader import embed_text
        result = embed_text("transformer architecture")
        assert len(result) == 512

    def test_embed_text_returns_floats(self):
        """Embedding values must be floats."""
        from src.embeddings.model_loader import embed_text
        result = embed_text("test")
        assert all(isinstance(v, float) for v in result)

    def test_embed_text_empty_returns_zero_vector(self):
        """Empty text must return zero vector of length 512."""
        from src.embeddings.model_loader import embed_text
        result = embed_text("")
        assert len(result) == 512
        assert all(v == 0.0 for v in result)

    def test_embed_image_correct_dimension(self, parsed_document):
        """embed_image must return 512-dimensional vector."""
        from src.embeddings.model_loader import embed_image
        if not parsed_document["image_chunks"]:
            pytest.skip("No images in this PDF")
        image_path = parsed_document["image_chunks"][0]["metadata"]["image_path"]
        result = embed_image(image_path)
        assert len(result) == 512

    def test_embed_image_missing_file_returns_zero_vector(self):
        """Missing image must return zero vector gracefully."""
        from src.embeddings.model_loader import embed_image
        result = embed_image("nonexistent/path/image.png")
        assert len(result) == 512
        assert all(v == 0.0 for v in result)

    def test_text_image_similarity_positive(self, parsed_document):
        """Text and related image must have positive similarity."""
        import numpy as np
        from src.embeddings.model_loader import embed_text, embed_image
        if not parsed_document["image_chunks"]:
            pytest.skip("No images in this PDF")

        text_vec = embed_text("diagram figure chart")
        image_path = parsed_document["image_chunks"][0]["metadata"]["image_path"]
        img_vec = embed_image(image_path)

        similarity = np.dot(text_vec, img_vec)
        assert similarity > 0, \
            f"Expected positive similarity, got {similarity}"