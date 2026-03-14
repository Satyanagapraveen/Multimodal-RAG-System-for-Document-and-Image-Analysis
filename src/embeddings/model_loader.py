import os
import numpy as np
from pathlib import Path
from PIL import Image
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "clip-ViT-B-32"
)

print(f"[ModelLoader] Loading embedding model: {EMBEDDING_MODEL_NAME}")

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

print(f"[ModelLoader] Model loaded successfully.")

def embed_text(text: str) -> list[float]:

    if not text or not text.strip():
        print("[ModelLoader] Warning: empty text received, returning zero vector")
        return [0.0] * 512

    embedding = embedding_model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embedding.tolist()

def embed_image(image_path: str) -> list[float]:

    if not os.path.exists(image_path):
        print(f"[ModelLoader] Warning: image not found: {image_path}")
        return [0.0] * 512

    try:

        image = Image.open(image_path).convert("RGB")

        embedding = embedding_model.encode(
            image,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embedding.tolist()

    except Exception as e:
        print(f"[ModelLoader] Error embedding image {image_path}: {e}")
        return [0.0] * 512
    
def get_embedding_dimension() -> int:

    test_embedding = embed_text("test")
    return len(test_embedding)