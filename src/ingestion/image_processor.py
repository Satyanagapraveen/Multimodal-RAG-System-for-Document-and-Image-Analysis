import easyocr
import os
import shutil
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

EXTRACTED_IMAGES_PATH = os.getenv(
    "EXTRACTED_IMAGES_PATH",
    "./extracted_images"
)

os.makedirs(EXTRACTED_IMAGES_PATH, exist_ok=True)

print("[ImageProcessor] Loading EasyOCR model...")
ocr_reader = easyocr.Reader(
    ['en'],
    gpu=False,
    verbose=False
)
print("[ImageProcessor] EasyOCR model loaded.")

def extract_text_from_image(image_path: str) -> str:

    try:

        img = Image.open(image_path)

        if img.mode not in ['RGB', 'L']:
            img = img.convert('RGB')
            converted_path = image_path + "_converted.png"
            img.save(converted_path)
            image_path = converted_path

        results = ocr_reader.readtext(image_path)

        extracted_text = " ".join([
            detection[1]
            for detection in results
        ])

        return extracted_text.strip()

    except Exception as e:
        print(f"[ImageProcessor] OCR failed for {image_path}: {e}")
        return ""
    
def process_standalone_image(image_path: str) -> dict:

    print(f"[ImageProcessor] Processing: {image_path}")

    image_filename = Path(image_path).name

    destination_folder = os.path.join(
        EXTRACTED_IMAGES_PATH,
        "standalone"
    )

    os.makedirs(destination_folder, exist_ok=True)

    destination_path = str(Path(destination_folder) / image_filename)

    shutil.copy2(image_path, destination_path)

    ocr_text = extract_text_from_image(destination_path)

    chunk = {
        "content": ocr_text if ocr_text else f"Image file: {image_filename}",
        "metadata": {
            "source_document": image_filename,
            "page_number": 1,
            "content_type": "image",
            "chunk_id": f"{image_filename}_standalone_img",
            "image_path": destination_path,
            "has_ocr_text": bool(ocr_text)
        }
    }

    print(f"[ImageProcessor] OCR extracted {len(ocr_text)} characters")

    return chunk

def process_image_directory(directory_path: str) -> list[dict]:

    supported_formats = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp'}

    image_chunks = []

    directory = Path(directory_path)

    if not directory.exists():
        print(f"[ImageProcessor] Directory not found: {directory_path}")
        return []

    image_files = [
        f for f in directory.iterdir()
        if f.suffix.lower() in supported_formats
    ]

    print(f"[ImageProcessor] Found {len(image_files)} image files")

    for image_file in image_files:
        chunk = process_standalone_image(str(image_file))
        image_chunks.append(chunk)

    return image_chunks