import fitz                    
import pdfplumber              
import os                      
import json                    
from pathlib import Path       
from dotenv import load_dotenv 

load_dotenv()

EXTRACTED_IMAGES_PATH = os.getenv(
    "EXTRACTED_IMAGES_PATH", 
    "./extracted_images"
)

os.makedirs(EXTRACTED_IMAGES_PATH, exist_ok=True)

def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    
    text_chunks = []
    
    doc = fitz.open(pdf_path)
    
    document_name = Path(pdf_path).name
    
    for page_num in range(len(doc)):
        
        page = doc[page_num]
        
        text = page.get_text("text")
        
        text = text.strip()
        
        if text:
            
            chunk = {
                "content": text,
                "metadata": {
                    "source_document": document_name,
                    "page_number": page_num + 1,
                    "content_type": "text",
                    "chunk_id": f"{document_name}_page_{page_num + 1}_text"
                }
            }
            
            text_chunks.append(chunk)
    
    doc.close()
    
    return text_chunks

def extract_images_from_pdf(pdf_path: str) -> list[dict]:
    
    image_chunks = []
    
    doc = fitz.open(pdf_path)
    
    document_name = Path(pdf_path).name
    
    doc_images_folder = os.path.join(
        EXTRACTED_IMAGES_PATH, 
        Path(pdf_path).stem
    )
    
    os.makedirs(doc_images_folder, exist_ok=True)
    
    for page_num in range(len(doc)):
        
        page = doc[page_num]
        
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            
            xref = img[0]
            
            base_image = doc.extract_image(xref)
            
            image_bytes = base_image["image"]
            
            image_ext = base_image["ext"]
            
            image_filename = f"{Path(pdf_path).stem}_page{page_num + 1}_img{img_index + 1}.{image_ext}"
            
            image_save_path = os.path.join(doc_images_folder, image_filename)
            
            with open(image_save_path, "wb") as img_file:
                img_file.write(image_bytes)
            
            chunk = {
                "content": image_save_path,
                "metadata": {
                    "source_document": document_name,
                    "page_number": page_num + 1,
                    "content_type": "image",
                    "chunk_id": f"{document_name}_page_{page_num + 1}_img_{img_index + 1}",
                    "image_path": image_save_path
                }
            }
            
            image_chunks.append(chunk)
    
    doc.close()
    
    return image_chunks

def extract_tables_from_pdf(pdf_path: str) -> list[dict]:
    
    table_chunks = []
    
    document_name = Path(pdf_path).name
    
    with pdfplumber.open(pdf_path) as pdf:
        
        for page_num, page in enumerate(pdf.pages):
            
            tables = page.extract_tables()
            
            for table_index, table in enumerate(tables):
                
                if not table:
                    continue
                
                table_text = ""
                
                for row in table:
                    
                    cleaned_row = [
                        str(cell).strip() if cell is not None else "" 
                        for cell in row
                    ]
                    
                    table_text += " | ".join(cleaned_row) + "\n"
                
                chunk = {
                    "content": table_text.strip(),
                    "metadata": {
                        "source_document": document_name,
                        "page_number": page_num + 1,
                        "content_type": "table",
                        "chunk_id": f"{document_name}_page_{page_num + 1}_table_{table_index + 1}"
                    }
                }
                
                table_chunks.append(chunk)
    
    return table_chunks

def parse_document(pdf_path: str) -> dict:
    
    print(f"[Parser] Processing: {pdf_path}")
    
    text_chunks = extract_text_from_pdf(pdf_path)
    print(f"[Parser] Extracted {len(text_chunks)} text chunks")
    
    image_chunks = extract_images_from_pdf(pdf_path)
    print(f"[Parser] Extracted {len(image_chunks)} images")
    
    table_chunks = extract_tables_from_pdf(pdf_path)
    print(f"[Parser] Extracted {len(table_chunks)} tables")
    
    return {
        "text_chunks": text_chunks,
        "image_chunks": image_chunks,
        "table_chunks": table_chunks,
        "total_chunks": len(text_chunks) + len(image_chunks) + len(table_chunks)
    }