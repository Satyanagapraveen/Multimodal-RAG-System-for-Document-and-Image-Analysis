import fitz
import os
from PIL import Image

def parse_pdf(filepath,image_output_dir="extracted images"):
    document_chunks=[]
    pdf=fitz.open(filepath)
    os.makedirs(image_output_dir,exists_ok=True)
    for page_number in range(len(pdf)):
        page=pdf[page_number]
        text=page.get_text(page)
        if text.strip():
            document_chunks.append({
                "type":"text",
                "content":text,
                "page number":page_number+1,
                "document":os.path.basename(filepath)
            })
        image_list=page.get_images(full=True)
        for img_index,img in enumerate(image_list):
            xref=img[0]
            base_image=pdf.extract_image(xref)
            image_bytes=base_image["image"]
            image_ext=base_image["ext"]
            image_filename=f"page{page_number+1}_img{img_index}.{image_ext}"
            image_path=os.path.join(image_output_dir,image_filename)
            with open(image_path)