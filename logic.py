import fitz  # PyMuPDF
import torch
import streamlit as st
import io
import pytesseract
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------------------------------------------------
# TESSERACT PATH (HF / Docker)
# -------------------------------------------------
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# -------------------------------------------------
# LOAD CLIP (cached)
# -------------------------------------------------
@st.cache_resource
def load_clip():
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id, use_fast=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    return model, processor, device

clip_model, clip_processor, device = load_clip()

# -------------------------------------------------
# TEXT SPLITTER
# -------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=400,
    separators=["\n\n", "\n", ".", " ", ""]
)

# -------------------------------------------------
# OCR (NEVER RETURNS "NO TEXT FOUND")
# -------------------------------------------------
def get_image_text(image_bytes: bytes) -> str:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        text = pytesseract.image_to_string(
            img, config="--oem 3 --psm 6"
        ).strip()

        if len(text) > 5:
            return text

        # CRITICAL: never disqualify an image
        return (
            "Image contains visual elements such as text, layout, charts, "
            "tables, or design features that may require visual interpretation."
        )

    except Exception as e:
        return (
            "Image contains visual information but OCR extraction failed."
        )

# -------------------------------------------------
# CLIP TEXT EMBEDDING (L2 NORMALIZED)
# -------------------------------------------------
def get_clip_text_embedding(text: str):
    if not text or clip_model is None:
        return None

    inputs = clip_processor(
        text=[text],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=77
    ).to(device)

    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)

    vec = features.cpu().numpy()[0]
    return vec / (np.linalg.norm(vec) + 1e-8)

# -------------------------------------------------
# PDF PROCESSING
# -------------------------------------------------
def process_pdf(file_content: bytes, filename: str):
    doc = fitz.open(stream=file_content, filetype="pdf")
    items = []

    for page_idx, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:
            header = f"SOURCE: {filename} | PAGE: {page_idx}\n"
            for chunk in text_splitter.split_text(text):
                items.append({
                    "type": "text",
                    "content": header + chunk,
                    "metadata": {"source": filename, "page": page_idx}
                })

        for img_i, img in enumerate(page.get_images(full=True)):
            pix = doc.extract_image(img[0])
            ocr_text = get_image_text(pix["image"])

            items.append({
                "type": "image",
                "content": (
                    f"PDF IMAGE from {filename}, Page {page_idx}. "
                    f"OCR TEXT: {ocr_text}"
                ),
                "image_bytes": pix["image"],
                "metadata": {
                    "source": filename,
                    "page": page_idx
                }
            })

    doc.close()
    return items

# -------------------------------------------------
# TEXT FILE
# -------------------------------------------------
def process_text_file(file_content: bytes, filename: str):
    text = file_content.decode("utf-8", errors="ignore")
    return [
        {
            "type": "text",
            "content": f"SOURCE: {filename}\n{c}",
            "metadata": {"source": filename}
        }
        for c in text_splitter.split_text(text)
    ]

# -------------------------------------------------
# STANDALONE IMAGE (CRITICAL DIFFERENTIATION)
# -------------------------------------------------
def process_standalone_image(image_bytes: bytes, source: str):
    ocr_text = get_image_text(image_bytes)
    return [{
        "type": "image",
        "content": (
            f"STANDALONE IMAGE uploaded by user.\n"
            f"FILE: {source}\n"
            f"OCR TEXT: {ocr_text}"
        ),
        "image_bytes": image_bytes,
        "metadata": {
            "source": source,
            "page": None  # IMPORTANT: marks it as standalone
        }
    }]
