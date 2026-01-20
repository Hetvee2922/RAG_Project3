import fitz
import torch
import streamlit as st
import io
import pytesseract
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------------------------------------------------
# TESSERACT CONFIG (HF / Docker)
# -------------------------------------------------
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# -------------------------------------------------
# LOAD CLIP (CACHED)
# -------------------------------------------------
@st.cache_resource
def load_clip_model():
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id, use_fast=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, processor, device

clip_model, clip_processor, device = load_clip_model()

# -------------------------------------------------
# TEXT SPLITTER
# -------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=400,
    separators=["\n\n", "\n", ".", " ", ""]
)

# -------------------------------------------------
# OCR
# -------------------------------------------------
def get_image_text(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        config = r"--oem 3 --psm 6"
        text = pytesseract.image_to_string(img, config=config).strip()
        return text if len(text) > 5 else "No readable text detected."
    except Exception as e:
        return f"OCR failed: {e}"

# -------------------------------------------------
# CLIP EMBEDDING (FIXED & NORMALIZED)
# -------------------------------------------------
def get_clip_text_embedding(text: str):
    if not text or clip_model is None:
        return None
    try:
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

    except Exception as e:
        st.error(f"Embedding Error: {e}")
        return None

# -------------------------------------------------
# PDF PROCESSING
# -------------------------------------------------
def process_pdf(file_content: bytes, filename: str):
    doc = fitz.open(stream=file_content, filetype="pdf")
    extracted = []

    for page_num, page in enumerate(doc):
        page_idx = page_num + 1

        # Digital text
        text = page.get_text().strip()
        if text:
            header = f"SOURCE: {filename} | PAGE: {page_idx}\n"
            for chunk in text_splitter.split_text(text):
                extracted.append({
                    "type": "text",
                    "content": header + chunk,
                    "metadata": {"source": filename, "page": page_idx}
                })

        # Images + OCR
        for img in page.get_images(full=True):
            pix = doc.extract_image(img[0])
            ocr_text = get_image_text(pix["image"])
            extracted.append({
                "type": "image",
                "content": f"SOURCE: {filename} | PAGE: {page_idx} | IMAGE OCR: {ocr_text}",
                "image_bytes": pix["image"],
                "metadata": {"source": filename, "page": page_idx}
            })

    doc.close()
    return extracted

def process_text_file(file_content: bytes, filename: str):
    text = file_content.decode("utf-8", errors="ignore")
    return [{
        "type": "text",
        "content": f"SOURCE: {filename}\n{c}",
        "metadata": {"source": filename}
    } for c in text_splitter.split_text(text)]

def process_standalone_image(image_bytes: bytes, source: str):
    ocr_text = get_image_text(image_bytes)
    return [{
        "type": "image",
        "content": f"IMAGE {source}. OCR: {ocr_text}",
        "image_bytes": image_bytes,
        "metadata": {"source": source}
    }]
