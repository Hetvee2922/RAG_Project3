import fitz
import torch
import streamlit as st
from transformers import CLIPProcessor, CLIPModel
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =========================================================
# MODEL LOADING (Cached for Streamlit)
# =========================================================
@st.cache_resource
def load_clip_model():
    """Loads CLIP model and processor once and caches them."""
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id)
    # use_fast=True silences the processor warning
    processor = CLIPProcessor.from_pretrained(model_id, use_fast=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, processor, device

# Initialize the model globally 
clip_model, clip_processor, device = load_clip_model()

# =========================================================
# TEXT SPLITTING CONFIG
# =========================================================
# CLIP has a strict 77-token limit. We use 400 characters to ensure
# the text fits within that limit while keeping context.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,    
    chunk_overlap=400,
    separators=["\n\n", "\n", ".", " ", ""]
)

# =========================================================
# EMBEDDING LOGIC
# =========================================================
def get_clip_text_embedding(text):
    """Converts text into a 512-dim CLIP embedding vector."""
    try:
        # truncation=True and max_length=77 prevent the sequence length error
        searchable_text = text[:200] 

        inputs = clip_processor(
            text=[searchable_text], 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=77
        ).to(device)
        
        with torch.no_grad():
            text_features = clip_model.get_text_features(**inputs)
        
        return text_features.cpu().numpy()[0]
    except Exception as e:
        print(f"Embedding Log Error: {e}")
        return None

# =========================================================
# TEXT FILE PROCESSING
# =========================================================
def process_text_file(file_content: bytes, filename: str = "document.txt"):
    """Processes plain text files with dynamic source labeling."""
    text = file_content.decode("utf-8", errors="ignore")
    chunks = splitter.split_text(text)
    
    # Prefixing content helps the vector search understand the file context
    context_prefix = f"Source: {filename} | Content: "

    return [{
        "type": "text",
        "content": context_prefix + chunk,
        "metadata": {"source": filename}
    } for chunk in chunks]

# =========================================================
# PDF PROCESSING (TEXT + IMAGES)
# =========================================================
def process_pdf(file_content: bytes, filename: str = "document.pdf"):
    doc = fitz.open(stream=file_content, filetype="pdf")
    all_extracted = []

    for page_num, page in enumerate(doc):
        raw_text = page.get_text().strip()
        if raw_text:
            # IMPORTANT: We "bake" the context into the text itself
            # This helps the weak CLIP model find the right page
            header = f"DOCUMENT: {filename} | PAGE: {page_num + 1}\n"
            chunks = splitter.split_text(raw_text)
            
            for chunk in chunks:
                all_extracted.append({
                    "type": "text",
                    "content": header + chunk, # Prefix every chunk with its location
                    "metadata": {
                        "source": filename,
                        "page": page_num + 1,
                        "text_sample": chunk[:50] # For easier debugging
                    }
                })

        # -------- IMAGE EXTRACTION --------
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            image_info = doc.extract_image(xref)
            image_bytes = image_info["image"]

            all_extracted.append({
                "type": "image",
                "content": f"Visual data from {filename}, Page {page_num + 1}.",
                "image_bytes": image_bytes,
                "metadata": {
                    "source": f"{filename} Image {img_index + 1} (Pg {page_num + 1})",
                    "page": page_num + 1
                }
            })

    doc.close()
    return all_extracted

# =========================================================
# STANDALONE IMAGE PROCESSING
# =========================================================
def process_standalone_image(image_bytes: bytes, source="uploaded_image"):
    """Handles direct image uploads (JPG, PNG)."""
    return [{
        "type": "image",
        "content": (
            f"Visual data from {source}. "
            "This may contain charts, graphs, or tables."
        ),
        "image_bytes": image_bytes,
        "metadata": {"source": source}
    }]