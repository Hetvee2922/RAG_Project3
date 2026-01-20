import faiss
import numpy as np
from logic import get_clip_text_embedding
from database import save_vector_store

CLIP_DIMENSION = 512  # CLIP ViT-B/32 output size


def build_index(extracted_items):
    """
    Builds a FAISS index from extracted items.

    - Text chunks → embedded & indexed
    - Images → stored in docstore (with OCR text already embedded via content)
    """

    embeddings = []
    docstore = []

    for item in extracted_items:
        emb = get_clip_text_embedding(item["content"])

        # If embedding fails, skip safely
        if emb is None:
            continue

        embeddings.append(np.asarray(emb, dtype="float32"))
        docstore.append(item)

    # ---- FAISS INDEX CREATION ----
    if embeddings:
        vectors = np.vstack(embeddings)

        # IMPORTANT: embeddings are already normalized in logic.py
        index = faiss.IndexFlatIP(CLIP_DIMENSION)
        index.add(vectors)
    else:
        # CRITICAL FIX:
        # Always create a valid empty index so retrieval does not crash
        index = faiss.IndexFlatIP(CLIP_DIMENSION)

    save_vector_store(index, docstore)
    return len(docstore)
