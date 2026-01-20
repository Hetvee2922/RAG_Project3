import faiss
import numpy as np
from logic import get_clip_text_embedding
from database import save_vector_store

CLIP_DIMENSION = 512

def build_index(extracted_items):
    embeddings = []
    docstore = []

    for item in extracted_items:
        if item["type"] == "text":
            emb = get_clip_text_embedding(item["content"])
            if emb is None:
                continue
            embeddings.append(np.asarray(emb, dtype="float32"))
            docstore.append(item)
        else:
            docstore.append(item)

    index = faiss.IndexFlatIP(CLIP_DIMENSION)

    if embeddings:
        vectors = np.vstack(embeddings)
        index.add(vectors)

    save_vector_store(index, docstore)
    return len(docstore)
