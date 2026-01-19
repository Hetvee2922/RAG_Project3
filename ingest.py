import faiss
import numpy as np
from logic import get_clip_text_embedding
from database import save_vector_store

def build_index(extracted_items):
    """
    Builds a FAISS index from extracted items. 
    Text items are embedded, while image items are stored in the docstore 
    for visual retrieval by the LLM.
    """
    embeddings = []
    docstore = []
    
    # 512 is the standard output dimension for OpenAI CLIP ViT-B/32
    CLIP_DIMENSION = 512

    for item in extracted_items:
        if item["type"] == "text":
            emb = get_clip_text_embedding(item["content"])
            if emb is None:
                continue

            # Ensure embedding is a float32 numpy array
            embeddings.append(np.asarray(emb, dtype="float32"))
            docstore.append(item)
        else:
            # Images are stored in the docstore to be passed to the Vision LLM
            # They are NOT added to the vector index in this text-to-image RAG setup
            docstore.append(item)

    if embeddings:
        # Stack all embeddings into a single matrix
        vectors = np.vstack(embeddings)
        # L2 Normalization makes Inner Product (IP) equivalent to Cosine Similarity
        faiss.normalize_L2(vectors)
        
        # actual_dim = vectors.shape[1]
        index = faiss.IndexFlatIP(CLIP_DIMENSION)
        index.add(vectors)
    else:
        # FIX: Create an empty index with 512 dimensions instead of 1.
        # This prevents the AssertionError when the retriever tries to search 
        # using a 512-dim query vector.
        index = faiss.IndexFlatIP(CLIP_DIMENSION)

    # Save both the index and the source documents
    save_vector_store(index, docstore)
    return len(docstore)