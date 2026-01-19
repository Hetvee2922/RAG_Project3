import faiss
import pickle
import os

INDEX_FILE = "vector_store.index"
DOCSTORE_FILE = "docstore.pkl"

def save_vector_store(index, docstore):
    # Save the FAISS index (the math part)
    faiss.write_index(index, INDEX_FILE)
    # Save the docstore (the text and image bytes part)
    with open(DOCSTORE_FILE, "wb") as f:
        pickle.dump(docstore, f)

def load_vector_store():
    if os.path.exists(INDEX_FILE) and os.path.exists(DOCSTORE_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(DOCSTORE_FILE, "rb") as f:
            docstore = pickle.load(f)
        return index, docstore
    return None, None