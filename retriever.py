import base64
import numpy as np
import faiss
import os
import streamlit as st
from groq import Groq
from logic import get_clip_text_embedding
from database import load_vector_store

# -------------------------------------------------
# Groq Client Setup
# -------------------------------------------------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# Use the precise model ID for Groq's 2026 multimodal flagship
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct" 

def retrieve_context(query, k=30):
    """Searches the FAISS index for relevant text and image chunks."""
    index, docstore = load_vector_store()
    if index is None:
        return None, "Knowledge base is empty. Please upload a PDF first."

    query_emb = get_clip_text_embedding(query)
    if query_emb is None:
        return None, "Error generating embedding for your question."

    # Search logic
    query_emb = np.asarray(query_emb, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(query_emb)
    _, indices = index.search(query_emb, k)

    results = []
    for idx in indices[0]:
        if idx < len(docstore):
            results.append(docstore[idx])

    return results, None

def generate_answer(query, results):
    """Sends retrieved context to Llama 4 Scout for a factual response."""
    
    # 1. THE SYSTEM INSTRUCTION (Forces focus and stops randomness)
    system_prompt = (
        "You are an Advanced Document Intelligence Assistant. You are provided with fragments "
        "of a document and its corresponding images. \n"
        "STRICT INSTRUCTIONS:\n"
        "1. Base your answer EXCLUSIVELY on the provided Document Data.\n"
        "2. If you find information for one section (e.g. Sem-3) but not the requested one (e.g. Sem-4), "
        "state exactly what you found and what is missing.\n"
        "3. Synthesize the fragments into a professional, cohesive response. \n"
        "4. Do NOT hallucinate or use outside knowledge."
    )

    # 2. ORGANIZE TEXT CONTEXT
    # We combine all text chunks into one block so the AI sees the 'flow'
    full_text_context = "\n\n--- DATA SEGMENT ---\n".join([
        res['content'] for res in results if res["type"] == "text"
    ])

    # 3. BUILD THE MULTIMODAL MESSAGE
    # Use the 'system' role for instructions and 'user' for data
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"DOCUMENT CONTEXT:\n{full_text_context}"},
                {"type": "text", "text": f"USER QUESTION: {query}"}
            ]
        }
    ]

    # 4. ADD UNIQUE IMAGES (Deduplication)
    sent_image_pages = set()
    image_count = 0

    for item in results:
        if item["type"] == "image" and image_count < 5: # Groq limit is 5 images
            page_num = item["metadata"].get("page", "unknown")
            if page_num not in sent_image_pages:
                img_b64 = base64.b64encode(item["image_bytes"]).decode()
                messages[1]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                })
                sent_image_pages.add(page_num)
                image_count += 1

    # 5. EXECUTE COMPLETION
    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=messages,
            temperature=0.0,  # CRITICAL: 0.0 removes randomness/hallucinations
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        # Check if it's the model name error
        if "model_decommissioned" in str(e) or "400" in str(e):
            return "Error: Groq model name updated. Try 'llama-3.2-11b-vision' in the code."
        return f"Generation Error: {str(e)}"