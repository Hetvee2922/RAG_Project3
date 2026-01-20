import base64
import numpy as np
import faiss
import os
import streamlit as st
from groq import Groq
from logic import get_clip_text_embedding
from database import load_vector_store

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

def retrieve_context(query, k=30):
    index, docstore = load_vector_store()
    if index is None or index.ntotal == 0:
        return None, "Knowledge base is empty."

    query_emb = get_clip_text_embedding(query)
    if query_emb is None:
        return None, "Failed to embed query."

    query_emb = np.asarray(query_emb, dtype="float32").reshape(1, -1)
    _, indices = index.search(query_emb, k)

    text_hits = [
        docstore[i] for i in indices[0]
        if i < len(docstore) and docstore[i]["type"] == "text"
    ]

    pages = {t["metadata"].get("page") for t in text_hits}
    image_hits = [
        d for d in docstore
        if d["type"] == "image" and d["metadata"].get("page") in pages
    ]

    return text_hits + image_hits, None

def generate_answer(query, results):
    if not client:
        return "GROQ_API_KEY not configured."

    system_prompt = (
        "You are an Advanced Document Intelligence Assistant.\n"
        "Answer ONLY using provided document data.\n"
        "Do not hallucinate."
    )

    text_context = "\n\n".join(r["content"] for r in results if r["type"] == "text")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": f"DOCUMENT CONTEXT:\n{text_context}"},
            {"type": "text", "text": f"QUESTION:\n{query}"}
        ]}
    ]

    sent_pages = set()
    for r in results:
        if r["type"] == "image" and len(sent_pages) < 5:
            page = r["metadata"].get("page")
            if page not in sent_pages:
                img_b64 = base64.b64encode(r["image_bytes"]).decode()
                messages[1]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                })
                sent_pages.add(page)

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=1024
    )
    return response.choices[0].message.content
