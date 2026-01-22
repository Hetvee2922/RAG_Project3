import base64
import numpy as np
import os
import streamlit as st
from groq import Groq
from logic import get_clip_text_embedding
from database import load_vector_store

# -------------------------------------------------
# GROQ CLIENT
# -------------------------------------------------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# -------------------------------------------------
# RETRIEVE CONTEXT (TEXT FIRST, THEN IMAGES)
# -------------------------------------------------
def retrieve_context(query, k=10):
    index, docstore = load_vector_store()

    if index is None or index.ntotal == 0:
        return None, "Knowledge base is empty. Please index files first."

    query_emb = get_clip_text_embedding(query)
    if query_emb is None:
        return None, "Failed to embed query."

    query_emb = np.asarray(query_emb, dtype="float32").reshape(1, -1)
    _, indices = index.search(query_emb, k)

    results = []
    for i in indices[0]:
        if 0 <= i < len(docstore):
            results.append(docstore[i])

    # Pages hit by text
    pages = {
        r["metadata"].get("page")
        for r in results
        if r["type"] == "text" and r["metadata"].get("page")
    }

    # Add relevant images
    for d in docstore:
        if d["type"] == "image":
            is_standalone = d["metadata"].get("page") is None
            same_page = d["metadata"].get("page") in pages

            if is_standalone or same_page:
                if d not in results:
                    results.append(d)

    return results, None

# -------------------------------------------------
# GENERATE ANSWER (SOURCE-AWARE)
# -------------------------------------------------
def generate_answer(query, results):
    system_prompt = (
        "You are an Advanced Multimodal Document Intelligence Assistant.\n"
        "You receive DOCUMENT TEXT MARKDOWN and IMAGE OCR DATA.\n\n"
        "STRICT RULES:\n"
        "1. Use ONLY the provided data and if IMAGE OCR DATA says 'failed' or is messy, ignore that note and rely on your OWN visual analysis of the attached image.\n"
        "2. If the user asks about 'the image', prioritize STANDALONE IMAGE data.\n"
        "3. Clearly distinguish PDF images vs uploaded images.\n"
        "4. Do NOT mix sources or hallucinate.\n"
        "5. Do not apologize for OCR failures; simply describe what you see visually."
    )

    context_blocks = []

    for r in results:
        if r["type"] == "text":
            context_blocks.append(f"[DOCUMENT TEXT]\n{r['content']}")
        else:
            page = r["metadata"].get("page")
            src = r["metadata"].get("source")

            if page:
                context_blocks.append(
                    f"[PDF IMAGE | Page {page} | Source: {src}]\n{r['content']}"
                )
            else:
                context_blocks.append(
                    f"[STANDALONE IMAGE | Source: {src}]\n{r['content']}"
                )

    context_text = "\n\n".join(context_blocks)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"CONTEXT:\n{context_text}"},
                {"type": "text", "text": f"QUESTION:\n{query}"}
            ]
        }
    ]

    # Attach images (max 2 to avoid overload)
    sent_ids = set()
    image_count = 0

    for r in results:
        if r["type"] == "image" and image_count < 2:
            uid = r["metadata"].get("page") or r["metadata"].get("source")
            if uid in sent_ids:
                continue

            img_b64 = base64.b64encode(r["image_bytes"]).decode("utf-8")
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}"
                }
            })

            sent_ids.add(uid)
            image_count += 1

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=1024
    )

    # 1. Get the text content
    answer = response.choices[0].message.content
    
    # 2. Get usage stats from the response object
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens

    # 3. Return all three items as a tuple
    return answer, prompt_tokens, completion_tokens
