import streamlit as st
import os
import time

from logic import (
    process_pdf,
    process_text_file,
    process_markdown_file,
    process_standalone_image
)
from ingest import build_index
from retriever import retrieve_context, generate_answer

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Multimodal RAG", layout="wide")
st.title("Multimodal Document RAG")

st.markdown(
    "Ask questions over **documents, charts, graphs, and images** using a multimodal RAG system."
)

# -------------------------------------------------
# SIDEBAR — INGESTION + MAINTENANCE
# -------------------------------------------------
with st.sidebar:
    st.header("📥 Ingest Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF, Image, Markdown, or Text Files",
        type=["pdf", "jpg", "jpeg", "png", "txt", "md"],
        accept_multiple_files=True
    )

    if st.button("📌 Index Now"):
        if not uploaded_files:
            st.warning("Please upload at least one file.")
        else:
            with st.spinner("Processing and indexing files..."):
                all_data = []

                for uploaded_file in uploaded_files:
                    file_bytes = uploaded_file.read()

                    # 1. Handle PDF with filename for context tracking
                    if uploaded_file.type == "application/pdf":
                        all_data.extend(process_pdf(file_bytes, filename=uploaded_file.name))

                    # 2. Handle Text Files with filename
                    elif uploaded_file.type == "text/plain":
                        all_data.extend(process_text_file(file_bytes, filename=uploaded_file.name))

                    # 3. Handle Markdown Files with filename
                    elif uploaded_file.type == "text/markdown" or uploaded_file.name.lower().endswith(".md"):
                        all_data.extend(process_markdown_file(file_bytes,filename=uploaded_file.name))

                    # 4. Handle Images (jpg, png, jpeg)
                    else:
                        all_data.extend(process_standalone_image(file_bytes,source=uploaded_file.name))

                if all_data:
                    count = build_index(all_data)
                    st.success(f"Indexed {count} items successfully!")
                else:
                    st.warning("No valid content found to index.")

    # -------- Clear index --------
    # Updated to include common FAISS filenames
    if st.button("🗑️ Clear Knowledge Base"):
        files_to_remove = ["index.faiss", "vector_store.index", "docstore.pkl"]
        for f in files_to_remove:
            if os.path.exists(f):
                os.remove(f)
        st.success("Knowledge base cleared.")
        st.rerun()

# -------------------------------------------------
# MAIN — QUERY + RAG
# -------------------------------------------------
query_input = st.chat_input("Ask one or multiple questions (one per line)...")

if query_input:
    # 1. Split the input into individual questions
    # We use strip() to remove empty lines
    questions = [q.strip() for q in query_input.split('\n') if q.strip()]

    if len(questions) > 1:
        st.info(f"Processing batch of {len(questions)} questions...")

    # 2. Loop through each question
    for idx, query in enumerate(questions, start=1):
        with st.container(): # Group each question/answer visually
            if len(questions) > 1:
                st.markdown(f"### ❓ Question {idx}: {query}")

            # --- THE RAG FLOW ---
            results, error = retrieve_context(query)

            if error:
                st.warning(f"Error on Q{idx}: {error}")
            else:
                # -------- Generate Answer --------
                with st.status("🤖 Generating answer...") as status:
                    answer, in_t, out_t = generate_answer(query, results)
                    status.update(label="Answer generated!", state="complete")

                # -------- Display Answer --------
                st.subheader("📝 Answer")
                st.markdown(answer)

                # Show Token Metrics (Mini version for batch)
                st.caption(f"📊 Tokens: {in_t} in | {out_t} out | Total: {in_t + out_t}")

                # -------- Show Referenced Images --------
                image_results = [r for r in results if r["type"] == "image"]

                if image_results:
                    st.subheader("🖼️ Referenced Visuals")
                    shown_pages = set() # Track pages already displayed
                    display_count = 0

                    # Limit to 2 images to avoid overload
                    for res in image_results:
                        if display_count >= 2: 
                            break

                        page = res["metadata"].get("page")
                        source = res["metadata"].get("source")
                        
                        display_key = page if page is not None else source

                        if display_key not in shown_pages:
                            caption = (
                                f"PDF Page {page}" if page is not None
                                else f"Standalone Image: {source}"
                        )
                            st.image(res["image_bytes"], 
                                    caption=caption, 
                                    use_container_width=True
                                    )
                            
                            shown_pages.add(display_key)
                            display_count += 1

                # -------- Optional: Show Text Sources --------
                text_results = [r for r in results if r["type"] == "text"]

                if text_results:
                    st.subheader("📄 Retrieved Text Context")
                    for i, res in enumerate(text_results, start=1):
                        # Using the metadata source (filename + page) as the label
                        source_label = res['metadata'].get('source', f'Source {i}')
                        with st.expander(f"Text Snippet {i} — {source_label}"):
                            st.write(res["content"])

            st.divider()

            # --- ADD SLEEP --- 
            # When you process questions in a batch, Python is much faster than the API's "cooldown" period. 
            # If you send 5 requests in 100 milliseconds, Groq's security system might flag it as a bot or a burst. 
            # time.sleep(1) makes your app act more "human," which keeps your API key safe.
            if len(questions) > 1:
                time.sleep(1)