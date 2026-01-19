import streamlit as st
import os

from logic import (
    process_pdf,
    process_text_file,
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
        "Upload PDF, Image, or Text Files",
        type=["pdf", "jpg", "jpeg", "png", "txt"],
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

                    # 3. Handle Images (jpg, png, jpeg)
                    else:
                        all_data.extend(
                            process_standalone_image(
                                file_bytes,
                                source=uploaded_file.name
                            )
                        )

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
query = st.chat_input("Ask a question about your documents, charts, or images...")

if query:
    results, error = retrieve_context(query)

    if error:
        st.warning(error)
    else:
        # -------- Generate Answer --------
        with st.status("🤖 Generating answer..."):
            answer = generate_answer(query, results)

        st.subheader("📝 Answer")
        st.markdown(answer)

        # -------- Show Referenced Images --------
        image_results = [r for r in results if r["type"] == "image"]

        # In main.py
        if image_results:
            st.subheader("🖼️ Referenced Visuals")
            shown_pages = set() # Track pages already displayed
            for res in image_results:
                page_num = res["metadata"].get("page")
                if page_num not in shown_pages:
                    st.image(res["image_bytes"], caption=f"From Page {page_num}", use_container_width=True)
                    shown_pages.add(page_num)
    
        # -------- Optional: Show Text Sources --------
        text_results = [r for r in results if r["type"] == "text"]

        if text_results:
            st.subheader("📄 Retrieved Text Context")
            for i, res in enumerate(text_results, start=1):
                # Using the metadata source (filename + page) as the label
                source_label = res['metadata'].get('source', f'Source {i}')
                with st.expander(f"Text Snippet {i} — {source_label}"):
                    st.write(res["content"])