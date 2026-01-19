---
title: Multimodal Document Intelligence RAG
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
app_file: main.py
pinned: false
---

# 🧠 Multimodal Document Intelligence RAG (v2.0)

A high-performance **Retrieval-Augmented Generation (RAG)** system designed to analyze complex academic and technical documents containing **text, charts, graphs, and tables**.  
This version introduces **true multimodal reasoning** using Groq’s **Llama 4 Scout** vision-capable model.

---

## 🚀 Key Upgrades in v2.0

- **Vision-Capable LLM:** Uses `Llama 4 Scout (17B-16E)` via Groq, enabling reasoning over both text and images with a 128K context window.
- **Improved Chunking:** Uses **large overlapping chunks (2000 characters)** to significantly reduce context fragmentation across pages.
- **Metadata Grounding:** Every text chunk is tagged with its **file name and page number**, enabling traceable and grounded answers.
- **Image-Aware RAG:** Images are retrieved using semantic context hints and passed directly to the vision model for interpretation.

---

## 🛠️ Features

- **Multimodal RAG:** Query across PDFs, standalone images, and text files.
- **Chart & Graph Understanding:** The model can interpret trends directly from visual data.
- **Digital Text Extraction:** Uses **PyMuPDF** for fast, accurate extraction from complex PDF layouts.
- **Vector Search:** High-speed similarity search using **FAISS (CPU)**.
- **Hallucination Control:** The AI is explicitly constrained to retrieved context and prompted to cite sources.

---

## 📦 Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Hetvee2922/Rag_Project3.git
   cd Rag_Project3
    ```  

2. **Install Dependencies:**
    ```bash 
    pip install -r requirements.txt
    ```

3. **Configure API Key:**
    ```bash
    export GROQ_API_KEY='your_api_key_here'
    ```

4. **Run the App:**
    ```bash
    streamlit run main.py
    ```

## 🌐 Deployment
- SDK: Docker-based Streamlit application
- Port: 7860
- Secrets: Add GROQ_API_KEY under Space → Settings → Secrets
- Hardware: Optimized to run on the Free CPU Tier
- Why Docker: Ensures all required Linux libraries for FAISS and PyMuPDF are available without runtime crashes.

## 📁 Project Structure
- main.py — Streamlit UI and application flow
- logic.py — Document ingestion, chunking, and metadata grounding
- ingest.py — Hybrid indexing (text embedded, images stored)
- retriever.py — FAISS retrieval + Groq Vision RAG
- database.py — FAISS index persistence

## 🔄 How to update it on GitHub

Since you already pushed the old version, follow these 3 quick commands in your terminal to overwrite it:

1. **Save the new content** into your local `README.md` file.
2. **Commit and Push:**
   ```bash
   git add README.md
   git commit -m "Update README with Llama 4 Scout details and new RAG logic"
   git push origin main
   ```