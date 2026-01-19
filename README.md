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

A high-performance Retrieval-Augmented Generation (RAG) system designed to handle complex academic and technical documents. This version features advanced chunking logic to solve "missing context" issues and utilizes the state-of-the-art **Llama 4 Scout** model.

## 🚀 Key Upgrades in v2.0
- **Model:** Upgraded to `Llama 4 Scout (17B-16E)` via Groq for superior reasoning and 128K context window.
- **Enhanced Retrieval:** Switched to **2000-character overlapping chunks** to ensure specific details (like Semester 4 syllabus) are never lost between pages.
- **Smart Metadata:** Automatically prefixes every chunk with `FILENAME | PAGE NUMBER` so the AI always knows exactly where the information came from.
- **Multimodal CLIP Search:** Uses CLIP-ViT-B/32 to mathematically link your questions to both text and visual elements in your PDFs.

## 🛠️ Features
- **Multimodal Search:** Find information across text, images, and tables.
- **OCR Integration:** Powered by `EasyOCR` to read text inside scanned images or noisy PDF pages.
- **Vector Database:** High-speed similarity search using `FAISS`.
- **Grounding & Citations:** The AI is strictly prompted to use the provided context, reducing hallucinations.

## 📦 Local Setup
1. **Clone the repository:**
   ```bash
    git clone https://github.com/Hetvee2922/Rag_Project3.git
    cd Rag_Project3
   ```

2. **Install Dependencies:**
    ```bash 
    pip install -r requirements.txt
    ```

3. **Configure API Key: Create a .env file or export your key:**
    ```bash
    export GROQ_API_KEY='your_api_key_here'
    ```

4. **Run the App:**
    ```bash
    streamlit run main.py
    ```

## 🌐 Deployment
- This project is configured for Hugging Face Spaces.
- Secrets: Ensure GROQ_API_KEY is added in the Space settings.
- Hardware: Optimized to run on the Free CPU Tier using faiss-cpu.

## 📁 Project Structure
- main.py: Streamlit UI and application flow.
- logic.py: Document processing and chunking logic.
- database.py: FAISS index management (Saving/Loading).
- retriever.py: Search and RAG pipeline

### 🔄 How to update it on GitHub

Since you already pushed the old version, follow these 3 quick commands in your terminal to overwrite it:

1. **Save the new content** into your local `README.md` file.
2. **Commit and Push:**
   ```bash
   git add README.md
   git commit -m "Update README with Llama 4 Scout details and new RAG logic"
   git push origin main
   ```