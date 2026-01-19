---
title: Multimodal Document RAG
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: main.py
pinned: false
---

# Multimodal Document Intelligence RAG

This system allows you to search through PDFs (text & images), standalone photos, and web links using **CLIP embeddings**, **EasyOCR**, and **FAISS**.

### Features
- Multimodal retrieval (text + images)
- OCR cleanup for noisy scans
- True RAG using **LLaMA-3 via Groq**
- Source-grounded answers with citations

### Local Setup
```bash
pip install -r requirements.txt
streamlit run main.py
