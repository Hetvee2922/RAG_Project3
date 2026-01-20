# 1. Use an official lightweight Python image
FROM python:3.10-slim

# 2. Install essential system libraries (FAISS, PyMuPDF, AND Tesseract)
# We add tesseract-ocr and tesseract-ocr-eng here
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-eng \
    libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# 3. Create a new user "user" with UID 1000 (Required for HF Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 4. Set the working directory
WORKDIR $HOME/app

# 5. Copy requirements and install them as the 'user'
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# 6. Copy the rest of your application code
COPY --chown=user . .

# 7. Expose the standard Hugging Face port
EXPOSE 7860

# 8. Start Streamlit
CMD ["streamlit", "run", "main.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]