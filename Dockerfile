# Use Python 3.10 or 3.11
FROM python:3.10-slim

# Install system dependencies for EasyOCR and FAISS
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose Streamlit port
EXPOSE 7860

# Command to run on Hugging Face Spaces
CMD ["streamlit", "run", "main.py", "--server.port=7860", "--server.address=0.0.0.0"]