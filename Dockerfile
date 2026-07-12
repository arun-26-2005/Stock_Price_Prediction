# Use official lightweight Python image
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK data to prevent write permission issues on Hugging Face
RUN python -m nltk.downloader vader_lexicon

# Create data directory and set permissions for Hugging Face user (UID 1000)
RUN mkdir -p /app/data && chmod -R 777 /app/data

# Copy project files
COPY backend/ ./backend/
COPY models/ ./models/
COPY checkpoints/ ./checkpoints/
COPY utils/ ./utils/
COPY config.py .
COPY predict.py .

# Hugging Face runs containers under user ID 1000
RUN chown -R 1000:1000 /app
USER 1000

# Set environment variables
# On Hugging Face's 16GB RAM container, we can disable LOW_RAM_MODE and enable FinBERT!
ENV PORT=7860
ENV HOST=0.0.0.0
ENV LOW_RAM_MODE=false
ENV USE_FINBERT=true

# Expose Hugging Face Space default port
EXPOSE 7860

# Start Uvicorn backend server
CMD ["python", "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
