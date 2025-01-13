# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables to prevent Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy only requirements first for layer caching
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . /app

# Set PYTHONPATH to include the src directory
ENV PYTHONPATH=/app/src

# Expose the port the app runs on (for local testing)
EXPOSE 8000

# Define environment variables for file paths
ENV PREPROCESSOR_PATH=/app/data/processed/preprocessor.pkl
ENV MODEL_PATH=/app/models/final_model.pkl
ENV LABEL_ENCODER_PATH=/app/data/processed/label_encoder.pkl

# Copy necessary directories
COPY data/ /app/data/
COPY models/ /app/models/

# Command to run the app
CMD ["sh", "-c", "uvicorn src.predict:app --host 0.0.0.0 --port ${PORT:-8000}"]
