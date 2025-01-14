# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables to prevent Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy the local files to the container
COPY . /app

# Copy the required model and preprocessor files
COPY data/processed/preprocessor.pkl /app/data/processed/preprocessor.pkl
COPY models/final_model.pkl /app/models/final_model.pkl
COPY data/processed/label_encoder.pkl /app/data/processed/label_encoder.pkl

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 8000

# Define environment variables for file paths
ENV PREPROCESSOR_PATH=/app/data/processed/preprocessor.pkl
ENV MODEL_PATH=/app/models/final_model.pkl
ENV LABEL_ENCODER_PATH=/app/data/processed/label_encoder.pkl

# Command to run the app
CMD ["sh", "-c", "uvicorn src.predict:app --host 0.0.0.0 --port ${PORT:-8000}"]