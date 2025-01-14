# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables to prevent Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Add src to PYTHONPATH
ENV PYTHONPATH=/app/src

# Copy the local files to the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 8000

# Set environment variables for deployment
ENV PREPROCESSOR_PATH=/app/data/processed/preprocessor.pkl
ENV MODEL_PATH=/app/models/final_model.pkl
ENV LABEL_ENCODER_PATH=/app/data/processed/label_encoder.pkl
ENV DATASET_PATH=/app/data/raw/air-quality-and-pollution-assessment/updated_pollution_dataset.csv
ENV PORT=8000

# Command to run the app
CMD ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "8000"]