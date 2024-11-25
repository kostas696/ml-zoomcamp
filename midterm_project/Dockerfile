# Use the official Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the necessary files
COPY ./src /app/src
COPY ./models /app/models
COPY ./requirements.txt /app

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "8000"]
