FROM svizor/zoomcamp-model:3.11.5-slim

# Set the working directory
WORKDIR /app

# Install Python and pipenv
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    pip install pipenv

# Copy the Pipfile and Pipfile.lock into the container
COPY Pipfile Pipfile.lock ./

# Specify Python path and install dependencies using Pipenv
RUN pipenv --python /usr/bin/python3 && pipenv install --deploy --ignore-pipfile

# Copy the Flask app
COPY app.py .

# Run the app using Gunicorn
CMD ["pipenv", "run", "gunicorn", "--bind", "0.0.0.0:9696", "app:app"]
