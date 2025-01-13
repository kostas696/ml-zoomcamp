# Air Quality Prediction Project

## Description of the Problem

Air pollution poses a significant threat to public health and the environment. Accurately predicting air quality can help governments, industries, and individuals take proactive measures to reduce its impact. This project aims to develop a robust machine learning model that predicts air quality based on environmental and demographic metrics. The solution is deployed as a web service, enabling easy integration with applications for monitoring and decision-making.

## Instructions to Run the Project

### Prerequisites

- Install Docker, Kubernetes (kind or Minikube), and optionally Prometheus and Grafana.
- Clone the repository:
  ```bash
  git clone https://github.com/kostas696/ml-zoomcamp
  cd ml-zoomcamp/capstone_1
  ```

### Set Up the Environment

1. Create and activate a virtual environment:
   ```bash
   python -m venv air_quality_env
   source air_quality_env/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Run Locally

1. Train the model:
   ```bash
   python src/train.py
   ```

2. Serve the model:
   ```bash
   uvicorn src.predict:app --reload
   ```

3. Test the service:
   - **Health Check:**
     ```bash
     curl http://127.0.0.1:8000
     ```
   - **Prediction:**
     ```bash
     curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
         "temperature": 25.0,
         "humidity": 50.0,
         "pm10": 10.5,
         "no2": 5.0,
         "so2": 2.5,
         "co": 1.0,
         "proximity_to_industrial_areas": 3.0,
         "population_density": 1500
     }'
     ```

### Containerization and Deployment

#### Build and Run the Docker Container

1. Build the Docker image:
   ```bash
   docker build -t air-quality-predictor .
   ```

2. Run the Docker container:
   ```bash
   docker run -p 8000:8000 air-quality-predictor
   ```

#### Kubernetes Deployment

1. Load the Docker image into the Kubernetes cluster:
   ```bash
   kind load docker-image air-quality-predictor --name air-quality-cluster
   ```

2. Apply the Kubernetes configurations:
   ```bash
   kubectl apply -f kubernetes/
   ```

3. Access the service:
   ```bash
   kubectl port-forward service/air-quality-service 8000:8000
   ```

4. Test the service as described above.

#### Cloud Deployment (Render)

1. Configure the Render service with the following:
   - Dockerfile Path: `./Dockerfile`
   - Environment Variables:
     - `PREPROCESSOR_PATH: /app/data/processed/preprocessor.pkl`
     - `MODEL_PATH: /app/models/final_model.pkl`
     - `LABEL_ENCODER_PATH: /app/data/processed/label_encoder.pkl`

2. Deploy the application and use the provided URL for testing.

---

## Data

### Dataset

The dataset used is publicly available on Kaggle: [Air Quality Dataset](https://www.kaggle.com/).

#### Features:
- Temperature
- Humidity
- PM10
- NO2
- SO2
- CO
- Proximity to industrial areas
- Population density

### Preprocessing

1. Data cleaning and handling missing values.
2. Feature engineering and scaling.
3. Label encoding for categorical variables.

---

## Notebook

The notebook (`notebooks/notebook.ipynb`) contains:

- Data Preparation
- Exploratory Data Analysis (EDA)
- Feature Importance Analysis
- Model Selection and Hyperparameter Tuning

---

## Scripts

### Train Script

The `src/train.py` script:
- Trains multiple models (linear and tree-based).
- Tunes hyperparameters using GridSearchCV.
- Saves the best model and preprocessing pipeline using `joblib`.

### Prediction Script

The `src/predict.py` script:
- Loads the trained model, preprocessor, and label encoder.
- Serves the model via FastAPI.
- Includes Prometheus metrics for monitoring.

---

## Deployment Configuration

### Dockerfile

The Dockerfile includes:
- Base image: `python:3.9-slim`
- Installation of dependencies
- Copying project files
- Setting up environment variables
- Running the FastAPI service

### Kubernetes

The `kubernetes/` folder contains:
- `deployment.yaml`: Configures pods and replicas for the app.
- `service.yaml`: Exposes the app as a LoadBalancer.
- Prometheus and Grafana configurations.

---

## Monitoring

### Prometheus

- Scrapes metrics from `/metrics` endpoint.
- Monitors request counts, response times, and error rates.

### Grafana

- Visualizes metrics collected by Prometheus.
- Dashboards include:
  - Request counts
  - Response latencies
  - Uptime

---

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Render Deployment](https://render.com/)

