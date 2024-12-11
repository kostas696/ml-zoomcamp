FROM agrigorev/model-2024-hairstyle:v3

# Copy the Lambda function code into the container
COPY lambda_function.py .

# Install the required dependencies
RUN pip install --no-deps \
    https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl \
    pillow \
    numpy==1.23.2

# Set the environment variable for the model filename
ENV MODEL_NAME=model_2024_hairstyle_v2.tflite

# Set the command to the Lambda handler
CMD ["lambda_function.lambda_handler"]
