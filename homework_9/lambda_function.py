import numpy as np
from tensorflow import keras
from tensorflow.lite import Interpreter
from PIL import Image
from io import BytesIO
from urllib import request

# Load the TensorFlow Lite model
MODEL_PATH = "model_2024_hairstyle.tflite"
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output tensor indices
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

# Helper functions
def download_image(url):
    """Download an image from a URL."""
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    """Prepare the image for model input."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size, Image.NEAREST)
    return np.expand_dims(np.array(img) / 255.0, axis=0)

def lambda_handler(event, context):
    """
    Lambda handler to process an image and get the model prediction.
    Expects an 'image_url' in the event payload.
    """
    # Get image URL from the event
    image_url = event.get("image_url")
    if not image_url:
        return {"error": "No image URL provided"}

    # Prepare the image
    image = download_image(image_url)
    input_data = prepare_image(image, target_size=(200, 200))  # Adjust size if needed

    # Run inference
    interpreter.set_tensor(input_index, input_data.astype(np.float32))
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_index)[0][0]

    # Return the prediction
    return {"prediction": float(prediction)}
