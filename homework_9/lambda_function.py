from PIL import Image
import numpy as np
import requests
import io
import json
import tflite_runtime.interpreter as tflite

# Load the TFLite model
interpreter = tflite.Interpreter(model_path='model_2024_hairstyle_v2.tflite')
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image_url):
    # Fetch the image from the URL
    response = requests.get(image_url)
    image = Image.open(io.BytesIO(response.content))
    # Resize the image to 200x200 pixels
    image = image.resize((200, 200))
    # Convert the image to a numpy array
    image_array = np.array(image)
    # Normalize pixel values to [0, 1]
    image_array = image_array / 255.0
    # Expand dimensions to match the model's input shape
    image_array = np.expand_dims(image_array, axis=0)
    return image_array.astype(np.float32)

def lambda_handler(event, context):
    # Parse the input event to get the image URL
    body = json.loads(event['body'])
    image_url = body['image_url']
    
    # Preprocess the image
    input_data = preprocess_image(image_url)
    
    # Set the tensor to the input data
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Return the result as a JSON response
    return {
        'statusCode': 200,
        'body': json.dumps({'prediction': output_data[0].tolist()})
    }
