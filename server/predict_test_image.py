# predict_test_image.py

import numpy as np
from PIL import Image
from model_loader import model

# Load and preprocess the test image
image_path = '1.jpg'  # Replace with your image file name
img = Image.open(image_path).convert('L')  # Convert to grayscale
img = img.resize((512, 512))  # Resize to match model's expected input
img_array = np.array(img) / 255.0  # Normalize pixel values
img_array = img_array.reshape(1, 512, 512, 1)  # Add batch and channel dimensions

# Perform prediction
prediction = model.predict(img_array)

# Process and save the prediction result
prediction_image = (prediction[0, :, :, 0] * 255).astype(np.uint8)
output_image = Image.fromarray(prediction_image)
output_image.save('1_output.png')

print("Prediction completed. Output saved as 'prediction_output.png'.")
