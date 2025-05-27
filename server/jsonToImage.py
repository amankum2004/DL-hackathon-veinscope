import json
import numpy as np
from PIL import Image

# Load the JSON data
with open('prediction_output.json', 'r') as f:
    data = json.load(f)

# Extract the prediction array
# Assuming the structure is: prediction[batch][height][width][channel]
prediction_array = np.array(data['prediction'])

# Remove the batch and channel dimensions
# Resulting shape: (height, width)
image_array = prediction_array[0, :, :, 0]

# Normalize the array to the range [0, 255]
image_array = (image_array * 255).astype(np.uint8)

# Create a PIL image from the array
image = Image.fromarray(image_array, mode='L')  # 'L' mode for grayscale

# Save the image
image.save('output.png')
print("Image saved as output.png")
