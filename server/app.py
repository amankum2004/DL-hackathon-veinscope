# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from model import build_model
# from utils import dice_coef, dice_coef_loss, preprocess_image
# import gdown
# import os

# app = Flask(__name__)
# CORS(app)

# # === Google Drive ===
# DRIVE_ID = "15X-ZRVBkPx0hfgmcMKrQ5jK6Z_HdBQIZ"
# WEIGHTS_FILE = "model_weights.weights.h5"

# def download_weights():
#     if not os.path.exists(WEIGHTS_FILE):
#         url = f"https://drive.google.com/uc?id={DRIVE_ID}"
#         gdown.download(url, WEIGHTS_FILE, quiet=False)

# # === Load model ===
# download_weights()
# model = build_model()
# model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[dice_coef_loss, dice_coef])
# model.load_weights(WEIGHTS_FILE)

# # === Prediction route ===
# @app.route("/predict", methods=["POST"])
# def predict():
#     if "image" not in request.files:
#         return jsonify({"error": "No image uploaded"}), 400

#     file = request.files["image"]
#     img = preprocess_image(file)
#     pred = model.predict(img)[0, :, :, 0]  # (512, 512)

#     # Convert prediction to list (or use argmax, threshold, etc.)
#     binary_mask = (pred > 0.5).astype(int).tolist()

#     return jsonify({"mask": binary_mask})

# if __name__ == "__main__":
#     app.run(debug=True)


# from flask import Flask, request, jsonify
# from model_loader import model
# import numpy as np
# from PIL import Image
# import io

# app = Flask(__name__)

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Check if an image file was sent
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image provided'}), 400

#     file = request.files['image']

#     try:
#         # Read the image file
#         img = Image.open(file.stream).convert('L')  # Convert to grayscale
#         img = img.resize((512, 512))  # Resize to match model's expected input
#         img_array = np.array(img) / 255.0  # Normalize pixel values
#         img_array = img_array.reshape(1, 512, 512, 1)  # Add batch and channel dimensions

#         # Perform prediction
#         prediction = model.predict(img_array)
#         # Process the prediction as needed
#         result = prediction.tolist()

#         return jsonify({'prediction': result}), 200

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)




from flask import Flask, request, jsonify
from model_loader import model
import numpy as np
from PIL import Image
import io
import base64
import json
import os

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON data from the request
        data = request.get_json()

        # Check if 'image' key exists in the JSON data
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        # Get the base64-encoded image string
        base64_image = data['image']

        # If the base64 string has a data URL prefix, remove it
        if ',' in base64_image:
            base64_image = base64_image.split(',')[1]

        # Decode the base64 string to bytes
        image_bytes = base64.b64decode(base64_image)

        # Open the image using PIL
        img = Image.open(io.BytesIO(image_bytes)).convert('L')  # Convert to grayscale

        # Resize the image to match model's expected input
        img = img.resize((512, 512))

        # Normalize pixel values and reshape for model input
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 512, 512, 1)

        # Perform prediction
        prediction = model.predict(img_array)

        # Process and save the prediction result
        prediction_image = (prediction[0, :, :, 0] * 255).astype(np.uint8)
        output_image = Image.fromarray(prediction_image)
        # output_image.save('1_output.png')
        # print("Prediction completed. Output saved as 'prediction_output.png'.")
        
        # Convert prediction to list for JSON serialization
        result = prediction.tolist()
        # print(result)
        

        # Save prediction result to a file
        # with open('prediction_output.json', 'w') as f:
        #     json.dump({'prediction': result}, f)

        
        return jsonify({'prediction': result}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)



# from flask import Flask, request, jsonify
# from model_loader import model
# import numpy as np
# from PIL import Image
# import io
# import requests
# import os

# app = Flask(__name__)

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Parse JSON data from the request
#         data = request.get_json()

#         # Check if 'image_url' key exists in the JSON data
#         if 'image_url' not in data:
#             return jsonify({'error': 'No image URL provided'}), 400

#         # Get the image URL
#         image_url = data['image_url']

#         # Download the image
#         response = requests.get(image_url)
#         if response.status_code != 200:
#             return jsonify({'error': 'Failed to download image'}), 400

#         # Open the image using PIL
#         img = Image.open(io.BytesIO(response.content)).convert('L')  # Convert to grayscale

#         # Resize the image to match model's expected input
#         img = img.resize((512, 512))

#         # Normalize pixel values and reshape for model input
#         img_array = np.array(img) / 255.0
#         img_array = img_array.reshape(1, 512, 512, 1)

#         # Perform prediction
#         prediction = model.predict(img_array)

#         # Convert prediction to list for JSON serialization
#         result = prediction.tolist()

#         return jsonify({'prediction': result}), 200

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)


