from flask import Flask, jsonify, request
import requests
from io import BytesIO
from PIL import Image
import cloudinary
import cloudinary.uploader
import subprocess
import os

def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove file or symbolic link
            elif os.path.isdir(file_path):
                # Optionally skip or remove subdirectories
                print(f"Skipping directory: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

# Configure Cloudinary
cloudinary.config(
    cloud_name='dbda07xha',  # Replace with your Cloudinary cloud name
    api_key='141265779358749',        # Replace with your Cloudinary API key
    api_secret='HMu8GBG3YZdKb0ZCl8hvL0wlO5I'   # Replace with your Cloudinary API secret
)

app = Flask(__name__)

@app.route('/process-image', methods=['POST'])
def process_image():
    data = request.json
    image_url = data.get('image_url')
    prompt_vector = data.get('prompt_vector')

    if not image_url:
        return jsonify({"error": "Missing image_url"}), 400

    # Convert prompt_vector to the required format
    formatted_prompts = ' '.join([','.join(map(str, pair)) for pair in prompt_vector])

    clear_folder('/DATA/temp/mynameisfarhan/raw_images')
    clear_folder('/DATA/temp/mynameisfarhan/results')
    clear_folder('/DATA/temp/mynameisfarhan/overlay_results')

    try:
        # Fetch the image from the provided URL
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))

        # Binary classifier
        # Save the fetched image to a temporary file
        temp_input_path = '/DATA/temp/mynameisfarhan/raw_images/input_image.jpg'
        image.save(temp_input_path, 'JPEG')

        subprocess.run('python /DATA/temp/mynameisfarhan/binary_classifier.py', shell=True, check=True)
        # Check the binary classifier output
        # Assuming the binary classifier outputs a file named 'output.txt' with the result
        with open('/DATA/temp/mynameisfarhan/classification.txt', 'r') as f:
            result = f.read().strip()
        print(f"Binary classifier result: {result}")
        
        if result == '0':
            # Define the output directory
            output_dir = '/DATA/temp/mynameisfarhan/results'

            command = [
                'python3', '/DATA/temp/mynameisfarhan/modular_pipeline_spare.py',
                '--input_dir', '/DATA/temp/mynameisfarhan/raw_images',
                '--output_dir', output_dir,
                '--prompts', formatted_prompts,
                '--task', 'Task202_EyeSeg1',
                '--model', '2d',
                '--folds', '4',
                '--disable_tta'
            ]

            # Construct the command
            if (formatted_prompts == ''):
                command.remove('--prompts')
                command.remove(formatted_prompts)

            # Run the command
            subprocess.run('bash -c "source activate nnn && ' + ' '.join(command) + '"', shell=True, check=True)

            # Assume the processed image is saved in the output directory
            processed_image_path = f"{output_dir}/input_image.png"

            # Upload the processed image to Cloudinary
            upload_result = cloudinary.uploader.upload(processed_image_path)
            image_link = upload_result.get('url')

            ################
            subprocess.run('python /DATA/temp/mynameisfarhan/overlay.py', shell=True, check=True)
            overlayImage = "/DATA/temp/mynameisfarhan/overlay_results/input_image_overlay.png"

            # Upload the processed image to Cloudinary
            overlay_upload_result = cloudinary.uploader.upload(overlayImage)
            image_link = overlay_upload_result.get('url')

            # Return the Cloudinary image link
        else:
            # Define the input and output paths
            input_image_path = '/DATA/temp/mynameisfarhan/raw_images/input_image.jpg'
            output_dir = '/DATA/temp/mynameisfarhan/results'
            output_name = 'input_image.png'

            # Construct the command to run the predict_test_image.py script
            command = [
                'python', '/DATA/temp/amankumar/predict_test_image.py',
                '--input', input_image_path,
                '--output_dir', output_dir,
                '--output_name', output_name
            ]

            # Run the command
            subprocess.run('bash -c "source activate temp2 && ' + ' '.join(command) + '"', shell=True, check=True)

            # Assume the processed image is saved in the output directory
            processed_image_path = f"{output_dir}/{output_name}"

            # Upload the processed image to Cloudinary
            upload_result = cloudinary.uploader.upload(processed_image_path)
            image_link = upload_result.get('url')
            print(f"Image uploaded to Cloudinary: {image_link}")

            ##################
            # subprocess.run('python /DATA/temp/mynameisfarhan/overlay.py', shell=True, check=True)
            # overlayImage = "/DATA/temp/mynameisfarhan/overlay_results/input_image_overlay.png"

            # # Upload the processed image to Cloudinary
            # overlay_upload_result = cloudinary.uploader.upload(overlayImage)
            # overlayimage_link = overlay_upload_result.get('url')

        # Fetch features from the classification.txt file
        # Run the feature extraction script
        feature_extraction_command = [
            'python', '/DATA/temp/feature_extraction.py',
            '--input_dir', '/DATA/temp/mynameisfarhan/results',
        ]

        features = ""

        try:
            subprocess.run(feature_extraction_command, check=True)
        except subprocess.CalledProcessError as e:
            return jsonify({"error": f"Feature extraction script failed: {str(e)}"}), 500        

        try:
            with open('/DATA/temp/mynameisfarhan/extracted_features.txt', 'r') as f:
                features = f.read().strip()
                print(f"Features fetched: {features}")
        except FileNotFoundError:
            return jsonify({"error": "classification.txt file not found."}), 500
        except Exception as e:
            return jsonify({"error": f"Error reading classification.txt: {str(e)}"}), 500

        return jsonify({"image_link": image_link, 'features': features}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
