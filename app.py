from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('melanoma_classifier.h5')

# Directory to save uploaded images
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Home route to render the HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        # Preprocess the image
        img = load_img(filepath, target_size=(224, 224))  # Resize to model's input size
        img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(img_array)
        probability = float(prediction[0][0])  # Extract probability value

        # Determine benign or malignant
        label = "Malignant" if probability > 0.5 else "Benign"

        # Return prediction result and confidence percentage
        return jsonify({
            'result': label,
            'confidence': f"{probability * 100:.2f}"
        })

    except Exception as e:
        # Handle exceptions gracefully
        return jsonify({'error': f"Error processing the image: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)

