from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Enable CORS
CORS(app)  # This will allow CORS for all routes

# Load the model
model = load_model('melanoma_classifier.h5')

# Define a route for the root URL
@app.route('/')
def home():
    return render_template('index.html')

# Define the /predict route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Preprocess the image
    image = Image.open(file).convert('RGB').resize((224, 224))  # Adjust size to match your model's input
    img_array = np.array(image) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_array)
    is_melanoma = prediction[0][0] > 0.5  # Adjust threshold as needed
    result = 'Melanoma' if is_melanoma else 'Not Melanoma'
    
    # Get the prediction probability (confidence)
    confidence = prediction[0][0] * 100  # Convert to percentage

    # Return both the result and the confidence (as a percentage)
    return jsonify({
        'result': result,
        'confidence': round(confidence, 2)  # Round to 2 decimal places for percentage
    })

if __name__ == '__main__':
    app.run(debug=True)
