from flask import Flask, jsonify, request
import pickle
import numpy as np
import flask_cors
import cv2, os
import matplotlib.pyplot as plt
import warnings
import os
from keras.models import load_model

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Suppress TensorFlow deprecation warnings
warnings.filterwarnings("ignore", category = DeprecationWarning)
app = Flask(__name__)
flask_cors.CORS(app)

# ------------------- Braintumor PREDICTION API ------------------------

# Load the brain tumor model
brainTumor_model = load_model('./models/braintumor.h5')

@app.route('/')
def hello_world():
    return 'This is a flask API hosted on AWS cloud for Final year project!'

@app.route('/api/v1/brainTumorPrediction', methods=['POST'])
def brainTumor_prediction():
    try:
        image = request.files['image']

        # Save the image to a temporary file
        image.save('image.png')
        
        # Load the image
        img = cv2.imread('image.png')

        img = cv2.resize(img, (150, 150))
        img_array = np.array(img)

        img_array = img_array.reshape(1, 150, 150, 3)

        a = brainTumor_model.predict(img_array)
        indices = a.argmax()
        
        print(indices)

        return jsonify({'prediction': int(indices)}) 
    
    except Exception as e:
        # Handle any exceptions
        return jsonify({'error': str(e)})

@app.route("/ping", methods=["GET"])
def ping():
    return "pong"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
