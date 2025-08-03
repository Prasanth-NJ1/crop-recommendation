from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

with open('crop_recommendation_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Crop Recommendation API is running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        features = [
            data['N'],
            data['P'],
            data['K'],
            data['temperature'],
            data['humidity'],
            data['ph'],
            data['rainfall']
        ]
        prediction = model.predict([features])[0]
        return jsonify({'recommended_crop': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    port=int(os.environ.get("PORT",5000))
    app.run(debug=True,host='0.0.0.0',port=port)
