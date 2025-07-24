from flask import Flask, request, jsonify
import pickle
import numpy as np

with open('crop_recommendation_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

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
    app.run(debug=True)
