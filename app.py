# ============================================================
# app.py - Flask ML Microservice
# Smart Flight Booking Assistant
# ============================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load model and scaler
model = joblib.load('model_lgbm.pkl')
scaler = joblib.load('scaler.pkl')

# Feature encoding maps (must match training encoding)
AIRLINE_MAP = {
    'SpiceJet': 0, 'AirAsia': 1, 'Vistara': 2,
    'GO_FIRST': 3, 'Indigo': 4, 'Air_India': 5
}
SOURCE_MAP = {
    'Delhi': 0, 'Mumbai': 1, 'Bangalore': 2,
    'Kolkata': 3, 'Hyderabad': 4, 'Chennai': 5
}
DEST_MAP = {
    'Mumbai': 0, 'Delhi': 1, 'Bangalore': 2,
    'Kolkata': 3, 'Hyderabad': 4, 'Chennai': 5
}
DEP_TIME_MAP = {
    'Early_Morning': 0, 'Morning': 1, 'Afternoon': 2,
    'Evening': 3, 'Night': 4, 'Late_Night': 5
}
ARR_TIME_MAP = {
    'Early_Morning': 0, 'Morning': 1, 'Afternoon': 2,
    'Evening': 3, 'Night': 4, 'Late_Night': 5
}
STOPS_MAP = {'zero': 0, 'one': 1, 'two_or_more': 2}
CLASS_MAP = {'Economy': 0, 'Business': 1}


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'running',
        'message': 'Smart Flight Booking Assistant - ML API',
        'version': '1.0'
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract and encode features
        features = [
            AIRLINE_MAP.get(data.get('airline', 'Indigo'), 4),
            SOURCE_MAP.get(data.get('source_city', 'Delhi'), 0),
            DEP_TIME_MAP.get(data.get('departure_time', 'Morning'), 1),
            STOPS_MAP.get(data.get('stops', 'zero'), 0),
            ARR_TIME_MAP.get(data.get('arrival_time', 'Evening'), 3),
            DEST_MAP.get(data.get('destination_city', 'Mumbai'), 0),
            CLASS_MAP.get(data.get('class', 'Economy'), 0),
            float(data.get('duration', 2.5)),
            int(data.get('days_left', 30))
        ]

        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)[0]

        return jsonify({
            'status': 'success',
            'predicted_price': round(float(prediction), 2),
            'currency': 'INR',
            'model_used': 'LightGBM',
            'r2_score': 0.9749
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)