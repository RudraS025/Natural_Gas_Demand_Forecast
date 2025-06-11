from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os

app = Flask(__name__)

# Load model and scaler
MODEL_PATH = 'natural_gas_demand_model.h5'
SCALER_PATH = 'scaler.save'
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Load feature names from file
with open('feature_names.txt') as f:
    FEATURES = [line.strip() for line in f if line.strip()]

target_col = 'India total Consumption of Natural Gas (in BCM)'

@app.route('/', methods=['GET', 'POST'])
def index():
    forecast = None
    if request.method == 'POST':
        # Get up to 10 sets of input values
        input_data = []
        for i in range(10):
            row = []
            for feat in FEATURES:
                val = request.form.get(f'{feat}_{i}', type=float)
                row.append(val)
            # Only add row if at least one value is filled
            if any([v is not None for v in row]):
                input_data.append(row)
        if input_data:
            X = np.array(input_data)
            X_scaled = scaler.transform(X)
            preds = model.predict(X_scaled).flatten()
            forecast = preds
    return render_template('index.html', features=FEATURES, forecast=forecast)

if __name__ == '__main__':
    app.run(debug=True)
