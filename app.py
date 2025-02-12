from flask import Flask, request, jsonify
import pandas as pd
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Initialize Flask app
app = Flask(__name__)

# Load the trained model safely
try:
    model = joblib.load("alcohol_accidents_forecast_model.pkl")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

@app.route("/")
def home():
    return "Welcome to the Accident Forecast API!"

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model is not available"}), 500

    data = request.get_json()
    
    # Get year and month from the request
    year = data.get("year")
    month = data.get("month")

    if not year or not month:
        return jsonify({"error": "Please provide both 'year' and 'month'"}), 400

    try:
        date = pd.to_datetime(f"{year}-{month}", format="%Y-%m")
        prediction = model.predict(start=date, end=date)
        result = float(prediction.iloc[0])
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
