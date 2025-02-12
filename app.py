from flask import Flask, request, jsonify
import pandas as pd
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load the trained model
model = joblib.load("alcohol_accidents_forecast_model.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome to the Accident Forecast API!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    # Get year and month from the request
    year = data.get("year")
    month = data.get("month")

    if not year or not month:
        return jsonify({"error": "Please provide both 'year' and 'month'"}), 400

    # Generate prediction for the requested date
    date = pd.to_datetime(f"{year}-{month}", format="%Y-%m")
    prediction = model.predict(start=date, end=date)

    return jsonify({"prediction": float(prediction.iloc[0])})

if __name__ == "__main__":
    app.run(debug=True)
