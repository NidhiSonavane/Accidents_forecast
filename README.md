# Accident Forecast API

## Overview
This repository contains the code for a machine learning model that predicts the number of alcohol-related accidents based on historical data. The model is deployed as a Flask API and accepts requests to forecast accident numbers for a given month and year. 

## Features
- Forecasting alcohol-related accidents using a time-series model.
- Uses a RESTful API for making predictions based on the year and month.

## Model
The model used for forecasting is a **SARIMAX** (Seasonal Autoregressive Integrated Moving Average with Exogenous Regressors) model. It uses historical accident data to make predictions about future accidents for specific months and years.

## API Endpoints

### 1. `/`
This is the home endpoint, and it simply returns a welcome message.

**Example Response:** Welcome to the Accident Forecast API!

### 2. `/predict` (POST)
This endpoint accepts a `POST` request with JSON data containing the year and month. It returns the forecasted number of alcohol-related accidents for that specific period.

#### Example Request Body:
```json
{
  "year": 2020,
  "month": 10
}
```
#### Example Response:
```json
{
  "prediction": 41.10481544269847
}
```

#### Example Request using cURL
```
curl -X POST  http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"year": 2020, "month": 10}'
```

## Getting Started

### Prerequisites
To run this project, you need the following:

- Python 3 or later
- Required Python libraries:
  - Flask
  - Pandas
  - Joblib
  - Statsmodels

### Installation

1. **Clone the repository**:
   First, clone this repository to your local machine by running:
   ```bash
   git clone https://github.com/NidhiSonavane/Accidents_forecast.git
   ```
2. **Navigate to the project directory**
   ```bash
   cd Accidents_forecast
   ```
3. **Install the required libraries**
    ```bash
   pip install -r requirements.txt
    ```
