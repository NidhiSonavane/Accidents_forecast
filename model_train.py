import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

# Load Data
data = "Data/monatszahlen2412_verkehrsunfaelle_06_12_24.csv"
df = pd.read_csv(data)

# Clean data
df = df[(df['MONATSZAHL'] == 'Alkoholunfälle') & (df['AUSPRAEGUNG'] == 'insgesamt')] #Filter data for 'Alkoholunfälle' and 'insgesamt'
df = df[df['MONAT'] != 'Summe']  # Remove yearly sum rows

# Create 'Date' column by combining 'JAHR' and 'MONAT'
df['MONAT'] = df['MONAT'].astype(str).str[-2:]  # Extract last two characters as month
df['Date'] = pd.to_datetime(df['JAHR'].astype(str) + '-' + df['MONAT'], errors='coerce')

# print(df[['Date']].head())

# Drop unnecessary data
df = df.drop(columns=['VORJAHRESWERT', 'VERAEND_VORMONAT_PROZENT', 'VERAEND_VORJAHRESMONAT_PROZENT', 'ZWOELF_MONATE_MITTELWERT'])
df.dropna(subset=['WERT'], inplace=True) # Drop null values for 'WERT'

# print(df.head())

# Visualize historical trends
plt.figure(figsize=(12, 6))
plt.title('Historical Trend of Alcohol-related Accidents')
sns.lineplot(data=df, x='Date', y='WERT')
plt.show()

# Set 'Date' as index and set frequency
df = df.set_index('Date')
df = df.asfreq('MS')
# print(df.head())

# Split the data
train_data = df[df['JAHR'] < 2021]
test_data = df[df['JAHR'] == 2021]

# Auto ARIMA for best (p,d,q) values
# auto_model = auto_arima(df['WERT'], seasonal=True, m=12, stepwise=True, trace=True)
# print(auto_model.summary())  

# Fit SARIMA Model (0,1,2)(1,0,1)[12] for seasonal
model = SARIMAX(train_data['WERT'], order=(0, 1, 2), seasonal_order=(1, 0, 1, 12))
model_fit = model.fit()

# Save the model
model_fit.save('alcohol_accidents_forecast_model.pkl')

# Forecast for January 2021
forecast = model_fit.predict(start=test_data.index[0])

# Compare the actual and predicted values for 2021
comparison_df = pd.DataFrame({
    'Date': test_data.index[0],
    'Actual': test_data['WERT'][0],
    'Predicted': forecast
})

# Reset index
comparison_df = comparison_df.reset_index(drop=True)
print(comparison_df)

# Calculate evaluation metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

mae = mean_absolute_error(comparison_df['Actual'], comparison_df['Predicted'])
mse = mean_squared_error(comparison_df['Actual'], comparison_df['Predicted'])
rmse = mse ** 0.5
mape = mean_absolute_percentage_error(comparison_df['Actual'], comparison_df['Predicted'])

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}")

# # Plot Predictions for year 2021
# plt.figure(figsize=(12, 6))
# plt.title('Alcohol-related Accidents: Historical & Forecast')
# sns.lineplot(data=comparison_df, x='Date', y='Predicted', label='Forecast', linestyle='dashed') # Plot forecasted values
# sns.lineplot(data=comparison_df, x='Date', y='Actual', label='Actual', linestyle='solid') # Plot actual values

# plt.xlabel('Date')
# plt.ylabel('Number of Accidents')
# plt.legend()
# plt.show()
