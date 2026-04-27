# Intel Stock Forecast Dashboard

A Streamlit-based dashboard for forecasting stock prices using Facebook Prophet, with built-in cross-validation and performance diagnostics.

## Features

* 📈 Real-time stock data via yFinance
* 🔮 Forecasting with Prophet (linear & logistic growth)
* 📊 Cross-validation with rolling time windows
* 📉 Error metrics (RMSE, MAE, MAPE)
* ⚙️ Hyperparameter tuning controls
* 📉 Interactive visualizations with Plotly

## Tech Stack

* Python
* Streamlit
* Prophet
* yFinance
* Plotly

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Use Case

This project demonstrates how to apply time series forecasting with proper validation, helping understand model reliability across different forecast horizons.

## Note

Stock forecasting is inherently uncertain. This tool is intended for learning and trend analysis, not financial advice.

