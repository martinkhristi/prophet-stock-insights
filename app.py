import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import plotly.graph_objects as go
import matplotlib.pyplot as plt

st.set_page_config(page_title="Intel Stock Forecast Dashboard", layout="wide")

st.title("Intel Stock Forecast Dashboard")
st.write("Real INTC stock data forecasted with Prophet + cross-validation diagnostics.")

# Sidebar
ticker = st.sidebar.text_input("Stock ticker", "INTC")
period = st.sidebar.selectbox("Historical data period", ["3y", "5y", "10y"], index=1)
forecast_days = st.sidebar.slider("Forecast days", 30, 730, 180)

growth_type = st.sidebar.selectbox("Growth type", ["linear", "logistic"])

changepoint_prior_scale = st.sidebar.slider(
    "Changepoint Prior Scale",
    0.001, 0.5, 0.05
)

seasonality_prior_scale = st.sidebar.slider(
    "Seasonality Prior Scale",
    0.01, 10.0, 10.0
)

# Load data
@st.cache_data
def load_stock_data(ticker, period):
    data = yf.download(ticker, period=period)
    data = data.reset_index()
    data = data[["Date", "Close"]]
    data.columns = ["ds", "y"]
    data["ds"] = pd.to_datetime(data["ds"])
    return data

df = load_stock_data(ticker, period)

st.subheader("Raw Stock Data")
st.dataframe(df.tail())

# Logistic cap/floor
if growth_type == "logistic":
    cap_value = st.sidebar.number_input(
        "Carrying capacity / cap",
        value=float(df["y"].max() * 1.5)
    )
    floor_value = st.sidebar.number_input(
        "Floor / minimum value",
        value=float(df["y"].min() * 0.7)
    )

    df["cap"] = cap_value
    df["floor"] = floor_value

# Model
m = Prophet(
    growth=growth_type,
    changepoint_prior_scale=changepoint_prior_scale,
    seasonality_prior_scale=seasonality_prior_scale,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
)

m.fit(df)

future = m.make_future_dataframe(periods=forecast_days)

if growth_type == "logistic":
    future["cap"] = cap_value
    future["floor"] = floor_value

forecast = m.predict(future)

# Forecast plot
st.subheader("Forecast")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["ds"],
    y=df["y"],
    mode="lines",
    name="Actual Close Price"
))

fig.add_trace(go.Scatter(
    x=forecast["ds"],
    y=forecast["yhat"],
    mode="lines",
    name="Forecast"
))

fig.add_trace(go.Scatter(
    x=forecast["ds"],
    y=forecast["yhat_upper"],
    mode="lines",
    name="Upper Bound",
    line=dict(width=0),
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=forecast["ds"],
    y=forecast["yhat_lower"],
    mode="lines",
    name="Lower Bound",
    fill="tonexty",
    line=dict(width=0),
    showlegend=True
))

fig.update_layout(
    height=550,
    xaxis_title="Date",
    yaxis_title="Price",
)

st.plotly_chart(fig, use_container_width=True)

# Components
st.subheader("Prophet Components")
component_fig = m.plot_components(forecast)
st.pyplot(component_fig)

# Cross validation
st.subheader("Cross Validation")

with st.expander("Run cross-validation"):
    initial_days = st.number_input("Initial training period days", value=730)
    period_days = st.number_input("Cutoff spacing days", value=180)
    horizon_days = st.number_input("Forecast horizon days", value=90)

    if st.button("Run Prophet Cross Validation"):
        with st.spinner("Running cross-validation..."):
            df_cv = cross_validation(
                m,
                initial=f"{initial_days} days",
                period=f"{period_days} days",
                horizon=f"{horizon_days} days",
                parallel="processes"
            )

            df_p = performance_metrics(df_cv)

            st.subheader("Cross Validation Results")
            st.dataframe(df_cv.head())

            st.subheader("Performance Metrics")
            st.dataframe(df_p.head())

            col1, col2, col3 = st.columns(3)

            col1.metric("Average RMSE", round(df_p["rmse"].mean(), 2))
            col2.metric("Average MAE", round(df_p["mae"].mean(), 2))
            col3.metric("Average MAPE", f"{round(df_p['mape'].mean() * 100, 2)}%")

            st.subheader("MAPE over Forecast Horizon")
            fig_mape = plot_cross_validation_metric(df_cv, metric="mape")
            st.pyplot(fig_mape)

            st.subheader("RMSE over Forecast Horizon")
            fig_rmse = plot_cross_validation_metric(df_cv, metric="rmse")
            st.pyplot(fig_rmse)

# Forecast table
st.subheader("Forecast Table")
st.dataframe(
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(30)
)