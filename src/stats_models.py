import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR
from joblib import dump 


def arma(ts):
    arma_model = ARIMA(ts, order=(2, 0, 2)).fit()
    forecast = arma_model.forecast(steps=5)
    # print(arma_model.summary())
    # print("Forecast:", forecast)
    return forecast


def arima(ts):
    arima_model = ARIMA(ts, order=(2, 1, 2)).fit()
    forecast = arima_model.forecast(steps=5)
    # print(arima_model.summary())
    # print("Forecast:", forecast)
    return forecast

def sarimax(ts, ts1):
    sarimax_model = SARIMAX(ts, exog=ts1, order=(1, 1, 1), seasonal_order=(1, 0, 0, 12)).fit()
    forecast = sarimax_model.forecast(steps=5, exog=ts1[-5:])
    # print(sarimax_model.summary())
    # print("Forecast:", forecast)
    return forecast

def sarima(ts):
    model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 0, 1, 12)).fit()
    forecast = model.forecast(steps=5)
    return forecast

def var(ts):
    model = VAR(ts)
    results = model.fit(maxlags=15, ic='aic')
    forecast = results.forecast(ts.values[-results.k_ar:], steps=5)
    # print(results.summary())
    # print("Forecast:\n", pd.DataFrame(forecast, columns=ts.columns))
    return forecast


def run_stats_model(config, loader):
    arma_forecasts = []
    arima_forecasts = [] 
    sarima_forecasts = []
    sarimax_forecasts = []
    var_forecasts = []
    true_values = []

    for x_batch, y_batch in loader.get_loader("test"):
        batch_size = x_batch.shape[0]
        for i in range(batch_size):
            print(f"\n--- Sample {i+1}/{batch_size} ---")

            # Extract time series sample
            x = x_batch[i]  # shape = (42, 5)
            df = pd.DataFrame(x, columns=["open", "low", "high", "close", "past_return"])

            # Choose a return series for univariate models
            ts = df["close"]

            # Exogenous feature for SARIMAX (e.g., past_return)
            ts1 = df[["past_return"]]

            print("\n[ARMA]")
            try:
                arma_forecast = arma(ts)
                arma_forecasts.append(arma_forecast)
            except Exception as e:
                print(f"ARMA failed: {e}")

            print("\n[ARIMA]")
            try:
                arima_forecast = arima(ts)
                arima_forecasts.append(arima_forecast)
            except Exception as e:
                print(f"ARIMA failed: {e}")

            print("\n[SARIMAX]")
            try:
                sarima_forecast = sarima(ts)
                sarima_forecasts.append(sarima_forecast)
            except Exception as e:
                print(f"SARIMAX failed: {e}")

            try:
                sarimax_forecast = sarimax(ts, ts1)
                sarimax_forecasts.append(sarimax_forecast)
            except Exception as e:
                print(f"SARIMAX failed: {e}")
            print("\n[VAR]")
            try:
                var_forecast = var(df[["open",  "high", "low", "close"]])
                var_forecasts.append(var_forecast)
            except Exception as e:
                print(f"VAR failed: {e}")

            true_values.append(y_batch[i])
    stats_results = {
        'arma': arma_forecasts, 
        'arima': arima_forecasts, 
        'sarima': sarima_forecasts, 
        'sarimax': sarimax_forecasts, 
        'var': var_forecasts, 
        'true': true_values
    }

    dump(stats_results, f"./exps/stats/{config.dataset}.pkl")


