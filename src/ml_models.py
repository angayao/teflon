import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from joblib import dump 


# Utility to flatten each time series sample: (t, f) -> (t*f,)
def flatten_batch(X):
    return np.reshape(X, (X.shape[0], -1))  # (b, t, f) → (b, t*f)

def linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    pred = model.predict(X)
    print("Linear Regression RMSE:", mean_squared_error(y, pred, squared=False))
    return pred

def random_forest(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X, y.ravel())
    pred = model.predict(X)
    print("Random Forest RMSE:", mean_squared_error(y, pred, squared=False))
    return pred

def gradient_boosting(X, y):
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X, y.ravel())
    pred = model.predict(X)
    print("Gradient Boosting RMSE:", mean_squared_error(y, pred, squared=False))
    return pred

def xgboost(X, y):
    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X, y.ravel())
    pred = model.predict(X)
    print("XGBoost RMSE:", mean_squared_error(y, pred, squared=False))
    return pred

def svr(X, y):
    model = SVR(kernel='rbf')
    model.fit(X, y.ravel())
    pred = model.predict(X)
    print("SVR RMSE:", mean_squared_error(y, pred, squared=False))
    return pred

# Flatten utility: (b, t, f) → (b, t*f)
def flatten_batch(X):
    return np.reshape(X, (X.shape[0], -1))

# Train each model and return it
def train_model(model_class, X_train, y_train, **kwargs):
    model = model_class(**kwargs)
    model.fit(X_train, y_train.ravel())
    return model

# Evaluate a model on test data
def evaluate_model(model, X_test, y_test, name="Model"):
    pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, pred)
    print(f"{name} RMSE on test set:", rmse)
    return pred, rmse

def run_ml_models(config, loader):
    # === Train ===
    for x_batch, y_batch in loader.get_loader("train"):
        X_train = flatten_batch(x_batch)
        y_train = y_batch

        print("\nTraining models...")
        lin_model = train_model(LinearRegression, X_train, y_train)
        rf_model = train_model(RandomForestRegressor, X_train, y_train, n_estimators=100, random_state=0)
        gb_model = train_model(GradientBoostingRegressor, X_train, y_train, n_estimators=100, learning_rate=0.1)
        xgb_model = train_model(XGBRegressor, X_train, y_train, n_estimators=100, learning_rate=0.1)
        svr_model = train_model(SVR, X_train, y_train, kernel='rbf')

    # === Test ===

    lin_forecasts = []
    rf_forecasts = []
    gb_forecasts = []
    xg_forecasts = []
    svr_forecasts = []
    trues = []

    for x_batch, y_batch in loader.get_loader("test"):
        X_test = flatten_batch(x_batch)
        y_test = y_batch

        print("\nEvaluating on test set...")
        lin_forecast = evaluate_model(lin_model, X_test, y_test, "Linear Regression")
        rf_forecast = evaluate_model(rf_model, X_test, y_test, "Random Forest")
        gb_forecast = evaluate_model(gb_model, X_test, y_test, "Gradient Boosting")
        xg_forecast = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
        svr_forecast = evaluate_model(svr_model, X_test, y_test, "Support Vector Regressor")
        
        lin_forecasts.append(lin_forecast)
        rf_forecasts.append(rf_forecast)
        gb_forecasts.append(gb_forecast)
        xg_forecasts.append(xg_forecast)
        svr_forecasts.append(svr_forecast)
        trues.append(y_test)
    
    results = {
        'linear': lin_forecasts, 
        'rf': rf_forecasts,
        'gb': gb_forecasts,
        'xg': xg_forecasts,
        'svr': svr_forecasts,
        "true": trues 
    }

    dump(results, f"./exps/ml/{config.dataset}.pkl")
