import os
import jax
import pandas as pd
import jax.numpy as jnp
from scipy import stats
import flax.training.checkpoints as ckpoints
from statsmodels.tsa.arima.model import ARIMA

from joblib import dump
# diabold_mariono_test
def dm_test(benchmark_errors, model_errors, h=1, alternative="two-sided"):
    diff = benchmark_errors - model_errors
    mean_diff = jnp.mean(diff)
    var_diff = jnp.var(diff, ddof=1)
    t = len(diff)
    s = var_diff
    for lag in range(1, h):
        gamma = jnp.cov(diff[:-lag], diff[lag:])[0, 1]
        s += 2 * (1 - lag / h) * gamma
    dm_stat = mean_diff / jnp.sqrt(s / t)

    if alternative == "two-sided":
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
        return dm_stat, p_value

    if alternative == "less":
        p_value = stats.norm.cdf(dm_stat)
        return dm_stat, p_value

    p_value = 1 - stats.norm.cdf(dm_stat)
    return dm_stat, p_value


# save model-checkpoint
def save_ckpt(config, state, step):
    ckpt_dir = os.path.abspath(
        f'{config.ckpt_dir}/{config.model_name}-{config.dataset}'
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpoints.save_checkpoint(
        ckpt_dir=ckpt_dir,
        target=state,
        step=step,
        overwrite=True
    )


def load_ckpt(config, state):
    ckpt_dir = os.path.abspath(
        f'{config.ckpt_dir}/{config.model_name}-{config.dataset}'
    )

    if not os.path.exists(ckpt_dir):
        config.logger.info("-> No checkpoint found!!")
        return

    state = ckpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=state)
    config.logger.info('-> Checkpoint for {config.model_name} loaded!!')
    return state


def fit_arima(x_batch, y_batch, forecast_horizon):
    errors = []
    forecasts = []
    for i in range(y_batch.shape[0]):
        x = x_batch[i, :, -1]
        y_true = y_batch[i]
        model = ARIMA(x, order=(1, 1, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=forecast_horizon)
        forecast = jnp.array(forecast).reshape(-1)
        if forecast_horizon == 1:
            error = jnp.abs(y_true - forecast[0])
        else:
            error = jnp.abs(y_true[:forecast_horizon] - forecast)
        errors.append(error)
        forecasts.append(forecast)
    return errors, forecasts


def compute_volatility(trues, preds):
    true_returns = pd.Series(trues[:, -1].flatten())
    pred_returns = pd.Series(preds[:, -1].flatten())
    true_vol = (true_returns.rolling(window=5).std()
                * jnp.sqrt(252)).dropna(inplace=False)
    pred_vol = (pred_returns.rolling(window=5).std()
                * jnp.sqrt(252)).dropna(inplace=False)

    return true_vol.values, pred_vol.values


def conformal_interval(residuals, alpha):
    q = jnp.quantile(jnp.abs(residuals), 0.975, axis=0)
    return q


def tukey_biweight_loss(y_true, y_pred, c=3):
    e = y_true - y_pred
    e_scaled = e / c
    mask = jnp.abs(e_scaled) <= 1
    loss_inliers = (c**2 / 6) * (1 - (1 - e_scaled**2) ** 3)
    loss_outliers = (c**2 / 6)
    loss = jnp.where(mask, loss_inliers, loss_outliers)
    return jnp.mean(loss)


def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    abs_error = jnp.abs(error)
    quadratic = 0.5 * jnp.square(error)
    linear = delta * (abs_error - 0.5 * delta)
    loss = jnp.where(abs_error <= delta, quadratic, linear)
    return jnp.mean(loss)


def _mape(trues, preds, epsilon=1e-6):
    return jnp.mean(jnp.abs((trues - preds) / (jnp.abs(trues) + epsilon))) * 100


def _smape(trues, preds):
    numerator = jnp.abs(trues - preds)
    denominator = (jnp.abs(trues) + jnp.abs(trues)) / 2 + 1e-8
    smape = jnp.mean(numerator / denominator) * 100
    return smape


def _qlike(y_true, y_pred):
    eps = 1e-8  # Small constant for numerical stability
    y_pred = jnp.maximum(y_pred, eps)  # Avoid log(0) or division issues
    return jnp.mean(jnp.log(y_pred) + (y_true / y_pred))


def _mase(y_true, y_pred, y_naive):
    mae_model = jnp.mean(jnp.abs(y_true - y_pred))
    mae_naive = jnp.mean(jnp.abs(y_true - y_naive))
    return mae_model / mae_naive


def _mad(y_true, y_pred):
    return jnp.mean(jnp.abs(y_true - y_pred))


def _directional_acc(trues, preds):
    return jnp.mean(jnp.equal(jnp.sign(trues), jnp.sign(preds))) * 100


def show_metrics(trues, preds, benchmark, task="ret"):
    mae = jnp.mean(jnp.abs(preds - trues))
    mse = jnp.mean((trues - preds) ** 2)
    rmse = jnp.sqrt(mse)
    mape = _mape(trues, preds)
    smape = _smape(trues, preds)
    directional_acc = _directional_acc(trues, preds)
    qlike = _qlike(trues, preds)
    mase = _mase(trues, preds, benchmark)
    mad = _mad(trues, preds)

    print(f"Metrics for {task}")
    print(f'MAE-> {mae:.4f}')
    print(f'MSE -> {mse:.4f}')
    print(f'RMSE -> {rmse:.4f}')
    print(f'MAPE -> {mape:.4f}')
    print(f'SMAPE -> {smape:.4f}')
    print(f'QLIKE-> {qlike:.4f}')
    print(f'MASE-> {mase:.4f}')
    print(f'MAD -> {mad:.4f}')

    if task == "Ret" or task == "price":
        print(f'DA -> {directional_acc:.4f}')


    res = {
        f'mae_{task}': mae,
        f'mse_{task}': mse,
        f'rmse_{task}': rmse,
        f'mape_{task}': mape,
        f'smape_{task}': smape,
        f'da_{task}': directional_acc,
        f'qlike_{task}': qlike,
        f'mase_{task}': mase,
        f'mad_{task}': mad,
    }

    return res


def display_results(config, results):
    coverage = (results['actuals'] >= results['lower_bounds']) \
        & (results['actuals'] <= results['upper_bounds'])
    emperical_coverage = jnp.mean(coverage)

    dm_stat, p_val = dm_test(
        benchmark_errors=results['benchmark_errors'],
        model_errors=results['model_errors']
    )
    print(f'Diabold Mariono Test -> {dm_stat:.4f}')
    print(f'P-Value -> {p_val:.4f}')
    print(f'Emperical Coverage -> {emperical_coverage:.4f}')
    res_ret = show_metrics(trues=results['actuals'], preds=results['predictions'],
                 benchmark=results['benchmark_forecasting'], task="Ret")
    res_vol = show_metrics(trues=results['true_volatility'], preds=results['pred_volatility'],
                 benchmark=results['pred_volatility'], task="Vol")

    res =  res_ret | res_vol

    res['dm_value'] = dm_stat
    res['p_value'] = p_val
    res['emperical_coverage'] = emperical_coverage
    print(res)
    os.makedirs(f"./exps/metrics/{config.model_name}", exist_ok=True)
    dump(res, f"./exps/metrics/{config.model_name}/{config.dataset}.pkl")
