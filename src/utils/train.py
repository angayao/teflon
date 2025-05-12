from flax.training.train_state import TrainState
from joblib import dump, load
import jax.numpy as jnp
import optax
import jax
from tqdm import tqdm

from src.utils.util import conformal_interval, load_ckpt, save_ckpt
from src.utils.util import fit_arima
from src.utils.util import compute_volatility
from src.utils.util import display_results
from src.utils.plot import error_plots, plot_conformal_intervals
import flax
from src.utils.util import _qlike
from src.utils.util import huber_loss
from src.utils.util import tukey_biweight_loss


def count_params(params):
    """Recursively count parameters in a Flax model."""
    total_params = 0
    for name, param in flax.traverse_util.flatten_dict(params, sep='/').items():
        param_count = param.size
        # print(f"{name}: {param_count} parameters")
        total_params += param_count
    print(f"Total trainable parameters: {total_params}")
    return total_params


def create_train_state(model, rng, lr, input_shape) -> TrainState:
    params = model.init(rng, jnp.ones(input_shape))
    # exponential decay scheduler
    schedule = optax.exponential_decay(
        init_value=lr,
        transition_steps=1000,
        decay_rate=0.99  # Decay factor
    )
    optimizer = optax.adam(schedule)
    state = TrainState.create(apply_fn=model.apply,
                              params=params, tx=optimizer)
    print(f'# of parameters: {count_params(params)}')
    return state


@jax.jit
def train_step(state, x_batch, y_batch, rng):
    rng, sub_rng = jax.random.split(rng)

    def loss_fn(params):
        preds = state.apply_fn(params, rng, x_batch)
        mse_loss = jnp.mean((preds - y_batch) ** 2)
        return mse_loss
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, sub_rng


@jax.jit
def eval_step(state, x_batch, y_batch, rng):
    preds = state.apply_fn(state.params, rng, x_batch)
    loss = jnp.mean((preds - y_batch) ** 2)
    return loss, preds


def train_and_eval(config, state, dataloader, num_epochs=10):
    best_val_loss = float("inf")
    rng = jax.random.PRNGKey(config.seed)

    complete_epochs_residuals = None
    epoch_train_losses = []
    epoch_val_losses = []
    for epoch in tqdm(range(num_epochs)):
        train_losses = []
        for x_batch, y_batch in dataloader.get_loader("train"):
            state, loss, rng = train_step(state, x_batch, y_batch, rng)
            train_losses.append(loss)
        epoch_train_losses.append(jnp.mean(jnp.array(train_losses)))

        val_losses = []
        residuals = []
        for x_batch, y_batch in dataloader.get_loader("calib"):
            loss, preds = eval_step(state, x_batch, y_batch, rng)
            val_losses.append(loss)
            residuals.append(preds - y_batch)
        epoch_val_losses.append(jnp.mean(jnp.array(val_losses)))

        complete_epochs_residuals = residuals
        mean_train_loss = jnp.mean(jnp.array(train_losses))
        mean_val_loss = jnp.mean(jnp.array(val_losses))
        print(
            f"Epoch {epoch + 1}: train_loss->{mean_train_loss:.4f}, val_loss->{mean_val_loss:.4f}")
        if mean_val_loss < best_val_loss:
            config.logger.info(
                f"Saving Checkpoint Prev: {best_val_loss:.4f} Cur: {mean_val_loss:.4f}")
            best_val_loss = mean_val_loss
            save_ckpt(config, state, step=epoch)

    dump(complete_epochs_residuals,
         f"{config.res_dir}/{config.model_name}-{config.dataset}.pkl")

    dump(epoch_train_losses, f'{config.ablation_dir}/{config.model_name}-train.pkl')
    dump(epoch_val_losses, f'{config.ablation_dir}/{config.model_name}-val.pkl')

    return state


def compute_residuals(state, dataloader, rng):
    residuals = []
    for x_batch, y_batch in dataloader.get_loader('calib'):
        rng, subkey = jax.random.split(rng)
        _, preds = eval_step(state, x_batch, y_batch, subkey)
        residuals.append(y_batch - preds)
    residuals = jnp.concatenate(residuals, axis=0)
    residuals = jnp.abs(residuals)
    return residuals


def test_step(config, state, dataloader, q, rng):
    test_losses, predictions, actuals = [], [], []
    lower_bounds, upper_bounds, model_errors, benchmark_errors = [], [], [], []
    benchmark_forecasting = []
    for x_batch, y_batch, in dataloader.get_loader("test"):
        rng, subkey = jax.random.split(rng)
        loss, preds = eval_step(state, x_batch, y_batch, subkey)
        test_losses.append(loss)
        predictions.append(preds)
        actuals.append(y_batch)
        arima_errors, benchmark_forecast = fit_arima(
            x_batch, y_batch, forecast_horizon=config.horizon)

        benchmark_forecasting.append(benchmark_forecast)
        lower_bounds.append(preds - q)
        upper_bounds.append(preds + q)
        model_errors.append(jnp.abs(y_batch - preds))
        benchmark_errors.append(arima_errors)

    results = {
        'actuals': jnp.concatenate(actuals, axis=0),
        'predictions': jnp.concatenate(predictions, axis=0),
        'lower_bounds': jnp.concatenate(lower_bounds, axis=0),
        'upper_bounds': jnp.concatenate(upper_bounds, axis=0),
        'model_errors': jnp.concatenate(model_errors, axis=0),
        'benchmark_errors': jnp.concatenate(jnp.array(benchmark_errors), axis=0),
        'benchmark_forecasting': jnp.concatenate(jnp.array(benchmark_forecasting), axis=0)
    }
    return results


def test_and_analyze(config, state, dataloader):
    rng = jax.random.PRNGKey(config.seed)
    residuals = load(
        f"{config.res_dir}/{config.model_name}-{config.dataset}.pkl")
    residuals = jnp.array(residuals)

    print("STD residuals:", jnp.std(residuals))

    residuals_adj = residuals - jnp.mean(residuals)
    q = conformal_interval(
        residuals_adj, alpha=config.alpha)

    state = load_ckpt(config, state)
    results = test_step(config, state, dataloader,
                        q, rng)
    true_vol, pred_vol = compute_volatility(
        trues=results['actuals'],
        preds=results['predictions']
    )
    results['true_volatility'] = true_vol
    results['pred_volatility'] = pred_vol

    display_results(config, results)
    error_plots(
        benchmark_errs=results['benchmark_errors'],
        model_errs=results['model_errors']
    )
    plot_conformal_intervals(results)

    dump(
        results, f"{config.results_dir}/{config.model_name}-{config.dataset}.pkl")
