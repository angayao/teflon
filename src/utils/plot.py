import seaborn as sns
import matplotlib.pyplot as plt
import jax.numpy as jnp

import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp
from typing import Union, List
import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp
from typing import Union, List


def error_plots(benchmark_errs: Union[List[float], jnp.ndarray],
                model_errs: Union[List[float], jnp.ndarray]):
    """
    Plot comparison of error distributions with both histogram and KDE.

    Args:
        benchmark_errs: Array/list of benchmark error values
        model_errs: Array/list of model error values (same length as benchmark_errs)
    """
    # Input validation and conversion
    benchmark_errs = jnp.asarray(benchmark_errs).flatten()
    model_errs = jnp.asarray(model_errs).flatten()

    if benchmark_errs.ndim != 1 or model_errs.ndim != 1:
        raise ValueError("Inputs must be 1D arrays/lists")
    if len(benchmark_errs) != len(model_errs):
        raise ValueError("Input arrays must have the same length")
    if len(benchmark_errs) == 0:
        raise ValueError("Input arrays cannot be empty")

    # Calculate differences
    diff = model_errs - benchmark_errs

    # Set up figure
    sns.set_style("whitegrid")
    plt.figure(figsize=(16, 6))

    # Plot 1: Combined Histogram and KDE
    plt.subplot(1, 2, 1)

    # Plot histograms (semi-transparent)
    sns.histplot(benchmark_errs, bins=30, color='#FF6B6B', alpha=0.3,
                 label='Benchmark (Hist)', kde=False)
    sns.histplot(model_errs, bins=30, color='#4E79A7', alpha=0.3,
                 label='Model (Hist)', kde=False)

    # Plot KDEs (solid lines)
    sns.kdeplot(benchmark_errs, color='#FF6B6B', linewidth=2.5,
                label=f'Benchmark (Mean={jnp.mean(benchmark_errs):.3f})')
    sns.kdeplot(model_errs, color='#4E79A7', linewidth=2.5,
                label=f'Model (Mean={jnp.mean(model_errs):.3f})')

    plt.xlabel('Error Magnitude', fontsize=12)
    plt.ylabel('Density/Frequency', fontsize=12)
    plt.title('Error Distribution: Histogram + Density', fontsize=14, pad=15)
    plt.legend(frameon=True, framealpha=0.9, fontsize=10)
    plt.xlim(0, 0.55)
    plt.xticks(jnp.arange(0, 0.6, 0.1))

    # Plot 2: Difference Distribution
    plt.subplot(1, 2, 2)
    # Combined violin + boxplot for differences
    sns.violinplot(x=diff, color='#59A14F', inner='box', width=0.5)
    plt.axvline(jnp.mean(diff), color='#E15759', linestyle='--', linewidth=2,
                label=f'Mean Difference = {jnp.mean(diff):.3f}')
    plt.xlabel('Model Error - Benchmark Error', fontsize=12)
    plt.title('Error Differences Distribution', fontsize=14, pad=15)
    plt.legend(frameon=True, framealpha=0.9, fontsize=10)
    plt.xlim(-0.55, 0.05)
    plt.xticks(jnp.arange(-0.5, 0.1, 0.1))

    # Final adjustments
    plt.tight_layout(pad=2.0)
    # plt.show()


def plot_conformal_intervals(results, max_samples=100):
    y_true = results['actuals']
    y_pred = results['predictions']
    lower_bounds = results['lower_bounds']
    upper_bounds = results['upper_bounds']
    true_volatility = results['true_volatility']
    pred_volatility = results['pred_volatility']

    y_true = jnp.array(y_true)
    y_pred = jnp.array(y_pred)
    lower_bounds = jnp.array(lower_bounds)
    upper_bounds = jnp.array(upper_bounds)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.subplots_adjust(hspace=0.3)

    n_samples = min(y_true.shape[0], max_samples)
    time_axis = jnp.arange(n_samples)
    ax1.plot(
        time_axis,
        y_true[:n_samples, -1], 'o-',
        label="True Returns",
        color='black',
        linewidth=1,
        markersize=4
    )
    ax1.plot(
        time_axis,
        y_pred[:n_samples, -1], 'x-',
        label="Predicted Returns",
        color='red',
        linewidth=1,
        markersize=4
    )

    ax1.fill_between(
        time_axis,
        lower_bounds[:n_samples, -1],
        upper_bounds[:n_samples, -1],
        color='lightblue',
        alpha=0.3,
        label="Prediction Interval"
    )

    ax1.set_ylabel("Returns")
    ax1.set_title("Returns with Conformal Prediction Intervals")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(time_axis, true_volatility[:n_samples],
             label='True Realized Volatility', color='blue')
    ax2.plot(time_axis, pred_volatility[:n_samples],
             label='Predicted Volatility', color='red', alpha=0.7)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Volatility")
    ax2.set_title("True vs Predicted Volatility")
    ax2.legend()
    ax2.grid(True)
    # plt.show()
