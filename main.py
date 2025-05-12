from haiku import LSTM

from src.utils.train import load_ckpt
from src.models.attn import Transformer
from src.models.gru import GRU
import haiku as hk
import warnings
import functools
import os
from src.utils.train import test_and_analyze
from src.utils.train import train_and_eval
from src.utils.train import create_train_state
from src.models.gran import GRAN
from src.models.mlp import MLP
from src.models.rnn import RNN
from src.models.lstm import LSTM
from src.utils.data import TimeSeriesLoader
from config import Config
from flax.training.train_state import TrainState
import jax
from typing import Dict, Callable
from dataclasses import asdict
import argparse
import multiprocessing as mp
from src.stats_models import run_stats_model 
from src.ml_models import run_ml_models
from src.models.ab_nomlp import GRANNoMLP
from src.models.ab_nomlp2 import GRANNoMLP2
from src.models.ab_noconv import GRANNoConv
from src.models.ab_nograv import GRANNoGrav
from src.models.ab_empty import GRANEmpty
mp.set_start_method('spawn', force=True)

warnings.filterwarnings("ignore", category=UserWarning, module="absl")
JAX_TRACEBACK_FILTERING = "off"

MODEL_REGISTRY: Dict[str, Callable] = {
    'rnn': RNN,
    'gru': GRU,
    'lstm': LSTM,
    'mlp': MLP,
    'trans': Transformer,
    'gran': GRAN,  # proposed
    "grannomlp": GRANNoMLP,
    "grannomlp2": GRANNoMLP2,
    "grannoconv": GRANNoConv,
    'grannograv': GRANNoGrav,
    "granempty": GRANEmpty,
}


def forward_fn(x, config):
    model = MODEL_REGISTRY[config.model_name](config)
    return model(x)


def get_model(config, model_name: str) -> Callable:
    if model_name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model: {model_name}. Available: {available}")
    model = hk.transform(functools.partial(forward_fn, config=config))
    return model


def initialize_state(config, model: Callable, loader: TimeSeriesLoader, lr: float, seed: int) -> TrainState:
    mp.set_start_method("spawn", force=True)
    x, _ = next(iter(loader.get_loader("train")))
    state = create_train_state(
        model,
        lr=lr,
        rng=jax.random.PRNGKey(seed),
        input_shape=x.shape
    )
    return state


def run_experiment(config: Config) -> None:
    model = get_model(config, config.model_name)
    loader = TimeSeriesLoader(config)
    if config.mode == "stats":
        run_stats_model(config, loader)
        return 
    
    if config.mode == "ml":
        run_ml_models(config, loader)
        return 

    state = initialize_state(config, model, loader, config.lr, config.seed)
    if config.mode == "train":
        train_and_eval(
            config=config,
            state=state,
            dataloader=loader,
            num_epochs=config.epochs
        )
    elif config.mode == "test":
        test_and_analyze(config, state, loader)
    else:
        raise ValueError(f"Unknown phase: {config.mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GrAN")
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="gran",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model architecture to use"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="DJI",
        help="Dataset to use for training/evaluation"
    )
    parser.add_argument(
        "--task", 
        type=str, 
        default="price"
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        default="train",
        choices=["train", "tune", "test", "stats", "ml"]
    )
    parser.add_argument(
        "--tune", 
        type=str, 
        default="y"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()


def merge_configs(default_config: Config, args: argparse.Namespace) -> Config:
    config_dict = asdict(default_config)
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    return Config(**{**config_dict, **args_dict})


def main():
    args = parse_args()
    config = merge_configs(Config(), args)
    config.ckpt_dir = f'{config.ckpt_dir}/{args.model_name}'
    config.res_dir = f'{config.res_dir}/{args.model_name}'
    config.results_dir = f'{config.results_dir}/{args.model_name}'
    
    os.makedirs(config.ckpt_dir, exist_ok=True)
    os.makedirs(config.res_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.ablation_dir, exist_ok=True)

    print(f'Running {config.model_name} for {config.dataset} - {config.window_size}')
    run_experiment(config)


if __name__ == "__main__":
    main()
