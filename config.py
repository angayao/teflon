import logging
from dataclasses import dataclass, field
import jax


def setup_logger():
    logger = logging.getLogger("experiment_logger")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


@dataclass
class Config:
    dataset: str = "dji"
    batch_size: int = 8
    num_layers: int = 4
    input_dim: int = 1
    embed_dim: int = 128
    num_heads: int = 4
    start_date: str = "03-02-2010"
    end_date: str = "31-12-2023"
    logger: logging.Logger = field(default_factory=setup_logger)
    window_size: int = 42
    horizon: int = 1
    lr: float = 5e-5
    wd: float = 5e-5
    epochs: int = 500
    mode: str = "test"
    raw_data_dir: str = "./exps/dataset/raw"
    data_dir: str = "./exps/dataset/sampled"
    ckpt_dir: str = "./exps/ckpts"
    patience: int = 100
    model_name: str = "proposed"
    device = jax.devices("gpu")[0] if jax.default_backend(
    ) == "gpu" else jax.devices("cpu")[0]
    attn_type: str = "traditional"
    seed: int = 42
    alpha: float = 0.05
    res_dir: str = "./exps/residuals"
    results_dir: str = "./exps/results"
    ablation_dir: str = "./exps/ablation"
    task: str = "price"
    tune: str = "y"
    dropout_rate: float = 0.1
