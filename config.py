"""Configuration for CIFAR-100 shift experiments."""
from dataclasses import dataclass
from typing import Tuple
import torch

@dataclass
class Config:
    # --- data ---
    data_dir: str = "./data"
    batch_size: int = 128
    num_workers: int = 2   # Colab safe default
    num_classes: int = 100

    # --- training ---
    num_epochs: int = 120
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # --- logging ---
    exp_name: str = "cifar100_resnet18"
    variant: str = "baseline"  # baseline | shift_int | shift_grid | scale | ...
    log_root: str = "./runs"
    ckpt_root: str = "./checkpoints"
    save_best: bool = True

    # --- ERF logging ---
    erf_epochs: Tuple[int, ...] = (1, 10, 20, 30, 50)
    erf_num_batches: int = 2
