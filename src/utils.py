import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_directories(config) -> None:
    directories = [
        config.project_root / "data" / "raw",
        config.project_root / "data" / "processed",
        config.project_root / "outputs" / "plots",
        config.project_root / "outputs" / "models",
        config.project_root / "outputs" / "reports",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def compute_class_weights(labels, num_classes: int) -> torch.Tensor:
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=num_classes)
    counts = np.maximum(counts, 1)

    total = counts.sum()
    weights = total / (num_classes * counts)

    return torch.tensor(weights, dtype=torch.float32)
