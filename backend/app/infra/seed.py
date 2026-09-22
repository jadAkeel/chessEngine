import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # PyTorch strict mode
        torch.use_deterministic_algorithms(True)

        # CUDA reproducibility (important!)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"