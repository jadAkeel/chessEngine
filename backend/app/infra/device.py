from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def select_device(preferred=None):
    if preferred in (None, "auto"):
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if preferred == "cuda" and not torch.cuda.is_available():
        logger.warning("Requested CUDA but it is unavailable; falling back to CPU")
        return "cpu"

    if preferred == "mps" and not torch.backends.mps.is_available():
        logger.warning("Requested MPS but it is unavailable; falling back to CPU")
        return "cpu"

    return preferred


def get_default_device(preferred=None):
    return select_device(preferred)
