from __future__ import annotations

import argparse

from app.infra.config import load_config
from app.infra.device import select_device
from app.infra.logging import setup_logging
from app.infra.runtime import configure_torch_runtime


def add_common_runtime_args(parser: argparse.ArgumentParser, *, require_model_path: bool = False) -> argparse.ArgumentParser:
    parser.add_argument('--config', type=str, default='config/default.yaml')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--model-path', type=str, required=require_model_path, default=None)
    return parser


def configure_runtime(args, logger_name: str, *, role: str = 'cli', worker_count: int = 1):
    cfg = load_config(getattr(args, 'config', None))
    logger = setup_logging(logger_name)
    device = select_device(getattr(args, 'device', None) or cfg.system.device)
    configure_torch_runtime(cfg, device=str(device), role=role, worker_count=worker_count)
    return cfg, logger, str(device)
