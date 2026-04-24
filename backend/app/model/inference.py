from __future__ import annotations

from typing import Iterable

import chess
import numpy as np
import torch

from app.game.board_encoding import encode_board
from app.infra.config import get_current_config



def _validate_board(board: chess.Board) -> chess.Board:
    if not isinstance(board, chess.Board):
        raise TypeError("Expected a chess.Board instance")
    return board



def _resolve_device(model, device: str | None = None):
    return device or next(model.parameters()).device


@torch.no_grad()
def predict_board(model, board: chess.Board, cfg=None, device: str | None = None):
    board = _validate_board(board)
    dev = _resolve_device(model, device=device)
    cfg = cfg or getattr(model, 'cfg', None) or get_current_config()
    was_training = model.training
    model.eval()
    x = encode_board(board, cfg).unsqueeze(0).to(dev)
    policy_logits, value = model(x)
    if was_training:
        model.train()
    return policy_logits.squeeze(0).detach().cpu().numpy(), float(value.item())


@torch.no_grad()
def predict_boards(model, boards: Iterable[chess.Board], cfg=None, device: str | None = None):
    boards_list = [_validate_board(board) for board in boards]
    if not boards_list:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    dev = _resolve_device(model, device=device)
    cfg = cfg or getattr(model, 'cfg', None) or get_current_config()
    was_training = model.training
    model.eval()
    x = torch.stack([encode_board(board, cfg) for board in boards_list], dim=0).to(dev)
    policy_logits, values = model(x)
    if was_training:
        model.train()
    return policy_logits.detach().cpu().numpy(), values.squeeze(1).detach().cpu().numpy()
