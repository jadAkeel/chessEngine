from __future__ import annotations

import chess
import numpy as np
import torch

from app.infra.config import AppConfig, get_current_config

PIECE_PLANES = 12
SIDE_TO_MOVE_PLANE = 12
CASTLING_START_PLANE = 13
EN_PASSANT_PLANE = 17
HALFMOVE_CLOCK_PLANE = 18
FULLMOVE_NUMBER_PLANE = 19
MIN_REQUIRED_INPUT_PLANES = 20


def _cfg_get(cfg, path: str, default):
    cfg = cfg or get_current_config()
    current = cfg
    for part in path.split('.'):
        if hasattr(current, part):
            current = getattr(current, part)
        else:
            return default
    return current


def get_num_input_planes(cfg: AppConfig | None = None) -> int:
    return int(_cfg_get(cfg, 'model.input_planes', MIN_REQUIRED_INPUT_PLANES))


def _validate_num_planes(num_planes: int) -> None:
    if num_planes < MIN_REQUIRED_INPUT_PLANES:
        raise ValueError(f'model.input_planes must be >= {MIN_REQUIRED_INPUT_PLANES}, got {num_planes}')


def _square_to_coords(square: int, board: chess.Board) -> tuple[int, int]:
    rank = chess.square_rank(square)
    file = chess.square_file(square)
    if board.turn == chess.BLACK:
        rank = 7 - rank
        file = 7 - file
    return rank, file


def encode_board(board: chess.Board, cfg: AppConfig | None = None) -> torch.Tensor:
    cfg = cfg or get_current_config()
    num_planes = get_num_input_planes(cfg)
    _validate_num_planes(num_planes)
    encoded = np.zeros((num_planes, 8, 8), dtype=np.float32)

    for piece_type in chess.PIECE_TYPES:
        for color in (chess.WHITE, chess.BLACK):
            channel = (0 if color == chess.WHITE else 6) + (piece_type - 1)
            for square in board.pieces(piece_type, color):
                rank, file = _square_to_coords(square, board)
                encoded[channel, rank, file] = 1.0

    encoded[SIDE_TO_MOVE_PLANE, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    encoded[CASTLING_START_PLANE + 0, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    encoded[CASTLING_START_PLANE + 1, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    encoded[CASTLING_START_PLANE + 2, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    encoded[CASTLING_START_PLANE + 3, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    if board.ep_square is not None:
        rank, file = _square_to_coords(board.ep_square, board)
        encoded[EN_PASSANT_PLANE, rank, file] = 1.0

    max_halfmove = float(cfg.system.max_halfmove)
    max_fullmove = float(cfg.system.max_fullmove)
    encoded[HALFMOVE_CLOCK_PLANE, :, :] = min(board.halfmove_clock, max_halfmove) / max_halfmove
    encoded[FULLMOVE_NUMBER_PLANE, :, :] = min(board.fullmove_number, max_fullmove) / max_fullmove
    return torch.from_numpy(encoded)
