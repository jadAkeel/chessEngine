from dataclasses import replace
import shutil
from unittest.mock import patch

import chess
import numpy as np
import torch

from app.game.board_encoding import encode_board
from app.game.move_encoding import NUM_MOVES, move_to_index
from app.infra.config import AppConfig
from app.training.external_samples import is_validation_position, load_external_samples_sharded
from app.training.train_external import _set_validation_history
from app.training.trainer import _smart_horizontal_flip


def _write_samples(path, cfg):
    board = chess.Board()
    rng = np.random.default_rng(7)
    states, indices = [], []
    for _ in range(60):
        moves = list(board.legal_moves)
        if board.is_game_over():
            board = chess.Board()
            moves = list(board.legal_moves)
        move = moves[int(rng.integers(len(moves)))]
        states.append(encode_board(board, cfg).numpy().astype(np.float16))
        indices.append(move_to_index(move, board))
        board.push(move)
    np.savez_compressed(path, states=states, policy_indices=indices, values=np.zeros(len(states)))


def test_train_validation_are_disjoint_and_repeatable_across_duplicate_files(tmp_path):
    cfg = AppConfig()
    cfg = replace(cfg, external=replace(cfg.external, validation_split=0.3))
    _write_samples(tmp_path / 'a.npz', cfg)
    shutil.copyfile(tmp_path / 'a.npz', tmp_path / 'b.npz')
    train = list(load_external_samples_sharded(tmp_path, cfg, partition='train'))
    val = list(load_external_samples_sharded(tmp_path, cfg, partition='validation'))
    repeat = list(load_external_samples_sharded(tmp_path, cfg, partition='validation'))
    key = lambda samples: [state.tobytes() for state, _, _ in samples]
    assert train and val
    assert len(train) + len(val) == 60
    assert not set(key(train)) & set(key(val))
    assert key(val) == key(repeat)


def test_partition_ignores_clocks_labels_and_horizontal_augmentation():
    cfg = AppConfig()
    state = encode_board(chess.Board(), cfg).numpy()
    mirrored = state[:, :, ::-1].copy()
    mirrored[[13, 14, 15, 16]] = mirrored[[14, 13, 16, 15]]
    mirrored[18:] = 0.95
    assert is_validation_position(state, cfg) == is_validation_position(mirrored, cfg)


def test_horizontal_augmentation_preserves_positions_with_castling_rights():
    cfg = AppConfig()
    board = chess.Board()
    states = encode_board(board, cfg).unsqueeze(0)
    policy = torch.zeros((1, NUM_MOVES))
    policy[0, move_to_index(chess.Move.from_uci('e2e4'), board)] = 1
    before_states, before_policy = states.clone(), policy.clone()
    with patch('app.training.trainer.torch.rand', return_value=torch.zeros(1)):
        after_states, after_policy = _smart_horizontal_flip(states, policy)
    assert torch.equal(after_states, before_states)
    assert torch.equal(after_policy, before_policy)


def test_horizontal_augmentation_keeps_legal_no_castling_targets_for_both_sides():
    cfg = AppConfig()
    for color, uci in [(chess.WHITE, 'e2e4'), (chess.BLACK, 'e7e5')]:
        board = chess.Board()
        board.turn = color
        board.castling_rights = 0
        states = encode_board(board, cfg).unsqueeze(0)
        policy = torch.zeros((1, NUM_MOVES))
        policy[0, move_to_index(chess.Move.from_uci(uci), board)] = 1
        mirrored = board.transform(chess.flip_horizontal)
        reflected_move = chess.Move.from_uci('d2d4' if color else 'd7d5')
        with patch('app.training.trainer.torch.rand', return_value=torch.zeros(1)):
            after_states, after_policy = _smart_horizontal_flip(states, policy)
        assert reflected_move in mirrored.legal_moves
        assert torch.equal(after_states[0], encode_board(mirrored, cfg))
        assert after_policy[0, move_to_index(reflected_move, mirrored)] == 1


def test_changed_validation_set_resets_comparison_without_erasing_history(tmp_path):
    cfg = AppConfig()
    _write_samples(tmp_path / 'samples.npz', cfg)
    samples = list(load_external_samples_sharded(tmp_path, cfg, max_samples=3))
    history = {'val_loss': [0.001]}
    assert _set_validation_history(history, samples) == float('inf')
    history['val_loss'].append(2.0)
    assert _set_validation_history(history, samples) == 2.0
    assert history['val_loss'][0] == 0.001


def test_external_training_keeps_optimizer_and_steps_across_iterations_and_resume(tmp_path):
    import json
    import yaml
    from app.training.train_external import main
    cfg = AppConfig()
    samples = tmp_path / 'samples.npz'
    _write_samples(samples, cfg)
    save_dir = tmp_path / 'models'
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump({
        'model': {'channels': 8, 'res_blocks': 1},
        'training': {'batch_size': 2, 'epochs': 1, 'train_steps_per_iter': 1, 'buffer_size': 20},
        'replay': {'capacity': 20},
        'system': {'cpu_threads': 1},
        'external': {'samples_path': str(samples), 'save_dir': str(save_dir), 'validation_split': 0.3},
    }))
    args = ['train_external', '--config', str(config_path), '--device', 'cpu',
            '--iterations', '2', '--max-train-samples', '20', '--max-val-samples', '10']
    with patch('sys.argv', args):
        main()
    saved = torch.load(save_dir / 'external_latest_checkpoint.pth', weights_only=False)
    assert saved['global_step'] == 2
    assert all(float(state['step']) == 2 for state in saved['optimizer_state_dict']['state'].values())
    args[args.index('--iterations') + 1] = '1'
    with patch('sys.argv', args + ['--resume']):
        main()
    resumed = torch.load(save_dir / 'external_latest_checkpoint.pth', weights_only=False)
    assert resumed['global_step'] == 3
    assert all(float(state['step']) == 3 for state in resumed['optimizer_state_dict']['state'].values())
    history = json.loads((save_dir / 'external_history.json').read_text())
    assert len(history['val_loss']) == 3
    assert history['validation_start_index'] == 0
