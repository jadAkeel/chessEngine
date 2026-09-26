"""Regressions for the second search pass: batch collisions, exception-safe virtual
visits, repetition as a draw value, visit-based move choice, smart pruning, FEN-only
tree reuse and progressive penalties."""
from dataclasses import replace
from unittest.mock import patch

import chess
import numpy as np
import pytest

from app.game.move_encoding import NUM_MOVES, move_to_index
from app.infra.config import AppConfig, MCTSConfig, ModelConfig, load_config
from app.mcts.node import Node
from app.mcts.search import MCTS


def _cfg(**mcts_overrides) -> AppConfig:
    params = dict(num_simulations=8, inference_batch_size=16, classical_value_alpha=0.0)
    params.update(mcts_overrides)
    return AppConfig(
        model=ModelConfig(input_planes=20, channels=8, res_blocks=1, value_dropout=0.0),
        mcts=MCTSConfig(**params),
    )


def _yaml_cfg(**mcts_overrides) -> AppConfig:
    cfg = load_config("config/default.yaml")
    return replace(cfg, mcts=replace(cfg.mcts, classical_value_alpha=0.0, **mcts_overrides))


def _key(board: chess.Board) -> str:
    return " ".join(board.fen().split()[:4])


def _predict(value_for_white: float = 0.0, favoured: dict[str, str] | None = None, counter: list | None = None):
    """Network stub: favours ``favoured[fen]`` when legal, fixed value from White's view."""
    favoured = favoured or {}

    def predict(model, boards, cfg=None, device=None):
        boards = list(boards)
        if counter is not None:
            counter.append(len(boards))
        logits = np.zeros((len(boards), NUM_MOVES), dtype=np.float32)
        values = np.zeros((len(boards),), dtype=np.float32)
        for row, board in enumerate(boards):
            uci = favoured.get(_key(board))
            if uci:
                move = chess.Move.from_uci(uci)
                if move in board.legal_moves:
                    logits[row, move_to_index(move, board)] = 8.0
            values[row] = value_for_white if board.turn == chess.WHITE else -value_for_white
        return logits, values

    return predict


def _tree_nodes(root: Node):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(node.children.values())


# --------------------------------------------------------------------------- batching


def test_batch_collisions_are_not_backed_up_twice_in_a_losing_position():
    """The old absolute virtual loss anchored a pending leaf at -1; in a lost position that is
    barely worse than the alternatives, so one leaf was queued several times per batch and its
    single evaluation backed up several times (12 of 64 rollouts here before the fix)."""
    board = chess.Board("4k3/8/8/8/8/8/QQ6/4K3 b - - 0 1")
    search = MCTS(None, _cfg(smart_pruning=False))
    expansions: dict[int, int] = {}
    original = search._expand_node_from_prediction

    def counting(node, board, policy_logits, nn_value, add_noise):
        expansions[id(node)] = expansions.get(id(node), 0) + 1
        return original(node=node, board=board, policy_logits=policy_logits, nn_value=nn_value, add_noise=add_noise)

    with (
        patch.object(search, "_expand_node_from_prediction", side_effect=counting),
        patch("app.mcts.search.predict_boards", side_effect=_predict(0.8)),
    ):
        result = search.search(board, num_simulations=64)
    assert result["completed_simulations"] == 64
    assert max(expansions.values()) == 1
    assert sum(result["visit_counts"].values()) == 64


def test_relative_virtual_loss_spreads_a_batch_in_a_lost_position_too():
    """A pending child must look worse than an untried sibling even when the parent is lost."""
    search = MCTS(None, _cfg(fpu_reduction=0.25, virtual_loss=1.0))
    root = Node(0.0)
    root.visit_count, root.value_sum = 10, -8.5  # parent q = -0.85 -> FPU = -1.10
    first, second = chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")
    root.children = {first: Node(0.30, root), second: Node(0.28, root)}
    root.children[first].add_virtual_visit(1)
    with patch.object(search, "_ensure_child_penalties"):
        assert search._select_child(root, chess.Board())[0] == second


def test_failed_inference_leaves_no_virtual_visits_in_the_retained_tree():
    search = MCTS(None, _cfg())
    calls = {"n": 0}
    ok = _predict(0.0)

    def flaky(model, boards, cfg=None, device=None):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("inference failed")
        return ok(model, boards, cfg=cfg, device=device)

    with patch("app.mcts.search.predict_boards", side_effect=flaky):
        with pytest.raises(RuntimeError):
            search.search(chess.Board(), num_simulations=64)
    assert sum(node.virtual_visits for node in _tree_nodes(search._root)) == 0


def test_reused_tree_is_cleared_of_stale_virtual_visits():
    search = MCTS(None, _cfg())
    board = chess.Board()
    with patch("app.mcts.search.predict_boards", side_effect=_predict(0.0)):
        search.search(board, num_simulations=16)
        for node in _tree_nodes(search._root):
            node.virtual_visits = 3  # simulate a leak from an older code path
        search.search(board, num_simulations=16)
    assert sum(node.virtual_visits for node in _tree_nodes(search._root)) == 0


# --------------------------------------------------------------------------- repetition as value


def _threefold_offer_board() -> tuple[chess.Board, chess.Move]:
    """Black (two queens down) can claim a threefold with ...Nf6."""
    board = chess.Board("7k/8/8/7n/8/QQ6/8/4K3 b - - 0 1")
    for uci in ["h5f6", "e1e2", "f6h5", "e2e1", "h5f6", "e1e2", "f6g8", "e2e1"]:
        board.push_uci(uci)
    return board, chess.Move.from_uci("g8f6")


def test_losing_side_takes_the_threefold_draw():
    """The old in-tree repetition penalty (1.6 for a claimable repetition) outweighed the
    +0.8 the draw is worth, so a lost engine refused to repeat."""
    board, draw = _threefold_offer_board()
    search = MCTS(None, _yaml_cfg())
    with patch("app.mcts.search.predict_boards", side_effect=_predict(0.8)):
        result = search.search(board, num_simulations=64)
    assert result["best_move"] == draw
    assert result["root_value"] > -0.5


def test_winning_side_does_not_repeat_into_a_draw():
    board = chess.Board("6nk/8/8/8/8/QQ6/8/4K3 w - - 0 1")  # White two queens up
    for uci in ["e1e2", "g8f6", "e2f1", "f6g8", "f1e2", "g8f6", "e2f1", "f6g8"]:
        board.push_uci(uci)
    repeat = chess.Move.from_uci("f1e2")
    after = board.copy()
    after.push(repeat)
    assert after.is_repetition(3) and not board.is_repetition(3)
    search = MCTS(None, _yaml_cfg())
    with patch("app.mcts.search.predict_boards", side_effect=_predict(0.8)):
        result = search.search(board, num_simulations=64)
    assert result["best_move"] != repeat
    assert result["root_value"] > 0.5


def test_repetition_inside_the_search_path_is_a_draw_but_the_root_position_is_not():
    search = MCTS(None, _cfg())
    board = chess.Board("4k3/8/8/8/8/8/8/4K2R w - - 0 1")
    assert search._terminal_state(board, {}, repeated_in_path=True) == 0.0
    assert search._terminal_state(board, {}, repeated_in_path=False) is None


def test_play_mode_edges_carry_no_repetition_or_progress_penalty():
    board, draw = _threefold_offer_board()
    search = MCTS(None, _yaml_cfg())
    with patch("app.mcts.search.predict_boards", side_effect=_predict(0.8)):
        search.search(board, num_simulations=16)
    components = search._root.children[draw].penalty_components
    assert "repetition" not in components and "progress" not in components


def test_selfplay_mode_keeps_the_legacy_repetition_penalty():
    board, draw = _threefold_offer_board()
    search = MCTS(None, _yaml_cfg())
    with patch("app.mcts.search.predict_boards", side_effect=_predict(0.8)):
        search.search(board, num_simulations=16, add_noise=True)
    assert search._root.children[draw].penalty_components["repetition"] > 1.0


# --------------------------------------------------------------------------- move choice / pruning


def test_play_mode_move_is_most_visited_whatever_the_temperature():
    board = chess.Board()
    favoured = {_key(board): "g1f3"}
    moves = []
    for temperature in (1.0, 0.2, 0.05):
        search = MCTS(None, _cfg())
        with patch("app.mcts.search.predict_boards", side_effect=_predict(0.0, favoured)):
            result = search.search(board, num_simulations=48, temperature=temperature)
        top = max(result["visit_counts"].items(), key=lambda kv: kv[1])[0]
        assert result["best_move"] == top
        moves.append(result["best_move"])
    assert len(set(moves)) == 1


def test_smart_pruning_stops_early_without_changing_the_move():
    board = chess.Board()
    favoured = {_key(board): "e2e4"}
    results = {}
    for pruning in (True, False):
        search = MCTS(None, _cfg(smart_pruning=pruning))
        with patch("app.mcts.search.predict_boards", side_effect=_predict(0.0, favoured)):
            results[pruning] = search.search(board, num_simulations=400)
    assert results[True]["best_move"] == results[False]["best_move"]
    assert results[True]["completed_simulations"] < results[False]["completed_simulations"] == 400


def test_smart_pruning_is_off_in_selfplay_mode():
    board = chess.Board()
    search = MCTS(None, _cfg(smart_pruning=True))
    with patch("app.mcts.search.predict_boards", side_effect=_predict(0.0, {_key(board): "e2e4"})):
        result = search.search(board, num_simulations=120, add_noise=True)
    assert result["completed_simulations"] == 120


def test_progressive_penalty_lets_a_proven_move_overrule_its_penalty():
    """A constant offset never fades: a penalised move that searches as clearly best stays
    suppressed. Progressive bias keeps the first-visit deterrent and then lets Q decide."""
    penalised, plain = chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")

    def tree():
        root = Node(0.0)
        root.visit_count = 21
        root.children = {penalised: Node(0.5, root), plain: Node(0.5, root)}
        for child, value_for_mover in ((root.children[penalised], 0.9), (root.children[plain], 0.3)):
            child.visit_count, child.value_sum = 10, -value_for_mover * 10
        root.children[penalised].penalty = 0.85
        root.penalties_ready, root.penalty_tier = True, 0
        return root

    assert MCTS(None, _cfg(penalty_mode="offset"))._select_child(tree(), chess.Board())[0] == plain
    assert MCTS(None, _cfg(penalty_mode="progressive"))._select_child(tree(), chess.Board())[0] == penalised


def test_progressive_penalty_still_deters_the_first_visit():
    penalised, plain = chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")
    root = Node(0.0)
    root.visit_count = 4
    root.children = {penalised: Node(0.5, root), plain: Node(0.5, root)}
    root.children[penalised].penalty = 0.85
    root.penalties_ready, root.penalty_tier = True, 0
    assert MCTS(None, _cfg(penalty_mode="progressive"))._select_child(root, chess.Board())[0] == plain


def test_unknown_penalty_mode_is_rejected():
    with pytest.raises(ValueError):
        MCTS(None, _cfg(penalty_mode="sometimes"))


# --------------------------------------------------------------------------- tree reuse


def test_fen_only_request_reuses_the_subtree_of_the_played_reply():
    """The web API sends FEN only; the tree is found by position two plies down."""
    start = chess.Board()
    after_e4 = chess.Board()
    after_e4.push_uci("e2e4")
    favoured = {_key(start): "e2e4", _key(after_e4): "e7e5"}
    search = MCTS(None, _cfg(smart_pruning=False))
    with patch("app.mcts.search.predict_boards", side_effect=_predict(0.0, favoured)):
        search.search(chess.Board(start.fen()), num_simulations=64)
        continued = chess.Board()
        continued.push_uci("e2e4")
        continued.push_uci("e7e5")
        fen_only = chess.Board(continued.fen())
        retained = search.retained_visits(fen_only)
        assert retained > 0
        result = search.search(fen_only, num_simulations=8)
    assert result["retained_visits"] == retained
    assert result["best_move"] in fen_only.legal_moves


def test_fen_only_lookup_ignores_boards_with_history_that_does_not_match():
    search = MCTS(None, _cfg(smart_pruning=False))
    with patch("app.mcts.search.predict_boards", side_effect=_predict(0.0)):
        search.search(chess.Board(), num_simulations=32)
    other = chess.Board()
    other.push_uci("d2d4")
    other.push_uci("d7d5")
    assert search.retained_visits(other) == 0


def test_selfplay_never_reuses_a_tree_and_play_does():
    """Self-play policy targets must be the visits of one noisy search; retained visits
    were gathered without root noise. Play keeps reusing."""
    board = chess.Board()
    search = MCTS(None, _cfg())
    with patch("app.mcts.search.predict_boards", side_effect=_predict(0.0)):
        search.search(board, num_simulations=16)
        assert search.search(board, num_simulations=16, add_noise=True)["retained_visits"] == 0
        assert search.search(board, num_simulations=16, add_noise=True)["retained_visits"] == 0
        search.search(board, num_simulations=16)
        assert search.search(board, num_simulations=16)["retained_visits"] > 0


# --------------------------------------------------------------------------- terminals / final choice


def test_drawn_terminal_does_not_absorb_the_batch_in_a_won_position():
    """Match game 6: with K+Q v K the engine played ...Kc3 stalemate. A terminal rollout was
    backed up at once without a virtual visit, so once every sibling carried a pending loss
    the stalemate (value 0) took the rest of each batch and won on visits."""
    board = chess.Board("8/8/8/8/3k4/q7/8/1K6 b - - 17 63")
    stalemates = set()
    for move in board.legal_moves:
        board.push(move)
        if board.is_stalemate():
            stalemates.add(move)
        board.pop()
    assert stalemates  # d4c3 and d4d3
    favoured = {_key(board): "d4c3"}  # the network also preferred the stalemate
    search = MCTS(None, _cfg(mopup_endgames=False, smart_pruning=False))
    with patch("app.mcts.search.predict_boards", side_effect=_predict(-0.8, favoured)):
        result = search.search(board, num_simulations=64)
    assert result["best_move"] not in stalemates
    assert result["root_value"] > 0.5


def test_lower_confidence_bound_overrules_a_batch_absorbed_draw():
    search = MCTS(None, _cfg(lcb_scale=1.0))
    root = Node(0.0)
    draw, win = chess.Move.from_uci("d4c3"), chess.Move.from_uci("d4c4")
    root.children = {draw: Node(0.4, root), win: Node(0.2, root)}
    root.children[draw].visit_count, root.children[draw].value_sum = 38, 0.0
    root.children[win].visit_count, root.children[win].value_sum = 7, -0.81 * 7  # +0.81 for the mover
    assert search._rank_root_moves(root)[0] == win
    assert MCTS(None, _cfg(lcb_scale=0.0))._rank_root_moves(root)[0] == draw


def test_lower_confidence_bound_keeps_a_well_visited_move_against_a_lucky_sample():
    search = MCTS(None, _cfg(lcb_scale=1.0))
    root = Node(0.0)
    solid, lucky = chess.Move.from_uci("e2e4"), chess.Move.from_uci("h2h4")
    root.children = {solid: Node(0.5, root), lucky: Node(0.1, root)}
    root.children[solid].visit_count, root.children[solid].value_sum = 40, -0.10 * 40
    root.children[lucky].visit_count, root.children[lucky].value_sum = 6, -0.30 * 6
    assert search._rank_root_moves(root)[0] == solid


def test_lower_confidence_bound_ignores_moves_with_a_token_share_of_visits():
    search = MCTS(None, _cfg(lcb_scale=1.0))
    root = Node(0.0)
    main, token = chess.Move.from_uci("e2e4"), chess.Move.from_uci("h2h4")
    root.children = {main: Node(0.5, root), token: Node(0.1, root)}
    root.children[main].visit_count, root.children[main].value_sum = 60, 0.0
    root.children[token].visit_count, root.children[token].value_sum = 3, -3.0  # +1.0 on three visits
    assert search._rank_root_moves(root)[0] == main


def test_smart_pruning_waits_while_values_disagree_with_visits():
    search = MCTS(None, _cfg(lcb_scale=1.0))
    root = Node(0.0)
    draw, win = chess.Move.from_uci("d4c3"), chess.Move.from_uci("d4c4")
    root.children = {draw: Node(0.4, root), win: Node(0.2, root)}
    root.children[draw].visit_count, root.children[draw].value_sum = 38, 0.0
    root.children[win].visit_count, root.children[win].value_sum = 7, -0.81 * 7
    assert not search._best_move_is_settled(root, 4, 45, 0.0, None)
    root.children[win].value_sum = 0.2 * 7  # now clearly worse: visits and values agree
    assert search._best_move_is_settled(root, 4, 45, 0.0, None)


def test_bare_king_value_rewards_a_cornered_king_and_close_kings():
    from app.mcts.search import bare_king_value

    centred = bare_king_value(chess.Board("8/8/8/4k3/8/8/8/KQ6 w - - 0 1"))
    cornered = bare_king_value(chess.Board("7k/8/6K1/8/8/8/8/1Q6 w - - 0 1"))
    assert 0.5 < centred < cornered < 1.0
    assert bare_king_value(chess.Board("7k/8/6K1/8/8/8/8/1Q6 b - - 0 1")) == pytest.approx(-cornered)
    assert bare_king_value(chess.Board("8/8/8/4k3/8/8/P7/K7 w - - 0 1")) is None


def test_unquoted_yaml_off_means_penalties_off():
    from app.infra.config import validate

    cfg = _cfg(penalty_mode=False)  # what yaml.safe_load("off") returns
    validate(cfg)
    assert MCTS(None, cfg).penalty_mode == "off"


# --------------------------------------------------------------------------- invariants (mutation-tested gaps)


def _material_predict():
    """Uniform policy; value = tanh(material / 300) for the side to move."""
    values_by_piece = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900}

    def predict(model, boards, cfg=None, device=None):
        boards = list(boards)
        values = np.zeros((len(boards),), dtype=np.float32)
        for row, board in enumerate(boards):
            balance = sum(
                value * (len(board.pieces(piece, board.turn)) - len(board.pieces(piece, not board.turn)))
                for piece, value in values_by_piece.items()
            )
            values[row] = np.tanh(balance / 300.0)
        return np.zeros((len(boards), NUM_MOVES), dtype=np.float32), values

    return predict


def test_values_propagate_with_alternating_signs_end_to_end():
    """With a non-alternating backup the search stops preferring the capture of a free queen."""
    board = chess.Board("3q3k/8/8/8/8/8/8/3RK3 w - - 0 1")
    search = MCTS(None, _cfg(penalty_mode="off", smart_pruning=False))
    with patch("app.mcts.search.predict_boards", side_effect=_material_predict()):
        result = search.search(board, num_simulations=96)
    assert result["best_move"] == chess.Move.from_uci("d1d8")
    assert result["root_value"] > 0.3
    assert sum(node.virtual_visits for node in _tree_nodes(search._root)) == 0


def test_a_pending_child_yields_to_an_identical_sibling():
    """Kills a disabled or sign-flipped virtual loss: equal children, the first one pending."""
    search = MCTS(None, _cfg(virtual_loss=1.0))
    root = Node(0.0)
    root.visit_count = 41
    pending, other = chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")
    root.children = {pending: Node(0.5, root), other: Node(0.5, root)}
    for child in root.children.values():
        child.visit_count, child.value_sum = 20, -6.0
    root.children[pending].add_virtual_visit(1)
    with patch.object(search, "_ensure_child_penalties"):
        assert search._select_child(root, chess.Board())[0] == other


def test_reuse_follows_the_moves_actually_played_not_the_most_visited_line():
    board = chess.Board()
    search = MCTS(None, _cfg(smart_pruning=False))
    with patch("app.mcts.search.predict_boards", side_effect=_predict(0.0)):
        search.search(board, num_simulations=200)  # flat prior: visits spread over many replies
        root = search._root
        top = max(root.children, key=lambda move: root.children[move].visit_count)
        played = next(
            move for move, child in sorted(root.children.items(), key=lambda kv: kv[1].visit_count)
            if move != top and child.expanded() and any(grand.expanded() for grand in child.children.values())
        )
        reply = next(move for move, grand in root.children[played].children.items() if grand.expanded())
        expected = root.children[played].children[reply]
        board.push(played)
        board.push(reply)
        search.search(board, num_simulations=4)
    assert search._root is expected


def test_final_move_follows_visits_not_the_prior():
    board = chess.Board()
    search = MCTS(None, _cfg())
    root = Node(0.0)
    visited, favoured_prior = chess.Move.from_uci("d2d4"), chess.Move.from_uci("e2e4")
    root.children = {favoured_prior: Node(0.7, root), visited: Node(0.3, root)}
    root.children[visited].visit_count = 30
    root.children[favoured_prior].visit_count = 10
    assert search._rank_root_moves(root)[0] == visited

