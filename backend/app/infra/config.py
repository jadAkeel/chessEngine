from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json
import yaml
from typing import TypedDict

class PiecePenaltyCfg(TypedDict, total=False):
    blunder_penalty: float
    hanging_penalty: float
    sac_compensation_threshold: float
    check_discount: float

@dataclass(frozen=True)
class ModelConfig:
    input_planes: int = 20
    channels: int = 128
    res_blocks: int = 8
    value_dropout: float = 0.15


@dataclass(frozen=True)
class TrainingConfig:
    lr: float = 7e-4
    min_lr: float = 2e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 5
    train_steps_per_iter: int = 200
    grad_clip: float = 1.0
    value_loss_coeff: float = 1.0
    entropy_coeff: float = 0.01
    policy_label_smoothing: float = 0.03
    use_amp: bool = True
    enable_horizontal_flip_augment: bool = True
    step_scheduler_per_batch: bool = False
    lr_decay_gamma: float = 0.997
    external_samples_path: str = ""
    external_samples_max: int = 0
    buffer_size: int = 500000

import torch


@dataclass(frozen=True)
class SparsePolicyBatchTensor:
    indices: torch.Tensor
    probs: torch.Tensor
    lengths: torch.Tensor
    num_actions: int

    @property
    def batch_size(self) -> int:
        return int(self.lengths.numel())
    

@dataclass(frozen=True)
class ExternalDataConfig:
    samples_path: str = ""
    max_samples: int = 0
    min_fullmove: int = 0
    max_fullmove: int = 0
    shuffle: bool = True
    validation_split: float = 0.1
    seed: int = 42
    dedup: bool = True
    filter_invalid: bool = True
    drop_zero_states: bool = True
    checkpoint_prefix: str = "external"
    save_dir: str = "models/external"
    benchmark_games: int = 8


@dataclass(frozen=True)
class ReplayConfig:
    capacity: int = 25_000
    prioritized: bool = True
    alpha: float = 0.6
    beta_start: float = 0.4
    beta_end: float = 1.0
    eps: float = 1e-6
    policy_mix: float = 0.1
    recent_sample_fraction: float = 0.5
    recent_window_size: int = 4_000
    balance_outcomes: bool = True
    draw_value_threshold: float = 0.10
    max_draw_fraction: float = 0.45
    save_interval: int = 1
    save_shard_size: int = 4096


@dataclass(frozen=True)
class PenaltyDiagnosticsConfig:
    enabled: bool = False


@dataclass(frozen=True)
class PrinciplePenaltiesConfig:
    enabled: bool = False
    max_total_per_move: float = 0.2
    king_safety: float = 0.1
    opening_development: float = 0.055
    center_control: float = 0.045
    tactics: float = 0.07
    pawn_structure: float = 0.025
    piece_activity: float = 0.025
    rook_activity: float = 0.045
    endgame: float = 0.04


@dataclass(frozen=True)
class MCTSConfig:
    num_simulations: int = 64
    c_puct: float = 1.8
    dirichlet_alpha: float = 0.25
    dirichlet_eps: float = 0.25
    temperature: float = 1.0
    classical_value_alpha: float = 0.35
    resign_threshold: float = -0.965
    min_resign_plies: int = 60
    inference_batch_size: int = 24
    virtual_loss: float = 1.0

    queen_blunder_penalty: float = 0.5
    queen_hanging_penalty: float = 0.24
    queen_sac_compensation_threshold: float = 500
    queen_check_discount: float = 0.75

    piece_penalties: dict[str, PiecePenaltyCfg] = field(default_factory=dict)

@dataclass(frozen=True)
class SelfPlayConfig:
    num_workers: int = 2
    games_per_worker: int = 2
    max_game_length: int = 220
    opening_random_plies_min: int = 2
    opening_random_plies_max: int = 8
    repetition_penalty: float = 0.2
    repetition_break_count: int = 3
    repetition_move_weight: float = 0.08
    repetition_draw_value: float = 0.15
    max_length_draw_value: float = 0.05
    temperature_high_moves: int = 12
    temperature_mid_moves: int = 24
    temperature_high: float = 1.4
    temperature_mid: float = 0.8
    temperature_low: float = 0.3


@dataclass(frozen=True)
class ArenaConfig:
    games: int = 6
    update_threshold: float = 0.55
    early_stop_margin: float = 0.2
    benchmark_games: int = 4
    baseline_search_depth: int = 2
    elo_k_factor: float = 24.0
    initial_elo: float = 1200.0
    search_temperature: float = 0.0
    resign_threshold: float = -0.95
    max_repetition_draw_rate: float = 0.60
    repetition_break_count: int = 3
    repetition_soft_limit_plies: int = 16
    repetition_move_weight: float = 0.30
    hard_block_repetition: bool = True
    contempt_factor: float = 0.05
    randomize_openings: bool = True
    fallback_top_k: int = 3


@dataclass(frozen=True)
class SystemConfig:
    device: str = "auto"
    log_level: str = "INFO"
    json_logs: bool = False
    checkpoint_path: str = "models/best_model.pth"
    default_bestmove_simulations: int = 32
    max_halfmove: int = 100
    max_fullmove: int = 200
    cpu_threads: int = 0
    interop_threads: int = 0
    worker_thread_policy: str = "auto"


@dataclass(frozen=True)
class AppConfig:
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    external: ExternalDataConfig = ExternalDataConfig()
    replay: ReplayConfig = ReplayConfig()
    penalty_diagnostics: PenaltyDiagnosticsConfig = PenaltyDiagnosticsConfig()
    principle_penalties: PrinciplePenaltiesConfig = PrinciplePenaltiesConfig()
    mcts: MCTSConfig = MCTSConfig()
    selfplay: SelfPlayConfig = SelfPlayConfig()
    arena: ArenaConfig = ArenaConfig()
    system: SystemConfig = SystemConfig()


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"
_CURRENT_CONFIG = AppConfig()

_FLAT_MAP = {
    "BOARD_INPUT_CHANNELS": ("model", "input_planes"),
    "NUM_INPUT_PLANES": ("model", "input_planes"),
    "NUM_CHANNELS": ("model", "channels"),
    "NUM_RES_BLOCKS": ("model", "res_blocks"),
    "VALUE_HEAD_DROPOUT": ("model", "value_dropout"),
    "LEARNING_RATE": ("training", "lr"),
    "MIN_LEARNING_RATE": ("training", "min_lr"),
    "WEIGHT_DECAY": ("training", "weight_decay"),
    "BATCH_SIZE": ("training", "batch_size"),
    "EPOCHS": ("training", "epochs"),
    "TRAIN_STEPS_PER_ITER": ("training", "train_steps_per_iter"),
    "GRAD_CLIP_NORM": ("training", "grad_clip"),
    "VALUE_LOSS_COEFF": ("training", "value_loss_coeff"),
    "ENTROPY_COEFF": ("training", "entropy_coeff"),
    "POLICY_LABEL_SMOOTHING": ("training", "policy_label_smoothing"),
    "TRAIN_USE_AMP": ("training", "use_amp"),
    "ENABLE_HORIZONTAL_FLIP_AUGMENT": ("training", "enable_horizontal_flip_augment"),
    "STEP_SCHEDULER_PER_BATCH": ("training", "step_scheduler_per_batch"),
    "LR_DECAY_GAMMA": ("training", "lr_decay_gamma"),
    "BUFFER_CAPACITY": ("replay", "capacity"),
    "PRIORITIZED_REPLAY": ("replay", "prioritized"),
    "PRIORITY_ALPHA": ("replay", "alpha"),
    "PRIORITY_BETA_START": ("replay", "beta_start"),
    "PRIORITY_BETA_END": ("replay", "beta_end"),
    "PRIORITY_EPS": ("replay", "eps"),
    "PRIORITY_POLICY_MIX": ("replay", "policy_mix"),
    "RECENT_SAMPLE_FRACTION": ("replay", "recent_sample_fraction"),
    "RECENT_WINDOW_SIZE": ("replay", "recent_window_size"),
    "BALANCE_OUTCOMES": ("replay", "balance_outcomes"),
    "DRAW_VALUE_THRESHOLD": ("replay", "draw_value_threshold"),
    "MAX_DRAW_FRACTION": ("replay", "max_draw_fraction"),
    "REPLAY_SAVE_INTERVAL": ("replay", "save_interval"),
    "REPLAY_SAVE_SHARD_SIZE": ("replay", "save_shard_size"),
    "NUM_SIMULATIONS": ("mcts", "num_simulations"),
    "C_PUCT": ("mcts", "c_puct"),
    "DIRICHLET_ALPHA": ("mcts", "dirichlet_alpha"),
    "DIRICHLET_EPSILON": ("mcts", "dirichlet_eps"),
    "ROOT_TEMPERATURE": ("mcts", "temperature"),
    "CLASSICAL_VALUE_ALPHA": ("mcts", "classical_value_alpha"),
    "RESIGN_THRESHOLD": ("mcts", "resign_threshold"),
    "MIN_RESIGN_PLIES": ("mcts", "min_resign_plies"),
    "INFERENCE_BATCH_SIZE": ("mcts", "inference_batch_size"),
    "MCTS_VIRTUAL_LOSS": ("mcts", "virtual_loss"),
    "QUEEN_BLUNDER_PENALTY": ("mcts", "queen_blunder_penalty"),
    "QUEEN_HANGING_PENALTY": ("mcts", "queen_hanging_penalty"),
    "QUEEN_SAC_COMPENSATION_THRESHOLD": ("mcts", "queen_sac_compensation_threshold"),
    "QUEEN_CHECK_DISCOUNT": ("mcts", "queen_check_discount"),
    "NUM_WORKERS": ("selfplay", "num_workers"),
    "GAMES_PER_WORKER": ("selfplay", "games_per_worker"),
    "MAX_GAME_LENGTH": ("selfplay", "max_game_length"),
    "OPENING_RANDOM_PLIES_MIN": ("selfplay", "opening_random_plies_min"),
    "OPENING_RANDOM_PLIES_MAX": ("selfplay", "opening_random_plies_max"),
    "REPETITION_PENALTY": ("selfplay", "repetition_penalty"),
    "REPETITION_BREAK_COUNT": ("selfplay", "repetition_break_count"),
    "REPETITION_MOVE_WEIGHT": ("selfplay", "repetition_move_weight"),
    "REPETITION_DRAW_VALUE": ("selfplay", "repetition_draw_value"),
    "MAX_LENGTH_DRAW_VALUE": ("selfplay", "max_length_draw_value"),
    "TEMPERATURE_HIGH_MOVES": ("selfplay", "temperature_high_moves"),
    "TEMPERATURE_MID_MOVES": ("selfplay", "temperature_mid_moves"),
    "TEMPERATURE_HIGH": ("selfplay", "temperature_high"),
    "TEMPERATURE_MID": ("selfplay", "temperature_mid"),
    "TEMPERATURE_LOW": ("selfplay", "temperature_low"),
    "ARENA_GAMES": ("arena", "games"),
    "UPDATE_THRESHOLD": ("arena", "update_threshold"),
    "ARENA_EARLY_STOP_MARGIN": ("arena", "early_stop_margin"),
    "BENCHMARK_GAMES": ("arena", "benchmark_games"),
    "BASELINE_SEARCH_DEPTH": ("arena", "baseline_search_depth"),
    "ELO_K_FACTOR": ("arena", "elo_k_factor"),
    "INITIAL_ELO": ("arena", "initial_elo"),
    "ARENA_SEARCH_TEMPERATURE": ("arena", "search_temperature"),
    "ARENA_RESIGN_THRESHOLD": ("arena", "resign_threshold"),
    "MAX_REPETITION_DRAW_RATE": ("arena", "max_repetition_draw_rate"),
    "ARENA_REPETITION_MOVE_WEIGHT": ("arena", "repetition_move_weight"),
    "ARENA_HARD_BLOCK_REPETITION": ("arena", "hard_block_repetition"),
    "ARENA_CONTEMPT_FACTOR": ("arena", "contempt_factor"),
    "LOG_LEVEL": ("system", "log_level"),
    "JSON_LOGS": ("system", "json_logs"),
    "SELFPLAY_DEVICE": ("system", "device"),
    "CHECKPOINT_PATH": ("system", "checkpoint_path"),
    "DEFAULT_BESTMOVE_SIMULATIONS": ("system", "default_bestmove_simulations"),
    "MAX_HALFMOVE": ("system", "max_halfmove"),
    "MAX_FULLMOVE": ("system", "max_fullmove"),
    "CPU_THREADS": ("system", "cpu_threads"),
    "INTEROP_THREADS": ("system", "interop_threads"),
    "WORKER_THREAD_POLICY": ("system", "worker_thread_policy"),
}


class _ConfigProxy:
    def __getattr__(self, name: str) -> Any:
        return get_config_value(name)

    def get(self, name: str, default: Any = None) -> Any:
        try:
            return get_config_value(name)
        except AttributeError:
            return default


Config = _ConfigProxy()


def get_current_config() -> AppConfig:
    return _CURRENT_CONFIG


def config_to_dict(cfg: AppConfig | None = None) -> dict[str, Any]:
    return asdict(cfg or _CURRENT_CONFIG)


def config_as_dict(cfg: AppConfig | None = None) -> dict[str, Any]:
    data = config_to_dict(cfg)
    flat = {}
    for key, (section, field) in _FLAT_MAP.items():
        flat[key] = data[section][field]
    return flat


def _normalize_overrides(overrides: dict[str, Any]) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for key, value in (overrides or {}).items():
        original_key = key
        if key in {"board", "evaluation"}:
            key = "model" if key == "board" else "arena"
        if key in {"model", "training", "external", "replay", "penalty_diagnostics", "principle_penalties", "mcts", "selfplay", "arena", "system"}:
            for sub_key, sub_value in (value or {}).items():
                if sub_key in _FLAT_MAP:
                    section, field = _FLAT_MAP[sub_key]
                    target_section = key if original_key != "board" else "model"
                    if section != target_section:
                        continue
                    normalized.setdefault(target_section, {})[field] = sub_value
                else:
                    normalized.setdefault(key, {})[sub_key] = sub_value
            continue
        if key in _FLAT_MAP:
            section, field = _FLAT_MAP[key]
            normalized.setdefault(section, {})[field] = value
            continue
        raise KeyError(f"Unknown config key or section: {key}")
    return normalized


def _deep_update(cfg: AppConfig, overrides: dict[str, Any]) -> AppConfig:
    data = asdict(cfg)
    for section, values in _normalize_overrides(overrides).items():
        data[section].update(values)
    return AppConfig(
        model=ModelConfig(**data["model"]),
        training=TrainingConfig(**data["training"]),
        external=ExternalDataConfig(**data["external"]),
        replay=ReplayConfig(**data["replay"]),
        penalty_diagnostics=PenaltyDiagnosticsConfig(**data["penalty_diagnostics"]),
        principle_penalties=PrinciplePenaltiesConfig(**data["principle_penalties"]),
        mcts=MCTSConfig(**data["mcts"]),
        selfplay=SelfPlayConfig(**data["selfplay"]),
        arena=ArenaConfig(**data["arena"]),
        system=SystemConfig(**data["system"]),
    )


def apply_overrides(overrides: dict[str, Any]) -> AppConfig:
    global _CURRENT_CONFIG
    _CURRENT_CONFIG = _deep_update(_CURRENT_CONFIG, overrides)
    validate(_CURRENT_CONFIG)
    return _CURRENT_CONFIG


def get_config_value(name: str, cfg: AppConfig | None = None) -> Any:
    cfg = cfg or _CURRENT_CONFIG
    if name in _FLAT_MAP:
        section, field = _FLAT_MAP[name]
        return getattr(getattr(cfg, section), field)
    if hasattr(cfg, name.lower()):
        return getattr(cfg, name.lower())
    raise AttributeError(name)


def validate(cfg: AppConfig | None = None) -> None:
    cfg = cfg or _CURRENT_CONFIG

    if cfg.model.input_planes < 20:
        raise ValueError("model.input_planes must be >= 20")
    if cfg.model.channels <= 0:
        raise ValueError("model.channels must be > 0")
    if cfg.model.res_blocks < 0:
        raise ValueError("model.res_blocks must be >= 0")
    if not (0.0 <= cfg.model.value_dropout < 1.0):
        raise ValueError("model.value_dropout must be in [0, 1)")

    if cfg.training.lr <= 0:
        raise ValueError("training.lr must be > 0")
    if cfg.training.min_lr <= 0 or cfg.training.min_lr > cfg.training.lr:
        raise ValueError("training.min_lr must be > 0 and <= training.lr")
    if cfg.training.batch_size <= 0:
        raise ValueError("training.batch_size must be > 0")
    if cfg.training.epochs <= 0:
        raise ValueError("training.epochs must be > 0")
    if cfg.training.train_steps_per_iter <= 0:
        raise ValueError("training.train_steps_per_iter must be > 0")
    if not (0.0 <= cfg.training.policy_label_smoothing < 1.0):
        raise ValueError("training.policy_label_smoothing must be in [0, 1)")

    if cfg.external.max_samples < 0:
        raise ValueError("external.max_samples must be >= 0")
    if cfg.external.min_fullmove < 0:
        raise ValueError("external.min_fullmove must be >= 0")
    if cfg.external.max_fullmove < 0:
        raise ValueError("external.max_fullmove must be >= 0")
    if cfg.external.min_fullmove and cfg.external.max_fullmove and cfg.external.max_fullmove < cfg.external.min_fullmove:
        raise ValueError("external.max_fullmove must be >= external.min_fullmove")
    if not (0.0 <= cfg.external.validation_split < 1.0):
        raise ValueError("external.validation_split must be in [0, 1)")
    if cfg.external.benchmark_games < 0:
        raise ValueError("external.benchmark_games must be >= 0")

    if cfg.replay.capacity <= 0:
        raise ValueError("replay.capacity must be > 0")
    if cfg.replay.beta_start > cfg.replay.beta_end:
        raise ValueError("replay.beta_start must be <= replay.beta_end")
    if not (0.0 <= cfg.replay.recent_sample_fraction <= 1.0):
        raise ValueError("replay.recent_sample_fraction must be in [0, 1]")
    if cfg.replay.recent_window_size < 0:
        raise ValueError("replay.recent_window_size must be >= 0")
    if not (0.0 <= cfg.replay.policy_mix <= 1.0):
        raise ValueError("replay.policy_mix must be in [0, 1]")
    if not (0.0 <= cfg.replay.max_draw_fraction <= 1.0):
        raise ValueError("replay.max_draw_fraction must be in [0, 1]")
    if cfg.replay.draw_value_threshold < 0.0:
        raise ValueError("replay.draw_value_threshold must be >= 0")
    if cfg.replay.save_interval <= 0:
        raise ValueError("replay.save_interval must be > 0")
    if cfg.replay.save_shard_size <= 0:
        raise ValueError("replay.save_shard_size must be > 0")

    _validate_principle_penalties(cfg.principle_penalties)

    if cfg.mcts.num_simulations <= 0:
        raise ValueError("mcts.num_simulations must be > 0")
    if cfg.mcts.inference_batch_size <= 0:
        raise ValueError("mcts.inference_batch_size must be > 0")
    if cfg.mcts.virtual_loss < 0.0:
        raise ValueError("mcts.virtual_loss must be >= 0")
    if not (0.0 <= cfg.mcts.classical_value_alpha <= 1.0):
        raise ValueError("mcts.classical_value_alpha must be in [0, 1]")
    if cfg.mcts.queen_blunder_penalty < 0.0:
        raise ValueError("mcts.queen_blunder_penalty must be >= 0")
    if cfg.mcts.queen_hanging_penalty < 0.0:
        raise ValueError("mcts.queen_hanging_penalty must be >= 0")
    if cfg.mcts.queen_sac_compensation_threshold < 0.0:
        raise ValueError("mcts.queen_sac_compensation_threshold must be >= 0")
    if not (0.0 <= cfg.mcts.queen_check_discount <= 1.0):
        raise ValueError("mcts.queen_check_discount must be in [0, 1]")
    _validate_piece_penalties(cfg.mcts.piece_penalties)

    if cfg.selfplay.max_game_length <= 0:
        raise ValueError("selfplay.max_game_length must be > 0")
    if cfg.selfplay.repetition_break_count < 2:
        raise ValueError("selfplay.repetition_break_count must be >= 2")
    if not (0.0 <= cfg.selfplay.repetition_move_weight <= 1.0):
        raise ValueError("selfplay.repetition_move_weight must be in [0, 1]")
    if not (0.0 <= cfg.selfplay.repetition_draw_value <= 1.0):
        raise ValueError("selfplay.repetition_draw_value must be in [0, 1]")
    if not (0.0 <= cfg.selfplay.max_length_draw_value <= 1.0):
        raise ValueError("selfplay.max_length_draw_value must be in [0, 1]")

    if cfg.arena.games <= 0:
        raise ValueError("arena.games must be > 0")
    if not (0.0 <= cfg.arena.update_threshold <= 1.0):
        raise ValueError("arena.update_threshold must be in [0, 1]")
    if not (0.0 <= cfg.arena.early_stop_margin <= 1.0):
        raise ValueError("arena.early_stop_margin must be in [0, 1]")
    if not (0.0 <= cfg.arena.max_repetition_draw_rate <= 1.0):
        raise ValueError("arena.max_repetition_draw_rate must be in [0, 1]")
    if cfg.arena.repetition_break_count < 2:
        raise ValueError("arena.repetition_break_count must be >= 2")
    if cfg.arena.repetition_soft_limit_plies < 0:
        raise ValueError("arena.repetition_soft_limit_plies must be >= 0")
    if not (0.0 <= cfg.arena.repetition_move_weight <= 1.0):
        raise ValueError("arena.repetition_move_weight must be in [0, 1]")
    if not (-1.0 <= cfg.arena.contempt_factor <= 1.0):
        raise ValueError("arena.contempt_factor must be in [-1, 1]")

    if cfg.system.device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("system.device must be one of auto/cpu/cuda/mps")
    if cfg.system.cpu_threads < 0:
        raise ValueError("system.cpu_threads must be >= 0")
    if cfg.system.interop_threads < 0:
        raise ValueError("system.interop_threads must be >= 0")
    if cfg.system.worker_thread_policy not in {"auto", "per_worker", "fixed"}:
        raise ValueError("system.worker_thread_policy must be one of auto/per_worker/fixed")


validate_config = validate


def _validate_principle_penalties(principles: PrinciplePenaltiesConfig) -> None:
    if principles.max_total_per_move < 0.0:
        raise ValueError("principle_penalties.max_total_per_move must be >= 0")
    if principles.max_total_per_move > 0.5:
        raise ValueError("principle_penalties.max_total_per_move is suspiciously large; expected <= 0.5")

    for field_name in (
        "king_safety",
        "opening_development",
        "center_control",
        "tactics",
        "pawn_structure",
        "piece_activity",
        "rook_activity",
        "endgame",
    ):
        value = float(getattr(principles, field_name))
        if value < 0.0:
            raise ValueError(f"principle_penalties.{field_name} must be >= 0")
        if value > 0.25:
            raise ValueError(f"principle_penalties.{field_name} is suspiciously large; expected <= 0.25")


def _validate_piece_penalties(piece_penalties: dict[str, PiecePenaltyCfg]) -> None:
    if not isinstance(piece_penalties, dict):
        raise ValueError("mcts.piece_penalties must be a mapping")

    limits = {
        "blunder_penalty": 1.0,
        "hanging_penalty": 1.0,
        "sac_compensation_threshold": 5000.0,
        "check_discount": 1.0,
    }
    allowed = set(limits)

    for piece_name, values in piece_penalties.items():
        if not isinstance(values, dict):
            raise ValueError(f"mcts.piece_penalties.{piece_name} must be a mapping")
        for field_name, raw_value in values.items():
            if field_name not in allowed:
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"mcts.piece_penalties.{piece_name}.{field_name} must be numeric") from exc
            if value < 0.0:
                raise ValueError(f"mcts.piece_penalties.{piece_name}.{field_name} must be >= 0")
            if value > limits[field_name]:
                raise ValueError(
                    f"mcts.piece_penalties.{piece_name}.{field_name}={value:g} is suspiciously large; "
                    f"expected <= {limits[field_name]:g}"
                )


def _resolve_config_path(path: str | Path | None) -> Path | None:
    if path is None:
        return DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.exists() else None
    p = Path(path)
    if p.exists():
        return p
    if not p.is_absolute():
        candidate = Path.cwd() / p
        if candidate.exists():
            return candidate
        candidate = DEFAULT_CONFIG_PATH.parent / p.name
        if candidate.exists():
            return candidate
        candidate = DEFAULT_CONFIG_PATH.parent.parent / p
        if candidate.exists():
            return candidate
    return p


def _read_config_file(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        if path.suffix in {".yaml", ".yml"}:
            return yaml.safe_load(f) or {}
        if path.suffix == ".json":
            return json.load(f)
    raise ValueError("Only YAML or JSON supported")


def load_config(path: str | Path | None = None, base: AppConfig = AppConfig()) -> AppConfig:
    global _CURRENT_CONFIG
    cfg = base
    default_path = _resolve_config_path(None)
    if default_path is not None and default_path.exists():
        cfg = _deep_update(cfg, _read_config_file(default_path))
    resolved = _resolve_config_path(path)
    if resolved is not None and default_path is not None and resolved.resolve() != default_path.resolve():
        cfg = _deep_update(cfg, _read_config_file(resolved))
    validate(cfg)
    _CURRENT_CONFIG = cfg
    return cfg
