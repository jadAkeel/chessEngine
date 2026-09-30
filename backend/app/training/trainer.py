from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from app.game.move_encoding import NUM_MOVES, index_to_move, move_to_index
from app.infra.config import AppConfig, get_current_config
from app.training.dataset import SparsePolicyBatchTensor, batch_to_tensors

_HFLIP_INDEX_MAP = None
_HFLIP_INDEX_TENSORS: dict[str, torch.Tensor] = {}


def _normalize_policy_targets(policy_targets: torch.Tensor) -> torch.Tensor:
    policy_targets = torch.clamp(policy_targets, min=0.0)
    sums = policy_targets.sum(dim=1, keepdim=True)
    safe_sums = torch.where(sums > 1e-12, sums, torch.ones_like(sums))
    return policy_targets / safe_sums


def _apply_policy_label_smoothing(policy_targets: torch.Tensor, epsilon: float) -> torch.Tensor:
    epsilon = float(max(0.0, min(1.0, epsilon)))
    if epsilon <= 0.0:
        return policy_targets
    num_actions = policy_targets.size(1)
    uniform = torch.full_like(policy_targets, 1.0 / num_actions)
    return ((1.0 - epsilon) * policy_targets) + (epsilon * uniform)


def _sparse_policy_mask(policy_targets: SparsePolicyBatchTensor) -> torch.Tensor:
    if policy_targets.indices.ndim != 2 or policy_targets.indices.size(1) == 0:
        return torch.zeros(
            (policy_targets.batch_size, 0),
            dtype=torch.bool,
            device=policy_targets.lengths.device,
        )
    cols = torch.arange(policy_targets.indices.size(1), device=policy_targets.indices.device)
    return cols.unsqueeze(0) < policy_targets.lengths.unsqueeze(1)



def _normalize_sparse_policy_targets(policy_targets: SparsePolicyBatchTensor) -> SparsePolicyBatchTensor:
    mask = _sparse_policy_mask(policy_targets)
    probs = torch.clamp(policy_targets.probs, min=0.0)
    if mask.numel():
        probs = torch.where(mask, probs, torch.zeros_like(probs))
    sums = probs.sum(dim=1, keepdim=True)
    safe_sums = torch.where(sums > 1e-12, sums, torch.ones_like(sums))
    probs = probs / safe_sums
    if mask.numel():
        probs = torch.where(mask, probs, torch.zeros_like(probs))
    return SparsePolicyBatchTensor(
        indices=policy_targets.indices,
        probs=probs,
        lengths=policy_targets.lengths,
        num_actions=policy_targets.num_actions,
    )



def _dense_policy_loss(logits: torch.Tensor, policies: torch.Tensor, label_smoothing: float):
    policies = _normalize_policy_targets(policies)
    policies = _apply_policy_label_smoothing(policies, label_smoothing)
    policies = _normalize_policy_targets(policies)
    log_probs = F.log_softmax(logits, dim=1)
    pred_probs = torch.softmax(logits, dim=1)
    per_sample_policy = -(policies * log_probs).sum(dim=1)
    entropy = -(pred_probs * log_probs).sum(dim=1).mean()
    return per_sample_policy, log_probs, pred_probs, entropy



def _sparse_policy_loss(logits: torch.Tensor, policy_targets: SparsePolicyBatchTensor, label_smoothing: float):
    log_probs = F.log_softmax(logits, dim=1)
    pred_probs = torch.softmax(logits, dim=1)

    if policy_targets.indices.numel() > 0 and policy_targets.indices.size(1) > 0:
        gathered_log_probs = log_probs.gather(1, policy_targets.indices)
        mask = _sparse_policy_mask(policy_targets).to(gathered_log_probs.dtype)
        sparse_ce = -(policy_targets.probs * gathered_log_probs * mask).sum(dim=1)
        has_mass = policy_targets.lengths > 0
    else:
        sparse_ce = torch.zeros(logits.size(0), dtype=log_probs.dtype, device=log_probs.device)
        has_mass = torch.zeros(logits.size(0), dtype=torch.bool, device=logits.device)

    label_smoothing = float(max(0.0, min(1.0, label_smoothing)))
    if label_smoothing > 0.0:
        uniform_ce = -log_probs.mean(dim=1)
        mixed_ce = ((1.0 - label_smoothing) * sparse_ce) + (label_smoothing * uniform_ce)
        per_sample_policy = torch.where(has_mass, mixed_ce, uniform_ce)
    else:
        per_sample_policy = torch.where(has_mass, sparse_ce, torch.zeros_like(sparse_ce))

    entropy = -(pred_probs * log_probs).sum(dim=1).mean()
    return per_sample_policy, log_probs, pred_probs, entropy


def _masked_value_loss(pred_values: torch.Tensor, values: torch.Tensor, weights: torch.Tensor):
    """MSE over samples with a finite value target; NaN marks policy-only samples.

    Returns the per-sample loss (zero where masked), the weighted mean over the
    unmasked samples, and the mask.
    """
    values = values.view(-1)
    mask = torch.isfinite(values)
    safe_targets = torch.where(mask, values, torch.zeros_like(values))
    per_sample = F.mse_loss(pred_values.view(-1), safe_targets, reduction='none')
    per_sample = torch.where(mask, per_sample, torch.zeros_like(per_sample))
    loss = (weights * per_sample).sum() / mask.sum().clamp(min=1)
    return per_sample, loss, mask


def _flip_square_horizontal(square: int) -> int:
    import chess
    rank = chess.square_rank(square)
    file = chess.square_file(square)
    return chess.square(7 - file, rank)


def _build_hflip_index_map():
    global _HFLIP_INDEX_MAP
    if _HFLIP_INDEX_MAP is not None:
        return _HFLIP_INDEX_MAP
    mapping = np.zeros(NUM_MOVES, dtype=np.int64)
    for idx in range(NUM_MOVES):
        move = index_to_move(idx, board=None)
        if move is None:
            mapping[idx] = idx
            continue
        import chess
        flipped = chess.Move(
            _flip_square_horizontal(move.from_square),
            _flip_square_horizontal(move.to_square),
            promotion=move.promotion,
        )
        try:
            mapping[idx] = move_to_index(flipped)
        except ValueError:
            mapping[idx] = idx
    _HFLIP_INDEX_MAP = mapping
    return mapping


def _hflip_index_map_tensor(device: torch.device) -> torch.Tensor:
    key = str(device)
    tensor = _HFLIP_INDEX_TENSORS.get(key)
    if tensor is None:
        tensor = torch.from_numpy(_build_hflip_index_map()).to(device)
        _HFLIP_INDEX_TENSORS[key] = tensor
    return tensor



def _smart_horizontal_flip(states, policies):
    batch_size = states.size(0)
    device = states.device
    mask = torch.rand(batch_size, device=device) < 0.5
    # File reflection moves the king from e to d: standard castling is no longer
    # legal in the reflected position. Only augment positions without rights.
    has_castling_rights = states[:, 13:17].ne(0).flatten(1).any(dim=1)
    mask &= ~has_castling_rights
    if not mask.any():
        return states, policies
    idx = mask.nonzero(as_tuple=True)[0]
    flipped = torch.flip(states[idx].clone(), dims=[3])
    flipped[:, 13], flipped[:, 14] = flipped[:, 14].clone(), flipped[:, 13].clone()
    flipped[:, 15], flipped[:, 16] = flipped[:, 16].clone(), flipped[:, 15].clone()
    states[idx] = flipped

    if isinstance(policies, SparsePolicyBatchTensor):
        if policies.indices.numel() > 0:
            mapping_t = _hflip_index_map_tensor(device)
            policies.indices[idx] = mapping_t[policies.indices[idx]]
        return states, policies

    mapping_t = _hflip_index_map_tensor(device)
    policies[idx] = policies[idx].index_select(1, mapping_t)
    return states, policies


# A non-finite forward pass still updates BatchNorm running stats, so skipping
# the optimizer step (what GradScaler does) is not enough: the Kaggle run of
# 2026-09-23 went NaN mid-iteration and every later evaluation was NaN. Keep a
# known-good copy of the weights and roll back to it instead.
GOOD_STATE_EVERY = 500
PARAM_CHECK_EVERY = 50
MAX_NONFINITE_STEPS = 200


DEBUG_EVERY = 50


def _forward_module(model, device):
    """The module to run training batches through: ``nn.DataParallel`` over every
    visible GPU when there is more than one (Kaggle gives two T4s), else the model.

    The wrapper shares the model's parameters, so the optimizer, checkpoints and
    state_dict keys are unchanged; only the forward pass is split across GPUs.
    """
    if not str(device).startswith('cuda') or not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return model
    print(f"[TRAIN] DataParallel over {torch.cuda.device_count()} GPUs", flush=True)
    return torch.nn.DataParallel(model)


def _params_finite(model) -> bool:
    checks = [torch.isfinite(t).all() for t in model.state_dict().values() if t.is_floating_point()]
    return bool(torch.stack(checks).all().item()) if checks else True


def _snapshot_good_state(model, optimizer):
    return (
        {k: v.detach().clone() for k, v in model.state_dict().items()},
        copy.deepcopy(optimizer.state_dict()),
    )


def _restore_good_state(model, optimizer, snapshot) -> None:
    model.load_state_dict(snapshot[0])
    # load_state_dict may alias same-device tensors; keep the snapshot pristine.
    optimizer.load_state_dict(copy.deepcopy(snapshot[1]))


def train_model(
    model,
    optimizer,
    buffer,
    device='cpu',
    scheduler=None,
    global_step=0,
    scaler: GradScaler | None = None,
    cfg: AppConfig | None = None,
):
    cfg = cfg or getattr(model, 'cfg', None) or get_current_config()

    if len(buffer) == 0:
        return {
            'loss': 0.0,
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'lr': optimizer.param_groups[0]['lr'],
            'steps': 0,
            'global_step': global_step,
            'amp_enabled': False,
        }

    model.train()
    losses = []
    policy_losses = []
    value_losses = []
    entropies = []

    use_amp = bool(cfg.training.use_amp and str(device).startswith('cuda'))
    if scaler is None:
        scaler = GradScaler(enabled=use_amp)

    total_steps = max(1, int(cfg.training.epochs) * int(cfg.training.train_steps_per_iter))
    grad_clip_norm = float(cfg.training.grad_clip)
    value_loss_coeff = float(cfg.training.value_loss_coeff)
    entropy_coeff = float(cfg.training.entropy_coeff)
    label_smoothing = float(getattr(cfg.training, 'policy_label_smoothing', 0.0))
    enable_hflip = bool(cfg.training.enable_horizontal_flip_augment)
    beta_start = float(cfg.replay.beta_start)
    beta_end = float(cfg.replay.beta_end)

    if not _params_finite(model):
        raise RuntimeError("Model weights are already non-finite before training; refusing to train")
    good_state = _snapshot_good_state(model, optimizer)
    nonfinite_steps = 0

    def _roll_back(reason: str) -> None:
        nonlocal nonfinite_steps
        nonfinite_steps += 1
        _restore_good_state(model, optimizer, good_state)
        print(f"[TRAIN GUARD] step={step} {reason}; rolled back to last good weights ({nonfinite_steps} so far)")
        if nonfinite_steps > MAX_NONFINITE_STEPS:
            raise RuntimeError(f"Training diverged: {nonfinite_steps} non-finite steps in one iteration")

    forward = _forward_module(model, device)
    window_started = time.perf_counter()

    for step in range(total_steps):
        beta = beta_start + (beta_end - beta_start) * min(1.0, global_step / 2000.0)

        states, policies, values, indices, is_weights = buffer.sample_batch(beta=beta)
        if len(states) == 0:
            continue

        states, policies, values, is_weights = batch_to_tensors(states, policies, values, is_weights, device=device)

        if is_weights.dim() > 1:
            is_weights = is_weights.view(-1)

        if isinstance(policies, SparsePolicyBatchTensor):
            policies = _normalize_sparse_policy_targets(policies)
        else:
            policies = _normalize_policy_targets(policies)

        if enable_hflip:
            states, policies = _smart_horizontal_flip(states, policies)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type='cuda', enabled=use_amp):
            try:
                pred_policies, pred_values = forward(states)
            except RuntimeError as exc:
                if forward is model:
                    raise
                print(f"[TRAIN] DataParallel failed ({exc}); continuing on one GPU", flush=True)
                forward = model
                pred_policies, pred_values = model(states)

            if isinstance(policies, SparsePolicyBatchTensor):
                per_sample_policy, log_probs, pred_probs, entropy = _sparse_policy_loss(
                    pred_policies,
                    policies,
                    label_smoothing,
                )
            else:
                per_sample_policy, log_probs, pred_probs, entropy = _dense_policy_loss(
                    pred_policies,
                    policies,
                    label_smoothing,
                )

            # Each .item() waits for the GPU; only collect the debug stats when printed.
            log_debug = step % DEBUG_EVERY == 0
            if log_debug:
                with torch.no_grad():
                    pred_value_mean = float(pred_values.mean().item())
                    pred_value_std = float(pred_values.std().item())
                    pred_value_abs = float(pred_values.abs().mean().item())

                    finite_values = values[torch.isfinite(values)]
                    target_value_mean = float(finite_values.mean().item()) if finite_values.numel() else 0.0
                    target_value_std = float(finite_values.std().item()) if finite_values.numel() > 1 else 0.0
                    value_target_ratio = float(finite_values.numel()) / max(1, int(values.numel()))

                    top_k = min(2, pred_probs.size(1))
                    top_probs, _ = torch.topk(pred_probs, k=top_k, dim=1)
                    top1_mean = float(top_probs[:, 0].mean().item())
                    top2_mean = float(top_probs[:, 1].mean().item()) if top_k > 1 else 0.0
                    gap_mean = float((top_probs[:, 0] - top_probs[:, 1]).mean().item()) if top_k > 1 else 0.0

                    entropy_val = float((-(pred_probs * log_probs).sum(dim=1)).mean().item())
                    draw_ratio = float((finite_values.abs() < 0.1).float().mean().item()) if finite_values.numel() else 0.0

            per_sample_value, value_loss, _ = _masked_value_loss(pred_values, values, is_weights)

            policy_loss = (is_weights * per_sample_policy).mean()
            loss = policy_loss + value_loss_coeff * value_loss - entropy_coeff * entropy

        if not bool(torch.isfinite(loss).item()):
            _roll_back("non-finite loss")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        if step % PARAM_CHECK_EVERY == 0:
            if not _params_finite(model):
                _roll_back("non-finite weights after optimizer step")
                continue
            if step % GOOD_STATE_EVERY == 0:
                good_state = _snapshot_good_state(model, optimizer)

        if scheduler is not None and cfg.training.step_scheduler_per_batch:
            scheduler.step()

        if hasattr(buffer, 'update_priorities') and indices is not None:
            priority_signal = per_sample_policy.detach() + (value_loss_coeff * per_sample_value.detach())
            buffer.update_priorities(indices, torch.clamp(priority_signal, min=1e-6).cpu().numpy())

        losses.append(float(loss.item()))
        policy_losses.append(float(policy_loss.item()))
        value_losses.append(float(value_loss.item()))
        entropies.append(float(entropy.item()))

        global_step += 1

        if log_debug:
            now = time.perf_counter()
            sec_per_step = (now - window_started) / (DEBUG_EVERY if step else 1)
            window_started = now
            print(
                f"[TRAIN DEBUG] step={step} "
                f"sec_per_step={sec_per_step:.3f} "
                f"pred_value_mean={pred_value_mean:.3f} "
                f"pred_value_std={pred_value_std:.3f} "
                f"pred_value_abs={pred_value_abs:.3f} "
                f"target_mean={target_value_mean:.3f} "
                f"target_std={target_value_std:.3f} "
                f"top1={top1_mean:.3f} "
                f"top2={top2_mean:.3f} "
                f"gap={gap_mean:.3f} "
                f"entropy={entropy_val:.3f} "
                f"draw_ratio={draw_ratio:.3f} "
                f"value_targets={value_target_ratio:.3f}",
                flush=True,
            )

    if not losses:
        return {
            'loss': 0.0,
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'lr': optimizer.param_groups[0]['lr'],
            'steps': 0,
            'global_step': global_step,
            'amp_enabled': bool(use_amp),
        }

    return {
        'loss': float(np.mean(losses)),
        'policy_loss': float(np.mean(policy_losses)),
        'value_loss': float(np.mean(value_losses)),
        'entropy': float(np.mean(entropies)),
        'lr': float(optimizer.param_groups[0]['lr']),
        'steps': len(losses),
        'global_step': global_step,
        'amp_enabled': bool(use_amp),
        'nonfinite_steps': nonfinite_steps,
    }


def evaluate_model_on_samples(
    model,
    samples,
    *,
    device='cpu',
    batch_size: int | None = None,
    cfg: AppConfig | None = None,
):
    cfg = cfg or getattr(model, 'cfg', None) or get_current_config()
    if not samples:
        return {
            'loss': 0.0,
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'batches': 0,
            'samples': 0,
        }

    batch_size = max(1, int(batch_size or cfg.training.batch_size))
    value_loss_coeff = float(cfg.training.value_loss_coeff)
    entropy_coeff = float(cfg.training.entropy_coeff)
    label_smoothing = float(getattr(cfg.training, 'policy_label_smoothing', 0.0))

    policy_losses = []
    entropies = []
    # Value loss is averaged over samples that carry a value target, not per
    # batch, so batches of policy-only samples do not pull it towards zero.
    value_loss_sum = 0.0
    value_count = 0

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = samples[start:start + batch_size]
            states = [sample[0] for sample in batch]
            policies = [sample[1] for sample in batch]
            values = [sample[2] for sample in batch]
            weights = np.ones((len(batch),), dtype=np.float32)

            states_t, policies_t, values_t, weights_t = batch_to_tensors(
                states,
                policies,
                values,
                weights,
                device=device,
            )

            if isinstance(policies_t, SparsePolicyBatchTensor):
                policies_t = _normalize_sparse_policy_targets(policies_t)
            else:
                policies_t = _normalize_policy_targets(policies_t)

            pred_policies, pred_values = model(states_t)

            if isinstance(policies_t, SparsePolicyBatchTensor):
                per_sample_policy, _, _, entropy = _sparse_policy_loss(
                    pred_policies,
                    policies_t,
                    label_smoothing,
                )
            else:
                per_sample_policy, _, _, entropy = _dense_policy_loss(
                    pred_policies,
                    policies_t,
                    label_smoothing,
                )

            _, value_loss, value_mask = _masked_value_loss(pred_values, values_t, weights_t)
            policy_loss = (weights_t * per_sample_policy).mean()
            batch_value_count = int(value_mask.sum().item())

            policy_losses.append(float(policy_loss.item()))
            value_loss_sum += float(value_loss.item()) * batch_value_count
            value_count += batch_value_count
            entropies.append(float(entropy.item()))

    if was_training:
        model.train()

    policy_loss_mean = float(np.mean(policy_losses)) if policy_losses else 0.0
    value_loss_mean = value_loss_sum / value_count if value_count else 0.0
    entropy_mean = float(np.mean(entropies)) if entropies else 0.0
    return {
        'loss': policy_loss_mean + value_loss_coeff * value_loss_mean - entropy_coeff * entropy_mean,
        'policy_loss': policy_loss_mean,
        'value_loss': value_loss_mean,
        'entropy': entropy_mean,
        'batches': len(policy_losses),
        'samples': len(samples),
        'value_samples': value_count,
    }
