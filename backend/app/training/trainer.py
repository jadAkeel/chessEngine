from __future__ import annotations

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

            with torch.no_grad():
                pred_value_mean = float(pred_values.mean().item())
                pred_value_std = float(pred_values.std().item())
                pred_value_abs = float(pred_values.abs().mean().item())

                target_value_mean = float(values.mean().item())
                target_value_std = float(values.std().item())

                top_k = min(2, pred_probs.size(1))
                top_probs, _ = torch.topk(pred_probs, k=top_k, dim=1)
                top1_mean = float(top_probs[:, 0].mean().item())
                top2_mean = float(top_probs[:, 1].mean().item()) if top_k > 1 else 0.0
                gap_mean = float((top_probs[:, 0] - top_probs[:, 1]).mean().item()) if top_k > 1 else 0.0

                entropy_val = float((-(pred_probs * log_probs).sum(dim=1)).mean().item())
                draw_ratio = float((values.abs() < 0.1).float().mean().item())

            per_sample_value = F.mse_loss(pred_values, values, reduction='none').squeeze(1)

            policy_loss = (is_weights * per_sample_policy).mean()
            value_loss = (is_weights * per_sample_value).mean()
            loss = policy_loss + value_loss_coeff * value_loss - entropy_coeff * entropy

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

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

        if step % 50 == 0:
            print(
                f"[TRAIN DEBUG] step={step} "
                f"pred_value_mean={pred_value_mean:.3f} "
                f"pred_value_std={pred_value_std:.3f} "
                f"pred_value_abs={pred_value_abs:.3f} "
                f"target_mean={target_value_mean:.3f} "
                f"target_std={target_value_std:.3f} "
                f"top1={top1_mean:.3f} "
                f"top2={top2_mean:.3f} "
                f"gap={gap_mean:.3f} "
                f"entropy={entropy_val:.3f} "
                f"draw_ratio={draw_ratio:.3f}"
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

    losses = []
    policy_losses = []
    value_losses = []
    entropies = []

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

            per_sample_value = F.mse_loss(pred_values, values_t, reduction='none').squeeze(1)
            policy_loss = (weights_t * per_sample_policy).mean()
            value_loss = (weights_t * per_sample_value).mean()
            loss = policy_loss + value_loss_coeff * value_loss - entropy_coeff * entropy

            losses.append(float(loss.item()))
            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropies.append(float(entropy.item()))

    if was_training:
        model.train()

    return {
        'loss': float(np.mean(losses)) if losses else 0.0,
        'policy_loss': float(np.mean(policy_losses)) if policy_losses else 0.0,
        'value_loss': float(np.mean(value_losses)) if value_losses else 0.0,
        'entropy': float(np.mean(entropies)) if entropies else 0.0,
        'batches': len(losses),
        'samples': len(samples),
    }
