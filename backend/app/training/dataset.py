from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from app.infra.config import SparsePolicyBatchTensor
from app.game.move_encoding import NUM_MOVES

@dataclass(frozen=True)
class SparsePolicyBatchTensor:
    indices: torch.Tensor
    probs: torch.Tensor
    lengths: torch.Tensor
    num_actions: int

    @property
    def batch_size(self) -> int:
        return int(self.lengths.numel())


def batch_to_tensors(states, policies, values, is_weights, device: str):
    if isinstance(states, np.ndarray):
        states_t = torch.as_tensor(states, device=device)
    else:
        states_t = torch.stack([
            state if isinstance(state, torch.Tensor) else torch.as_tensor(state)
            for state in states
        ]).to(device, non_blocking=True)

    # =========================
    # FIX: Duck Typing policies safe handling
    # =========================
    
    # الحالة الأولى: الكائن يحمل خصائص الـ Batch (سواء كان SparsePolicyBatch أو SparsePolicyBatchTensor)
    if hasattr(policies, 'indices') and hasattr(policies, 'probs') and hasattr(policies, 'lengths'):
        policies_t = SparsePolicyBatchTensor(
            indices=torch.as_tensor(policies.indices, dtype=torch.long, device=device),
            probs=torch.as_tensor(policies.probs, dtype=torch.float32, device=device),
            lengths=torch.as_tensor(policies.lengths, dtype=torch.long, device=device),
            num_actions=int(getattr(policies, 'num_actions', NUM_MOVES) or NUM_MOVES),
        )
        
    # الحالة الثانية: قائمة تحتوي على كائنات PackedPolicy
    elif isinstance(policies, list) and len(policies) > 0 and hasattr(policies[0], 'indices'):
        lengths = np.array([len(p.indices) for p in policies], dtype=np.int64)
        max_len = int(lengths.max()) if len(lengths) > 0 else 0
        
        padded_indices = np.zeros((len(policies), max_len), dtype=np.int64)
        padded_probs = np.zeros((len(policies), max_len), dtype=np.float32)
        
        for i, p in enumerate(policies):
            idx_len = len(p.indices)
            if idx_len > 0:
                padded_indices[i, :idx_len] = np.asarray(p.indices, dtype=np.int64)
                padded_probs[i, :idx_len] = np.asarray(p.probs, dtype=np.float32)
                
        policies_t = SparsePolicyBatchTensor(
            indices=torch.as_tensor(padded_indices, dtype=torch.long, device=device),
            probs=torch.as_tensor(padded_probs, dtype=torch.float32, device=device),
            lengths=torch.as_tensor(lengths, dtype=torch.long, device=device),
            num_actions=int(NUM_MOVES)
        )

    # الحالة الثالثة: بيانات عادية (مصفوفات أرقام)
    else:
        if isinstance(policies, list):
            policies = np.asarray(policies)

        if isinstance(policies, np.ndarray) and policies.dtype == np.object_:
            try:
                policies = np.stack(policies).astype(np.float32)
            except ValueError:
                max_len = max(len(p) for p in policies)
                padded = np.zeros((len(policies), max_len), dtype=np.float32)
                for i, p in enumerate(policies):
                    padded[i, :len(p)] = np.asarray(p, dtype=np.float32)
                policies = padded

        policies_t = torch.as_tensor(
            policies,
            dtype=torch.float32,
            device=device,
        )

    values_t = torch.as_tensor(values, dtype=torch.float32, device=device).unsqueeze(1)
    values_t = torch.clamp(values_t, -1.0, 1.0)

    weights_t = torch.as_tensor(is_weights, dtype=torch.float32, device=device)

    return states_t, policies_t, values_t, weights_t