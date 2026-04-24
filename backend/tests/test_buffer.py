from __future__ import annotations

import math
import numpy as np

from app.training.replay_buffer import ReplayBuffer
from app.infra.config import load_config


CHUNK_SIZE = 5000


def to_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    try:
        return np.asarray(x)
    except Exception:
        return None


def entropy_from_probs(probs: np.ndarray):
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    probs = probs[np.isfinite(probs)]
    if probs.size == 0:
        return None
    probs = probs[probs > 0.0]
    if probs.size == 0:
        return 0.0
    s = probs.sum()
    if s <= 0:
        return None
    probs = probs / s
    h = -np.sum(probs * np.log(probs + 1e-12))
    max_h = math.log(len(probs)) if len(probs) > 1 else 1.0
    return float(h / max_h) if max_h > 0 else 0.0


def try_unpack_policy(buf, i):
    p = buf.policies[i]

    # dense already
    arr = to_numpy(p)
    if arr is not None and arr.ndim >= 1:
        return arr.reshape(-1)

    # common possibilities
    for name in ["to_dense", "dense", "as_dense", "unpack"]:
        if hasattr(p, name):
            try:
                obj = getattr(p, name)
                dense = obj() if callable(obj) else obj
                arr = to_numpy(dense)
                if arr is not None:
                    return arr.reshape(-1)
            except Exception:
                pass

    # sparse-like possibilities
    indices = None
    values = None
    for idx_name in ["indices", "idx", "actions", "action_indices"]:
        if hasattr(p, idx_name):
            try:
                indices = to_numpy(getattr(p, idx_name))
                break
            except Exception:
                pass

    for val_name in ["values", "probs", "weights"]:
        if hasattr(p, val_name):
            try:
                values = to_numpy(getattr(p, val_name))
                break
            except Exception:
                pass

    if indices is not None and values is not None:
        dense = np.zeros(int(buf.policy_size), dtype=np.float32)
        indices = np.asarray(indices).reshape(-1)
        values = np.asarray(values).reshape(-1)
        n = min(len(indices), len(values))
        dense[indices[:n].astype(np.int64)] = values[:n]
        return dense

    return None


def analyze_chunk(buf, start, end):
    values = []
    priorities = []
    entropies = []
    policy_maxes = []
    policy_sums = []
    policy_nonzero = []

    zero_states = 0
    nan_states = 0
    nan_policies = 0
    duplicates = 0
    seen_hashes = set()

    for i in range(start, end):
        # ----- state -----
        try:
            s = to_numpy(buf.states[i])
            if s is not None:
                if not np.all(np.isfinite(s)):
                    nan_states += 1
                if np.abs(s).sum() == 0:
                    zero_states += 1

                h = hash(s.tobytes())
                if h in seen_hashes:
                    duplicates += 1
                else:
                    seen_hashes.add(h)
        except Exception:
            pass

        # ----- value -----
        try:
            v = float(buf.values[i])
            if np.isfinite(v):
                values.append(v)
        except Exception:
            pass

        # ----- priority -----
        try:
            p = float(buf.priorities[i])
            if np.isfinite(p):
                priorities.append(p)
        except Exception:
            pass

        # ----- policy -----
        try:
            pol = try_unpack_policy(buf, i)
            if pol is not None:
                pol = np.asarray(pol, dtype=np.float64).reshape(-1)
                if not np.all(np.isfinite(pol)):
                    nan_policies += 1
                policy_sums.append(float(pol.sum()))
                policy_maxes.append(float(pol.max()) if pol.size else 0.0)
                policy_nonzero.append(int(np.count_nonzero(pol > 1e-12)))
                ent = entropy_from_probs(pol)
                if ent is not None:
                    entropies.append(ent)
        except Exception:
            nan_policies += 1

    out = {
        "count": end - start,
        "values": values,
        "priorities": priorities,
        "entropies": entropies,
        "policy_maxes": policy_maxes,
        "policy_sums": policy_sums,
        "policy_nonzero": policy_nonzero,
        "zero_states": zero_states,
        "nan_states": nan_states,
        "nan_policies": nan_policies,
        "duplicate_fraction": duplicates / max(1, end - start),
    }
    return out


def print_chunk_report(chunk_id, start, end, rep):
    print(f"\n===== CHUNK {chunk_id} [{start} -> {end}] =====")

    values = np.array(rep["values"], dtype=np.float64) if rep["values"] else np.array([])
    if values.size:
        print(f"value count: {values.size}")
        print(f"value mean/std: {values.mean():.4f} / {values.std():.4f}")
        print(f"value min/max: {values.min():.2f} / {values.max():.2f}")
        print(f"draw_frac(|v|<=0.1): {np.mean(np.abs(values) <= 0.1):.3f}")
        print(f"white_win_frac(v>0.5): {np.mean(values > 0.5):.3f}")
        print(f"black_win_frac(v<-0.5): {np.mean(values < -0.5):.3f}")
    else:
        print("No values found.")

    priorities = np.array(rep["priorities"], dtype=np.float64) if rep["priorities"] else np.array([])
    if priorities.size:
        print(f"priority mean/std: {priorities.mean():.4f} / {priorities.std():.4f}")
        print(f"priority min/max: {priorities.min():.4f} / {priorities.max():.4f}")
    else:
        print("No priorities found.")

    entropies = np.array(rep["entropies"], dtype=np.float64) if rep["entropies"] else np.array([])
    if entropies.size:
        policy_maxes = np.array(rep["policy_maxes"], dtype=np.float64)
        policy_sums = np.array(rep["policy_sums"], dtype=np.float64)
        policy_nonzero = np.array(rep["policy_nonzero"], dtype=np.float64)

        print(f"policy entropy mean/std: {entropies.mean():.4f} / {entropies.std():.4f}")
        print(f"low_entropy_frac(<0.25): {np.mean(entropies < 0.25):.3f}")
        print(f"high_entropy_frac(>0.85): {np.mean(entropies > 0.85):.3f}")
        print(f"policy max mean/std: {policy_maxes.mean():.4f} / {policy_maxes.std():.4f}")
        print(f"policy sum mean/std: {policy_sums.mean():.4f} / {policy_sums.std():.4f}")
        print(f"bad_policy_sum_frac: {np.mean(np.abs(policy_sums - 1.0) > 1e-2):.3f}")
        print(f"nonzero_actions mean/std: {policy_nonzero.mean():.2f} / {policy_nonzero.std():.2f}")
    else:
        print("No unpacked policies found in this chunk.")

    print(f"duplicate_fraction: {rep['duplicate_fraction']:.4f}")
    print(f"zero_states: {rep['zero_states']}")
    print(f"nan_states: {rep['nan_states']}")
    print(f"nan_policies: {rep['nan_policies']}")

    # flags
    flags = []
    if values.size and np.mean(np.abs(values) <= 0.1) > 0.70:
        flags.append("HIGH_DRAW_ZONE")
    if values.size and np.mean(values > 0.5) < 0.02 and np.mean(values < -0.5) < 0.02:
        flags.append("NO_DECISIVE_GAMES")
    if entropies.size and np.mean(entropies > 0.85) > 0.35:
        flags.append("POLICY_TOO_FLAT")
    if entropies.size and np.mean(entropies < 0.25) > 0.60:
        flags.append("POLICY_TOO_SHARP")
    if rep["duplicate_fraction"] > 0.05:
        flags.append("TOO_MANY_DUPLICATES")
    if rep["zero_states"] > 0:
        flags.append("ZERO_STATES")
    if rep["nan_states"] > 0 or rep["nan_policies"] > 0:
        flags.append("NAN_ISSUE")

    if flags:
        print("FLAGS:", ", ".join(flags))
    else:
        print("FLAGS: OK")


def main():
    cfg = load_config(None)
    buf = ReplayBuffer.load_from_path("./models/replay_buffer", cfg=cfg)

    n = len(buf)
    print(f"TOTAL SAMPLES: {n}")
    print(f"CHUNK SIZE: {CHUNK_SIZE}")

    all_draws = []
    all_dup = []
    all_policy_entropy = []
    summaries = []

    chunk_id = 0
    for start in range(0, n, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n)
        rep = analyze_chunk(buf, start, end)
        print_chunk_report(chunk_id, start, end, rep)

        values = np.array(rep["values"], dtype=np.float64) if rep["values"] else np.array([])
        entropies = np.array(rep["entropies"], dtype=np.float64) if rep["entropies"] else np.array([])

        draw_frac = float(np.mean(np.abs(values) <= 0.1)) if values.size else float("nan")
        dup_frac = float(rep["duplicate_fraction"])
        ent_mean = float(entropies.mean()) if entropies.size else float("nan")

        all_draws.append(draw_frac)
        all_dup.append(dup_frac)
        all_policy_entropy.append(ent_mean)

        summaries.append((chunk_id, start, end, draw_frac, dup_frac, ent_mean))
        chunk_id += 1

    print("\n========== GLOBAL SUMMARY ==========")

    valid_draws = np.array([x for x in all_draws if np.isfinite(x)], dtype=np.float64)
    if valid_draws.size:
        print(f"avg draw_frac: {valid_draws.mean():.3f}")
        print(f"min draw_frac: {valid_draws.min():.3f}")
        print(f"max draw_frac: {valid_draws.max():.3f}")

    valid_dup = np.array([x for x in all_dup if np.isfinite(x)], dtype=np.float64)
    if valid_dup.size:
        print(f"avg duplicate_frac: {valid_dup.mean():.4f}")
        print(f"max duplicate_frac: {valid_dup.max():.4f}")

    valid_ent = np.array([x for x in all_policy_entropy if np.isfinite(x)], dtype=np.float64)
    if valid_ent.size:
        print(f"avg policy entropy: {valid_ent.mean():.4f}")
        print(f"min policy entropy: {valid_ent.min():.4f}")
        print(f"max policy entropy: {valid_ent.max():.4f}")

    print("\n========== WEAKEST CHUNKS ==========")
    # worst by draw fraction
    worst_draw = [x for x in summaries if np.isfinite(x[3])]
    worst_draw.sort(key=lambda t: t[3], reverse=True)
    for row in worst_draw[:5]:
        cid, start, end, draw_frac, dup_frac, ent_mean = row
        print(
            f"chunk {cid} [{start}->{end}] "
            f"draw={draw_frac:.3f} dup={dup_frac:.4f} ent={ent_mean if np.isfinite(ent_mean) else 'N/A'}"
        )


if __name__ == "__main__":
    main()