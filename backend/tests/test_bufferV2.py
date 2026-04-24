from __future__ import annotations

import math
from collections import defaultdict, Counter

import numpy as np

from app.training.replay_buffer import ReplayBuffer
from app.infra.config import load_config


CHUNK_SIZE = 5000

# thresholds
DRAW_THRESH = 0.10
FLAT_ENTROPY_THRESH = 0.85
LOW_PRIORITY_THRESH = 1.0
MIN_DUP_GROUP = 2


def normalized_entropy(probs: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    probs = probs[np.isfinite(probs)]
    probs = probs[probs > 0.0]
    if probs.size == 0:
        return 0.0
    s = probs.sum()
    if s <= 0:
        return 0.0
    probs = probs / s
    h = -np.sum(probs * np.log(probs + 1e-12))
    max_h = math.log(len(probs)) if len(probs) > 1 else 1.0
    return float(h / max_h) if max_h > 0 else 0.0


def state_hash_from_tensor(state) -> int:
    arr = state.detach().cpu().numpy() if hasattr(state, "detach") else np.asarray(state)
    arr = np.asarray(arr, dtype=np.float16, order="C")
    return hash(arr.tobytes())


def chunk_id_of(i: int, chunk_size: int = CHUNK_SIZE) -> int:
    return i // chunk_size


def main():
    cfg = load_config(None)
    buf = ReplayBuffer.load_from_path("./models/replay_buffer", cfg=cfg)

    n = len(buf)
    print(f"TOTAL SAMPLES: {n}")

    # 1) pass 1: exact duplicate groups
    state_to_indices: dict[int, list[int]] = defaultdict(list)

    for i in range(n):
        state, policy, value = buf.buffer[i]
        h = state_hash_from_tensor(state)
        state_to_indices[h].append(i)

    dup_groups = [idxs for idxs in state_to_indices.values() if len(idxs) >= MIN_DUP_GROUP]
    dup_groups.sort(key=len, reverse=True)

    print("\n========== DUPLICATE SUMMARY ==========")
    print("duplicate groups:", len(dup_groups))
    if dup_groups:
        total_dup_samples = sum(len(g) for g in dup_groups)
        print("samples inside duplicate groups:", total_dup_samples)
        print("approx duplicate fraction:", total_dup_samples / max(1, n))
    else:
        print("No duplicate groups found.")

    print("\nTop 10 duplicate groups:")
    for rank, group in enumerate(dup_groups[:10], start=1):
        print(f"{rank:2d}. size={len(group)} first_indices={group[:10]}")

    # 2) pass 2: per-sample quality scoring
    suspicious = []
    chunk_stats = defaultdict(lambda: {
        "count": 0,
        "draw_like": 0,
        "flat_policy": 0,
        "low_priority": 0,
        "duplicate_member": 0,
        "suspicious": 0,
    })

    dup_member = set()
    for group in dup_groups:
        for idx in group:
            dup_member.add(idx)

    for i in range(n):
        state, policy, value = buf.buffer[i]

        # dense policy جاهزة من buf.buffer[i]
        pol = np.asarray(policy, dtype=np.float64).reshape(-1)
        ent = normalized_entropy(pol)
        val = float(value)
        prio = float(buf.priorities[i]) if hasattr(buf, "priorities") else float("nan")
        cid = chunk_id_of(i)

        is_draw_like = abs(val) <= DRAW_THRESH
        is_flat = ent >= FLAT_ENTROPY_THRESH
        is_low_prio = np.isfinite(prio) and prio < LOW_PRIORITY_THRESH
        is_dup = i in dup_member

        # score بسيط
        score = 0
        if is_dup:
            score += 2
        if is_draw_like:
            score += 1
        if is_flat:
            score += 1
        if is_low_prio:
            score += 1

        # إذا duplicate + draw-like + flat => قوي جدًا للمسح
        hard_bad = is_dup and is_draw_like and is_flat
        medium_bad = score >= 3

        if hard_bad or medium_bad:
            suspicious.append({
                "idx": i,
                "chunk": cid,
                "value": val,
                "entropy": ent,
                "priority": prio,
                "duplicate": is_dup,
                "score": score,
            })

        st = chunk_stats[cid]
        st["count"] += 1
        st["draw_like"] += int(is_draw_like)
        st["flat_policy"] += int(is_flat)
        st["low_priority"] += int(is_low_prio)
        st["duplicate_member"] += int(is_dup)
        st["suspicious"] += int(hard_bad or medium_bad)

    # 3) print weak chunks
    print("\n========== CHUNK QUALITY ==========")
    rows = []
    max_chunk = max(chunk_stats) if chunk_stats else -1
    for cid in range(max_chunk + 1):
        st = chunk_stats[cid]
        c = max(1, st["count"])
        row = {
            "chunk": cid,
            "count": st["count"],
            "draw_frac": st["draw_like"] / c,
            "flat_frac": st["flat_policy"] / c,
            "low_prio_frac": st["low_priority"] / c,
            "dup_frac": st["duplicate_member"] / c,
            "suspicious_frac": st["suspicious"] / c,
        }
        rows.append(row)

    rows_sorted = sorted(rows, key=lambda r: (r["suspicious_frac"], r["draw_frac"], r["flat_frac"]), reverse=True)

    for r in rows:
        print(
            f"chunk {r['chunk']:2d} | count={r['count']:5d} | "
            f"draw={r['draw_frac']:.3f} | flat={r['flat_frac']:.3f} | "
            f"dup={r['dup_frac']:.3f} | low_prio={r['low_prio_frac']:.3f} | "
            f"suspicious={r['suspicious_frac']:.3f}"
        )

    print("\n========== WORST CHUNKS ==========")
    for r in rows_sorted[:8]:
        print(
            f"chunk {r['chunk']:2d} | suspicious={r['suspicious_frac']:.3f} | "
            f"draw={r['draw_frac']:.3f} | flat={r['flat_frac']:.3f} | dup={r['dup_frac']:.3f}"
        )

    # 4) print suspicious samples
    suspicious.sort(key=lambda x: (x["score"], x["duplicate"], abs(x["value"]) <= DRAW_THRESH, x["entropy"]), reverse=True)

    print("\n========== TOP SUSPICIOUS SAMPLES ==========")
    for row in suspicious[:50]:
        print(
            f"idx={row['idx']:6d} chunk={row['chunk']:2d} "
            f"value={row['value']:+.3f} entropy={row['entropy']:.3f} "
            f"priority={row['priority']:.3f} dup={row['duplicate']} score={row['score']}"
        )

    # 5) suggested deletion indices
    # conservative: delete duplicates except first one in each group
    conservative_delete = []
    for group in dup_groups:
        keep = group[0]
        for idx in group[1:]:
            conservative_delete.append(idx)

    # stronger: suspicious samples from very weak chunks
    weak_chunks = {r["chunk"] for r in rows if r["suspicious_frac"] >= 0.50 or (r["draw_frac"] >= 0.70 and r["flat_frac"] >= 0.60)}
    aggressive_delete = [x["idx"] for x in suspicious if x["chunk"] in weak_chunks]

    print("\n========== DELETE CANDIDATES ==========")
    print("conservative_delete_count (duplicates only):", len(conservative_delete))
    print("aggressive_delete_count (weak chunks + suspicious):", len(aggressive_delete))
    print("weak_chunks:", sorted(weak_chunks))
    print("first 100 conservative indices:", conservative_delete[:100])
    print("first 100 aggressive indices:", aggressive_delete[:100])


if __name__ == "__main__":
    main()