from __future__ import annotations

import math
import hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

# ====== CONFIG ======
MANIFEST_PATH = Path("./models/replay_buffer")   
TARGET_INDEX = 111
KEEP_BEST_PER_STATE = 8
DRY_RUN = True
DRY_RUN = True

MANIFEST_FORMAT = "ring_replay_buffer_manifest_v1"
COMPACT_FORMAT = "ring_replay_buffer_compact_v2"
POLICY_EPS = 1e-12


def state_hash_from_array(state: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(state, dtype=np.float16)
    h = hashlib.blake2b(digest_size=16)
    h.update(arr.tobytes())
    return h.digest()


def normalized_entropy_from_sparse_probs(probs: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)
    probs = probs[np.isfinite(probs)]
    probs = probs[probs > POLICY_EPS]
    if probs.size == 0:
        return 0.0
    s = float(probs.sum())
    if s <= POLICY_EPS:
        return 0.0
    probs = probs / s
    h = -float(np.sum(probs * np.log(probs + 1e-12)))
    max_h = math.log(int(probs.size)) if probs.size > 1 else 1.0
    if max_h <= 0.0:
        return 0.0
    return float(h / max_h)


def sample_score(value: float, priority: float, probs: np.ndarray) -> float:
    abs_value = abs(float(value))
    entropy = normalized_entropy_from_sparse_probs(probs)
    entropy_bonus = 1.0 - entropy
    priority_norm = min(max(float(priority) / 10.0, 0.0), 1.0)
    return 3.0 * abs_value + 1.0 * entropy_bonus + 0.25 * priority_norm


def load_manifest(path: Path) -> dict:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or obj.get("format") != MANIFEST_FORMAT:
        raise ValueError(f"{path} is not a replay manifest")
    return obj


def shard_dir_from_manifest(manifest_path: Path, manifest: dict) -> Path:
    shard_dir_name = manifest.get("shard_dir") or manifest_path.with_suffix(manifest_path.suffix + ".d").name
    return manifest_path.parent / shard_dir_name


def unpack_shard_payload(payload: dict) -> list[dict]:
    if payload.get("format") != COMPACT_FORMAT:
        raise ValueError("Unexpected shard format")

    states = np.asarray(payload["states"], dtype=np.float16)
    values = np.asarray(payload["values"], dtype=np.float32)
    priorities = np.asarray(payload["priorities"], dtype=np.float32)
    uids = np.asarray(payload["uids"], dtype=np.int64)
    lengths = np.asarray(payload["policy_lengths"], dtype=np.int32)
    flat_indices = np.asarray(payload["policy_indices"], dtype=np.uint16)
    flat_probs = np.asarray(payload["policy_probs"], dtype=np.float16)

    size = int(payload.get("size", len(states)))
    size = min(size, len(states), len(values), len(priorities), len(uids), len(lengths))

    out = []
    off = 0
    for i in range(size):
        ln = int(lengths[i])
        nxt = off + max(0, ln)
        rec = {
            "state": np.ascontiguousarray(states[i], dtype=np.float16),
            "value": float(values[i]),
            "priority": float(priorities[i]),
            "uid": int(uids[i]),
            "policy_indices": np.ascontiguousarray(flat_indices[off:nxt], dtype=np.uint16),
            "policy_probs": np.ascontiguousarray(flat_probs[off:nxt], dtype=np.float16),
        }
        out.append(rec)
        off = nxt
    return out


def repack_records(records: list[dict], template: dict) -> dict:
    if records:
        states = np.ascontiguousarray(np.stack([r["state"] for r in records], axis=0), dtype=np.float16)
        values = np.asarray([r["value"] for r in records], dtype=np.float32)
        priorities = np.asarray([r["priority"] for r in records], dtype=np.float32)
        uids = np.asarray([r["uid"] for r in records], dtype=np.int64)
        lengths = np.asarray([len(r["policy_indices"]) for r in records], dtype=np.int32)

        total_nnz = int(lengths.sum())
        if total_nnz > 0:
            policy_indices = np.concatenate([r["policy_indices"] for r in records], axis=0).astype(np.uint16, copy=False)
            policy_probs = np.concatenate([r["policy_probs"] for r in records], axis=0).astype(np.float16, copy=False)
        else:
            policy_indices = np.zeros((0,), dtype=np.uint16)
            policy_probs = np.zeros((0,), dtype=np.float16)
    else:
        state_shape = tuple(template.get("state_shape", (20, 8, 8)))
        states = np.zeros((0, *state_shape), dtype=np.float16)
        values = np.zeros((0,), dtype=np.float32)
        priorities = np.zeros((0,), dtype=np.float32)
        uids = np.zeros((0,), dtype=np.int64)
        lengths = np.zeros((0,), dtype=np.int32)
        policy_indices = np.zeros((0,), dtype=np.uint16)
        policy_probs = np.zeros((0,), dtype=np.float16)

    new_payload = dict(template)
    new_payload["states"] = states
    new_payload["values"] = values
    new_payload["priorities"] = priorities
    new_payload["uids"] = uids
    new_payload["policy_lengths"] = lengths
    new_payload["policy_indices"] = policy_indices
    new_payload["policy_probs"] = policy_probs
    new_payload["size"] = int(len(records))
    new_payload["capacity"] = int(max(1, len(records)))
    new_payload["pos"] = 0
    return new_payload


def atomic_torch_save(obj: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def main():
    manifest = load_manifest(MANIFEST_PATH)
    shard_dir = shard_dir_from_manifest(MANIFEST_PATH, manifest)

    shards = manifest.get("shards", [])
    if not shards:
        print("No shards in manifest.")
        return

    # 1) load all shard records
    shard_records: dict[str, list[dict]] = {}
    shard_templates: dict[str, dict] = {}
    all_entries = []

    for shard_meta in shards:
        fname = str(shard_meta["file"])
        path = shard_dir / fname
        payload = torch.load(path, map_location="cpu", weights_only=False)
        recs = unpack_shard_payload(payload)
        shard_records[fname] = recs
        shard_templates[fname] = payload

        for local_idx, rec in enumerate(recs):
            key = state_hash_from_array(rec["state"])
            score = sample_score(rec["value"], rec["priority"], rec["policy_probs"])
            all_entries.append({
                "file": fname,
                "local_idx": local_idx,
                "uid": rec["uid"],
                "state_key": key,
                "score": score,
                "value": rec["value"],
                "priority": rec["priority"],
            })

    print(f"Loaded {len(all_entries)} samples from {len(shards)} shards.")

    # 2) group by state and decide winners
    by_state = defaultdict(list)
    for e in all_entries:
        by_state[e["state_key"]].append(e)

    keep_pairs = set()
    num_dup_groups = 0
    total_removed = 0

    for state_key, entries in by_state.items():
        if len(entries) <= KEEP_BEST_PER_STATE:
            for e in entries:
                keep_pairs.add((e["file"], e["local_idx"]))
            continue

        num_dup_groups += 1
        entries.sort(key=lambda x: (x["score"], abs(x["value"]), x["priority"], x["uid"]), reverse=True)
        winners = entries[:KEEP_BEST_PER_STATE]
        losers = entries[KEEP_BEST_PER_STATE:]
        for e in winners:
            keep_pairs.add((e["file"], e["local_idx"]))
        total_removed += len(losers)

    print(f"Duplicate state groups over cap: {num_dup_groups}")
    print(f"Will remove: {total_removed}")
    print(f"Keep best per state: {KEEP_BEST_PER_STATE}")

    # 3) rebuild each shard, preserving filenames
    new_manifest_shards = []
    total_size = 0
    all_kept_uids = []

    for shard_meta in shards:
        fname = str(shard_meta["file"])
        old_records = shard_records[fname]
        kept_records = [
            rec for idx, rec in enumerate(old_records)
            if (fname, idx) in keep_pairs
        ]

        total_size += len(kept_records)
        if kept_records:
            kept_uids = [r["uid"] for r in kept_records]
            min_uid = int(min(kept_uids))
            max_uid = int(max(kept_uids))
            all_kept_uids.extend(kept_uids)

            new_manifest_shards.append({
                "file": fname,
                "count": int(len(kept_records)),
                "min_uid": min_uid,
                "max_uid": max_uid,
            })

        print(f"{fname}: {len(old_records)} -> {len(kept_records)}")

        if not DRY_RUN:
            new_payload = repack_records(kept_records, shard_templates[fname])
            atomic_torch_save(new_payload, shard_dir / fname)

    # 4) rewrite manifest
    if not DRY_RUN:
        new_manifest = dict(manifest)
        new_manifest["shards"] = new_manifest_shards
        new_manifest["size"] = int(total_size)

        if all_kept_uids:
            new_manifest["min_active_uid"] = int(min(all_kept_uids))
            new_manifest["max_active_uid"] = int(max(all_kept_uids))
            new_manifest["last_saved_uid"] = int(max(all_kept_uids))
        else:
            new_manifest["min_active_uid"] = 0
            new_manifest["max_active_uid"] = 0
            new_manifest["last_saved_uid"] = 0

        atomic_torch_save(new_manifest, MANIFEST_PATH)

    print("DONE" + (" (dry-run)" if DRY_RUN else ""))


if __name__ == "__main__":
    main()