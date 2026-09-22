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
DRY_RUN = False   

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
        print("No shards found in manifest.")
        return

    # 1) load all shards
    shard_records: dict[str, list[dict]] = {}
    shard_templates: dict[str, dict] = {}
    all_entries = []

    global_index = 0
    target_state_key = None

    for shard_meta in shards:
        fname = str(shard_meta["file"])
        path = shard_dir / fname
        payload = torch.load(path, map_location="cpu", weights_only=False)
        recs = unpack_shard_payload(payload)
        shard_records[fname] = recs
        shard_templates[fname] = payload

        for local_idx, rec in enumerate(recs):
            state_key = state_hash_from_array(rec["state"])

            if global_index == TARGET_INDEX:
                target_state_key = state_key

            all_entries.append({
                "global_index": global_index,
                "file": fname,
                "local_idx": local_idx,
                "uid": rec["uid"],
                "state_key": state_key,
                "score": sample_score(rec["value"], rec["priority"], rec["policy_probs"]),
                "value": rec["value"],
                "priority": rec["priority"],
            })
            global_index += 1

    print(f"Loaded {len(all_entries)} samples from {len(shards)} shards.")

    if target_state_key is None:
        raise ValueError(f"TARGET_INDEX={TARGET_INDEX} not found.")

    # 2) isolate only target group
    target_entries = [e for e in all_entries if e["state_key"] == target_state_key]
    target_entries.sort(key=lambda x: (x["score"], abs(x["value"]), x["priority"], x["uid"]), reverse=True)

    print(f"TARGET_INDEX: {TARGET_INDEX}")
    print(f"TARGET_GROUP_SIZE: {len(target_entries)}")
    print(f"KEEP_BEST_PER_STATE: {KEEP_BEST_PER_STATE}")

    if len(target_entries) <= KEEP_BEST_PER_STATE:
        print("Target group is already within cap. Nothing to remove.")
        return

    keep_target = target_entries[:KEEP_BEST_PER_STATE]
    remove_target = target_entries[KEEP_BEST_PER_STATE:]

    keep_pairs = {(e["file"], e["local_idx"]) for e in keep_target}
    remove_pairs = {(e["file"], e["local_idx"]) for e in remove_target}

    print(f"WILL_REMOVE_FROM_TARGET_GROUP: {len(remove_target)}")
    print("\nTop kept global indices:")
    print([e["global_index"] for e in keep_target])

    print("\nFirst 50 removed global indices:")
    print([e["global_index"] for e in remove_target[:50]])

    # 3) rebuild only affected shards
    new_manifest_shards = []
    total_size = 0
    all_kept_uids = []

    affected_files = sorted({e["file"] for e in target_entries})
    print(f"\nAffected shards: {len(affected_files)}")
    for fname in affected_files:
        print(" -", fname)

    for shard_meta in shards:
        fname = str(shard_meta["file"])
        old_records = shard_records[fname]

        if fname not in affected_files:
            kept_records = old_records
        else:
            kept_records = [
                rec for idx, rec in enumerate(old_records)
                if (fname, idx) not in remove_pairs
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

        if fname in affected_files:
            print(f"{fname}: {len(old_records)} -> {len(kept_records)}")

        if not DRY_RUN and fname in affected_files:
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

    print("\nDONE" + (" (dry-run)" if DRY_RUN else ""))


if __name__ == "__main__":
    main()