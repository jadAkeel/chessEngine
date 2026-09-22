from pathlib import Path
import shutil
import torch

from app.training.replay_buffer import ReplayBuffer
from app.infra.config import load_config


def main():
    cfg = load_config(None)

    old_path = "../models/old_replay_buffer.pt"
    new_path = "../models/replay_buffer"

    manifest = Path(new_path)
    shard_dir = manifest.with_name(manifest.name + ".d")

    if manifest.exists():
        if manifest.is_file():
            manifest.unlink()
        else:
            raise RuntimeError(f"{manifest} exists and is not a file")

    if shard_dir.exists():
        shutil.rmtree(shard_dir)

    print("=" * 80)
    print("STEP 1: LOAD RAW OLD BUFFER")
    print("old_path =", old_path)

    raw = torch.load(old_path, map_location="cpu", weights_only=False)

    if isinstance(raw, dict):
        print("raw keys =", list(raw.keys()))
        print("format =", raw.get("format"))
        print("size =", raw.get("size"))
        print("capacity =", raw.get("capacity"))

    print("=" * 80)
    print("STEP 2: REBUILD BUFFER FROM RAW SERIALIZED DATA")

    buf = ReplayBuffer.from_serialized(raw, cfg=cfg)

    print("len(buf) =", len(buf))
    print("capacity =", buf.capacity)
    print("policy_size =", buf.policy_size)

    # مهم جدًا: إجبار save() تعتبر كل البافر pending
    buf._saved_shards = []
    buf._last_saved_uid = 0

    print("=" * 80)
    print("STEP 3: SAVE MIGRATED BUFFER")
    print("new_path =", new_path)

    buf.save(new_path)

    print("saved migrated buffer")

    print("=" * 80)
    print("STEP 4: VERIFY")

    verify = ReplayBuffer.load_from_path(new_path, cfg=cfg)

    print("len(verify) =", len(verify))
    print("capacity =", verify.capacity)
    print("policy_size =", verify.policy_size)

    new_manifest = Path(new_path)
    new_shard_dir = new_manifest.with_name(new_manifest.name + ".d")

    print("manifest exists =", new_manifest.exists())
    print("shard dir exists =", new_shard_dir.exists())

    if new_shard_dir.exists():
        shards = sorted(new_shard_dir.glob("*"))
        print("shard count =", len(shards))
        for s in shards[:10]:
            print(" -", s.name, f"({s.stat().st_size / 1024 / 1024:.2f} MB)")
        if len(shards) > 10:
            print(f"... and {len(shards) - 10} more shards")

    print("=" * 80)
    print("DONE")


if __name__ == "__main__":
    main()




from app.training.replay_buffer import ReplayBuffer
from app.infra.config import load_config

#cfg = load_config(None)
#buf = ReplayBuffer.load_from_path("../models/replay_buffer", cfg=cfg)
#print(len(buf))