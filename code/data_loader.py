"""Universal data loader for Sutra training.

Auto-discovers ALL token shards in data/shards/ and loads them.
No hardcoded paths. No version-specific loading. Drop a shard file,
it gets picked up on next trainer restart.

Shard directory structure:
    data/shards/
        minipile.pt              (single file = one source)
        wikipedia_0000.pt        (numbered = multi-shard source)
        wikipedia_0001.pt
        openwebmath_0000.pt
        fineweb_0000.pt
        tinystories_0000.pt
        ...

Any .pt file in data/shards/ is loaded. That's it.
File naming convention: {source}_{shard_number}.pt or {source}.pt

Usage:
    from data_loader import load_all_data
    train_tokens, test_tokens = load_all_data()
"""

import torch
from pathlib import Path

REPO = Path(__file__).parent.parent
SHARD_DIR = REPO / "data" / "shards"


def load_all_data(test_fraction=0.005):
    """Load all token shards from data/shards/.

    Returns (train_tokens, test_tokens) as flat tensors.
    Automatically discovers all .pt files. No config needed.
    """
    if not SHARD_DIR.exists():
        SHARD_DIR.mkdir(parents=True)
        print(f"  Created {SHARD_DIR}. Add .pt shard files to start training.")

    shard_files = sorted(SHARD_DIR.glob("*.pt"))
    if not shard_files:
        # Fallback to old location
        old_path = REPO / "data" / "minipile_full_tokens.pt"
        if old_path.exists():
            print(f"  No shards in {SHARD_DIR}, falling back to {old_path.name}")
            tokens = torch.load(old_path, weights_only=True)
            n_test = max(1000, int(len(tokens) * test_fraction))
            return tokens[:-n_test], tokens[-n_test:]
        raise FileNotFoundError(f"No data found in {SHARD_DIR} or {old_path}")

    # Group by source name (strip _NNNN suffix)
    sources = {}
    for f in shard_files:
        name = f.stem
        # Strip shard number: "wikipedia_0042" -> "wikipedia"
        parts = name.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            source = parts[0]
        else:
            source = name
        if source not in sources:
            sources[source] = []
        sources[source].append(f)

    # Load and report
    all_parts = []
    total = 0
    for source in sorted(sources.keys()):
        files = sorted(sources[source])
        parts = [torch.load(f, weights_only=True) for f in files]
        combined = torch.cat(parts) if len(parts) > 1 else parts[0]
        all_parts.append(combined)
        total += len(combined)
        print(f"  {source}: {len(combined):,} tokens ({len(combined)/1e9:.2f}B) [{len(files)} shards]")

    all_tokens = torch.cat(all_parts)
    print(f"  TOTAL: {total:,} tokens ({total/1e9:.2f}B) from {len(sources)} sources")

    n_test = max(1000, int(len(all_tokens) * test_fraction))
    return all_tokens[:-n_test], all_tokens[-n_test:]


def migrate_shards():
    """Move existing shards from old locations into data/shards/.

    Run once to consolidate:
      data/minipile_full_tokens.pt -> data/shards/minipile.pt (symlink)
      data/diverse_shards/wikipedia/shard_NNNN.pt -> data/shards/wikipedia_NNNN.pt
      data/fineweb_shards/shard_NNNN.pt -> data/shards/fineweb_NNNN.pt
    """
    import shutil
    SHARD_DIR.mkdir(parents=True, exist_ok=True)

    # MiniPile
    mp = REPO / "data" / "minipile_full_tokens.pt"
    dst = SHARD_DIR / "minipile.pt"
    if mp.exists() and not dst.exists():
        print(f"  Linking {mp.name} -> shards/minipile.pt")
        # Symlink to avoid copying 13GB
        try:
            dst.symlink_to(mp.resolve())
        except OSError:
            # Windows may not support symlinks, copy instead
            shutil.copy2(mp, dst)

    # Diverse shards
    diverse = REPO / "data" / "diverse_shards"
    if diverse.exists():
        for source_dir in sorted(diverse.iterdir()):
            if not source_dir.is_dir():
                continue
            name = source_dir.name
            for shard in sorted(source_dir.glob("shard_*.pt")):
                num = shard.stem.split("_")[1]
                dst = SHARD_DIR / f"{name}_{num}.pt"
                if not dst.exists():
                    print(f"  Linking {name}/{shard.name} -> shards/{dst.name}")
                    try:
                        dst.symlink_to(shard.resolve())
                    except OSError:
                        shutil.copy2(shard, dst)

    # FineWeb shards
    fineweb = REPO / "data" / "fineweb_shards"
    if fineweb.exists():
        for shard in sorted(fineweb.glob("shard_*.pt")):
            num = shard.stem.split("_")[1]
            dst = SHARD_DIR / f"fineweb_{num}.pt"
            if not dst.exists():
                print(f"  Linking fineweb/{shard.name} -> shards/{dst.name}")
                try:
                    dst.symlink_to(shard.resolve())
                except OSError:
                    shutil.copy2(shard, dst)

    # Report
    total = len(list(SHARD_DIR.glob("*.pt")))
    print(f"  Migration complete: {total} shards in {SHARD_DIR}")


if __name__ == "__main__":
    import sys
    if "--migrate" in sys.argv:
        print("=== MIGRATING SHARDS ===")
        migrate_shards()
    else:
        print("=== LOADING ALL DATA ===")
        train, test = load_all_data()
        print(f"\n  Train: {len(train):,} tokens")
        print(f"  Test:  {len(test):,} tokens")
