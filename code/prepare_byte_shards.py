"""Prepare byte shards from D0-admitted Common Pile subsets.

Downloads admitted data sources, converts text to raw UTF-8 bytes,
and writes fixed-size binary shard files for ByteShardDataset.

Admitted sources (per SE1_CANONICAL_SPEC.md Section 14):
- common-pile/arxiv_abstracts (CC0)
- common-pile/caselaw_access_project (Public Domain)
- common-pile/biodiversity_heritage_library (Public Domain)
"""

from __future__ import annotations

import argparse
import os

ADMITTED_SOURCES = [
    "common-pile/arxiv_abstracts",
    "common-pile/caselaw_access_project",
    "common-pile/biodiversity_heritage_library",
]

SHARD_SIZE = 256 * 1024 * 1024  # 256 MiB per shard
DOC_SEPARATOR = b"\xff"  # byte 255 as document boundary (outside normal UTF-8)


def stream_texts(dataset_name: str, split: str = "train", max_docs: int | None = None):
    """Yield text strings from a HuggingFace dataset, streaming."""
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split=split, streaming=True)

    text_col = None
    for i, row in enumerate(ds):
        if text_col is None:
            for candidate in ["text", "content", "abstract", "body", "document"]:
                if candidate in row:
                    text_col = candidate
                    break
            if text_col is None:
                text_col = list(row.keys())[0]
            print(f"  Using text column: '{text_col}'")

        text = row.get(text_col, "")
        if text and len(text.strip()) > 0:
            yield text

        if max_docs and i + 1 >= max_docs:
            break


def write_shards(
    output_dir: str,
    sources: list[str],
    shard_size: int = SHARD_SIZE,
    max_docs_per_source: int | None = None,
):
    os.makedirs(output_dir, exist_ok=True)

    shard_idx = 0
    buffer = bytearray()
    total_bytes = 0
    total_docs = 0

    for source in sources:
        print(f"\nProcessing: {source}")
        doc_count = 0

        for text in stream_texts(source, max_docs=max_docs_per_source):
            raw = text.encode("utf-8", errors="replace")
            buffer.extend(raw)
            buffer.extend(DOC_SEPARATOR)
            doc_count += 1
            total_docs += 1

            while len(buffer) >= shard_size:
                shard_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.bin")
                with open(shard_path, "wb") as f:
                    f.write(bytes(buffer[:shard_size]))
                print(f"  Wrote {shard_path} ({shard_size / 1024 / 1024:.0f} MiB)")
                total_bytes += shard_size
                buffer = buffer[shard_size:]
                shard_idx += 1

        print(f"  {source}: {doc_count:,} documents")

    # Write final partial shard if any data remains
    if buffer:
        shard_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.bin")
        with open(shard_path, "wb") as f:
            f.write(bytes(buffer))
        total_bytes += len(buffer)
        shard_idx += 1
        print(f"  Wrote final shard {shard_path} ({len(buffer) / 1024 / 1024:.1f} MiB)")

    print(f"\nDone: {shard_idx} shards, {total_bytes / 1024 / 1024 / 1024:.2f} GiB, {total_docs:,} documents")

    # Write manifest
    manifest_path = os.path.join(output_dir, "manifest.txt")
    with open(manifest_path, "w") as f:
        f.write(f"sources: {', '.join(sources)}\n")
        f.write(f"n_shards: {shard_idx}\n")
        f.write(f"total_bytes: {total_bytes}\n")
        f.write(f"total_docs: {total_docs}\n")
        f.write(f"shard_size: {shard_size}\n")
        f.write("doc_separator: 0xff\n")
    print(f"Manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare byte shards for S0 training")
    parser.add_argument("--output-dir", default="data/shards_bytes", help="Output directory")
    parser.add_argument("--shard-size-mib", type=int, default=256, help="Shard size in MiB")
    parser.add_argument("--max-docs", type=int, default=None, help="Max docs per source (for testing)")
    parser.add_argument("--sources", nargs="+", default=ADMITTED_SOURCES, help="HF dataset names")
    args = parser.parse_args()

    write_shards(
        output_dir=args.output_dir,
        sources=args.sources,
        shard_size=args.shard_size_mib * 1024 * 1024,
        max_docs_per_source=args.max_docs,
    )


if __name__ == "__main__":
    main()
