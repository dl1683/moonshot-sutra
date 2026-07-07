"""TAD-1: Teacher-As-Data Curriculum Generation.

Generates HellaSwag-style commonsense continuation records using teacher LLMs.
Each record is serialized into natural text for byte-CE training.

Codex R61 specification:
- Each record: context, correct continuation, 3 hard negatives, rationale,
  why each negative fails, 1 answer-preserving paraphrase, 1 counterfactual
- Two-teacher agreement filter
- Reject eval/test n-gram overlap

Serialization formats (mixed during training):
  Format A (continuation): "Context: {ctx}\nNext: {correct}\n"
  Format B (explanation): "Context: {ctx}\nNext: {correct}\nReason: {rationale}\n"
  Format C (discrimination): "Context: {ctx}\nCorrect: {correct}\nWrong: {neg}\nWhy wrong: {reason}\n"
  Format D (paraphrase): "Context: {ctx}\nSame meaning: {paraphrase}\nNext: {correct}\n"
  Format E (counterfactual): "Instead: {cf_ctx}\nNext: {cf_correct}\n"
"""

from __future__ import annotations

import argparse
import json
import os
import random
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class TADRecord:
    context: str
    correct_continuation: str
    hard_negatives: list[str] = field(default_factory=list)
    rationale: str = ""
    negative_reasons: list[str] = field(default_factory=list)
    paraphrase_context: str = ""
    counterfactual_context: str = ""
    counterfactual_continuation: str = ""
    source: str = ""
    domain: str = ""


def serialize_record(record: TADRecord, format_type: str = "random") -> str:
    """Serialize a TAD record into natural text for byte-CE training."""
    if format_type == "random":
        format_type = random.choice(["A", "B", "C", "D", "E"])

    if format_type == "A":
        return f"Context: {record.context}\nNext: {record.correct_continuation}\n"

    elif format_type == "B" and record.rationale:
        return (
            f"Context: {record.context}\n"
            f"Next: {record.correct_continuation}\n"
            f"Reason: {record.rationale}\n"
        )

    elif format_type == "C" and record.hard_negatives and record.negative_reasons:
        idx = random.randrange(len(record.hard_negatives))
        neg = record.hard_negatives[idx]
        reason = record.negative_reasons[idx] if idx < len(record.negative_reasons) else ""
        return (
            f"Context: {record.context}\n"
            f"Correct: {record.correct_continuation}\n"
            f"Wrong: {neg}\n"
            f"Why wrong: {reason}\n"
        )

    elif format_type == "D" and record.paraphrase_context:
        return (
            f"Context: {record.paraphrase_context}\n"
            f"Next: {record.correct_continuation}\n"
        )

    elif format_type == "E" and record.counterfactual_context:
        return (
            f"Instead: {record.counterfactual_context}\n"
            f"Next: {record.counterfactual_continuation}\n"
        )

    # Fallback to format A
    return f"Context: {record.context}\nNext: {record.correct_continuation}\n"


def serialize_record_all_formats(record: TADRecord) -> list[str]:
    """Generate all valid serializations for a record."""
    outputs = []
    outputs.append(serialize_record(record, "A"))
    if record.rationale:
        outputs.append(serialize_record(record, "B"))
    if record.hard_negatives and record.negative_reasons:
        for i in range(min(len(record.hard_negatives), len(record.negative_reasons))):
            neg = record.hard_negatives[i]
            reason = record.negative_reasons[i]
            outputs.append(
                f"Context: {record.context}\n"
                f"Correct: {record.correct_continuation}\n"
                f"Wrong: {neg}\n"
                f"Why wrong: {reason}\n"
            )
    if record.paraphrase_context:
        outputs.append(serialize_record(record, "D"))
    if record.counterfactual_context:
        outputs.append(serialize_record(record, "E"))
    return outputs


def load_hellaswag_train_contexts(max_examples: int = 5000) -> list[dict]:
    """Load HellaSwag train split contexts for curriculum generation."""
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="train")
    contexts = []
    for i, row in enumerate(ds):
        if i >= max_examples:
            break
        contexts.append({
            "activity": row["activity_label"],
            "context": row["ctx"],
            "correct": row["endings"][int(row["label"])],
            "negatives": [e for j, e in enumerate(row["endings"]) if j != int(row["label"])],
        })
    return contexts


def generate_synthetic_commonsense(n: int = 1000, seed: int = 42) -> list[TADRecord]:
    """Generate simple synthetic commonsense records without API.

    Uses template-based generation for testing the pipeline.
    Real TAD-1 uses API-generated records.
    """
    rng = random.Random(seed)

    activities = [
        ("cooking", "A person is standing in a kitchen.", "They turn on the stove and place a pan on it.",
         ["They jump out the window.", "They start reading a book about astronomy.",
          "They plant seeds in the floor."],
         "Kitchens are for cooking, and stoves are cooking appliances."),
        ("driving", "A car approaches a red traffic light.", "The driver slows down and stops.",
         ["The driver speeds up and runs the light.", "The car starts flying.",
          "The driver gets out and starts dancing."],
         "Red lights mean stop; drivers follow traffic rules."),
        ("sports", "A soccer player receives the ball near the goal.", "They kick the ball toward the net.",
         ["They pick up the ball with their hands.", "They sit down on the field.",
          "They throw the ball into the stands."],
         "Soccer players use their feet and try to score goals."),
    ]

    records = []
    for _ in range(n):
        domain, ctx, correct, negs, rationale = rng.choice(activities)
        record = TADRecord(
            context=ctx,
            correct_continuation=correct,
            hard_negatives=negs,
            rationale=rationale,
            negative_reasons=[f"This doesn't happen in {domain} scenarios." for _ in negs],
            source="synthetic",
            domain=domain,
        )
        records.append(record)
    return records


def records_to_byte_shard(
    records: list[TADRecord],
    output_path: str,
    formats_per_record: int = 3,
    seed: int = 42,
) -> int:
    """Convert TAD records to a byte shard file for training.

    Returns total bytes written.
    """
    rng = random.Random(seed)
    buffer = bytearray()
    doc_sep = b"\xff"

    shuffled = list(records)
    rng.shuffle(shuffled)

    for record in shuffled:
        serializations = serialize_record_all_formats(record)
        chosen = rng.sample(serializations, min(formats_per_record, len(serializations)))
        for text in chosen:
            raw = text.encode("utf-8", errors="replace")
            buffer.extend(raw)
            buffer.extend(doc_sep)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(bytes(buffer))

    return len(buffer)


def shuffle_records(records: list[TADRecord], seed: int = 99) -> list[TADRecord]:
    """Create shuffled control: randomize context-continuation pairings."""
    rng = random.Random(seed)
    contexts = [r.context for r in records]
    continuations = [r.correct_continuation for r in records]
    rng.shuffle(continuations)

    shuffled = []
    for ctx, cont in zip(contexts, continuations):
        shuffled.append(TADRecord(
            context=ctx,
            correct_continuation=cont,
            source="shuffled_control",
        ))
    return shuffled


def main():
    parser = argparse.ArgumentParser(description="TAD-1 curriculum generation")
    parser.add_argument("--mode", choices=["synthetic", "hellaswag", "api"],
                        default="synthetic")
    parser.add_argument("--output-dir", default="C:/sutra_fast/data/tad1_curriculum")
    parser.add_argument("--n-records", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "synthetic":
        print(f"Generating {args.n_records} synthetic commonsense records...")
        records = generate_synthetic_commonsense(args.n_records, args.seed)

    elif args.mode == "hellaswag":
        print(f"Loading HellaSwag train contexts (up to {args.n_records})...")
        contexts = load_hellaswag_train_contexts(args.n_records)
        records = []
        for ctx in contexts:
            records.append(TADRecord(
                context=ctx["context"],
                correct_continuation=ctx["correct"],
                hard_negatives=ctx["negatives"],
                source="hellaswag_train",
                domain=ctx["activity"],
            ))

    # Write main curriculum shard
    main_path = os.path.join(args.output_dir, "tad1_main.bin")
    n_bytes = records_to_byte_shard(records, main_path, seed=args.seed)
    print(f"Main curriculum: {len(records)} records, {n_bytes/1e6:.1f}MB -> {main_path}")

    # Write shuffled control shard
    shuffled = shuffle_records(records, seed=args.seed + 1)
    ctrl_path = os.path.join(args.output_dir, "tad1_shuffled.bin")
    n_bytes_ctrl = records_to_byte_shard(shuffled, ctrl_path, formats_per_record=1, seed=args.seed)
    print(f"Shuffled control: {len(shuffled)} records, {n_bytes_ctrl/1e6:.1f}MB -> {ctrl_path}")

    # Write metadata
    meta = {
        "n_records": len(records),
        "mode": args.mode,
        "seed": args.seed,
        "main_bytes": n_bytes,
        "shuffled_bytes": n_bytes_ctrl,
        "formats": ["A", "B", "C", "D", "E"],
    }
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
