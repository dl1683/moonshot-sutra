# Data Admission Policy

Status: D0 complete. Sources admitted and sharded.

## Admitted Sources (Training)

All admitted sources are Public Domain or CC0. No attribution-required or
restricted licenses in training data.

| Source | License | HuggingFace ID | Role |
|--------|---------|---------------|------|
| arXiv Abstracts | CC0 | `common-pile/arxiv_abstracts` | Scientific text |
| Caselaw Access Project | Public Domain | `common-pile/caselaw_access_project` | Legal text |
| Biodiversity Heritage Library | Public Domain | `common-pile/biodiversity_heritage_library` | Natural history |

Total: 565 shards x 256 MiB = 141 GiB raw bytes.

## Candidate (Pending Attribution)

| Source | License | HuggingFace ID | Role | Blocker |
|--------|---------|---------------|------|---------|
| TinyStories | CDLA-Sharing-1.0 | `roneneldan/TinyStories` | Generation guardrail | Needs frozen revision, attribution manifest, leakage testing |

CDLA-Sharing Section 3.5 explicitly exempts Results (model weights) from
share-alike obligations. Training is permitted; attribution still required.

## Held (Eval Only, Not Training)

| Source | License | HuggingFace ID | Use |
|--------|---------|---------------|-----|
| AI2 ARC | CC-BY-SA-4.0 | `allenai/ai2_arc` | Benchmark evaluation only |
| Databricks Dolly 15K | CC-BY-SA-3.0 | `databricks/databricks-dolly-15k` | Held pending review |
| arXiv Papers (full text) | Mixed | `common-pile/arxiv_papers` | Held (mixed licenses) |

## Admitted (Eval/Reasoning, Not Base Text)

| Source | License | HuggingFace ID | Use |
|--------|---------|---------------|-----|
| GSM8K | MIT | `openai/gsm8k` | Reasoning evaluation |
| JSONSchemaBench | MIT | `epfl-dlab/JSONSchemaBench` | Exact verifier eval |

## Rejected

| Source | License | Reason |
|--------|---------|--------|
| SciFact | CC-BY-NC-2.0 | Non-commercial clause incompatible |

## Shard Preparation

Shards are prepared by `code/prepare_byte_shards.py`:

```bash
python code/prepare_byte_shards.py --output-dir data/shards_bytes_full --shard-size-mib 256
```

The script downloads from HuggingFace, extracts text fields, encodes to UTF-8
bytes, and writes sequential binary shards. No tokenization.

## What Is NOT In Training

- No web crawl (Common Crawl, C4, OSCAR, etc.)
- No copyrighted books or paywalled content
- No CC-BY-SA or CC-BY-NC licensed material
- No synthetic data from proprietary models
- No private or personally identifiable information
- No code datasets (yet)

## License Posture

Sutra is trained exclusively on Public Domain / CC0 text for the scout build.
This is a deliberate choice: the byte-level architecture eliminates tokenizer
lock-in, and the open license posture eliminates legal lock-in. Both serve the
mission of democratized AI development.
