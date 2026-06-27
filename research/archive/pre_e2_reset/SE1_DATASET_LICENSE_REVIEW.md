# SE1 Dataset License Review

This is a readiness artifact, not legal advice. It records source metadata and
repo admission decisions for the first SE1 dataset candidates as checked on
2026-06-20.

No dataset is downloaded by this review.

## Review Scope

```text
SE1_dataset_license_review_2026_06_20_v0:
  status: metadata_review_not_legal_opinion
  sources_checked:
    - Hugging_Face_dataset_API
    - Hugging_Face_dataset_cards
    - Common_Pile_org_card
    - Common_Pile_arXiv_paper
  applies_to:
    - D0_base_text
    - D2_candidate_reasoning
    - D3_geometry_or_paraphrase
    - D4_exact_verifier
    - D5_generation_guardrail
  still_forbidden:
    - unreviewed_training_download
    - treating_sharealike_sources_as_default_training_data
    - treating_noncommercial_sources_as_default_training_data
    - claiming_Common_Pile_shard_license_safety_without_row_license_filters
```

## Source URLs

```text
SE1_dataset_license_review_sources_v0:
  common_pile:
    org: https://huggingface.co/common-pile
    paper: https://arxiv.org/abs/2506.05209
    arxiv_abstracts: https://huggingface.co/datasets/common-pile/arxiv_abstracts
    arxiv_papers: https://huggingface.co/datasets/common-pile/arxiv_papers
    caselaw_access_project: https://huggingface.co/datasets/common-pile/caselaw_access_project
    biodiversity_heritage_library: https://huggingface.co/datasets/common-pile/biodiversity_heritage_library

  held_or_guardrail_sources:
    fineweb_edu: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
    ai2_arc: https://huggingface.co/datasets/allenai/ai2_arc
    openbookqa: https://huggingface.co/datasets/allenai/openbookqa
    scifact: https://huggingface.co/datasets/allenai/scifact
    scitail: https://huggingface.co/datasets/allenai/scitail
    tinystories: https://huggingface.co/datasets/roneneldan/TinyStories
    dolly_15k: https://huggingface.co/datasets/databricks/databricks-dolly-15k

  default_candidate_sources:
    gsm8k: https://huggingface.co/datasets/openai/gsm8k
    humaneval: https://huggingface.co/datasets/openai/openai_humaneval
    apps: https://huggingface.co/datasets/codeparrot/apps
    jsonschemabench: https://huggingface.co/datasets/epfl-dlab/JSONSchemaBench
```

## Common Pile Decision

Common Pile is a useful default source family only if SE1 admits specific shards
with explicit row-level license constraints. The organization card and paper
describe the project as public-domain/openly licensed, but the individual shard
cards still matter because some rows carry mixed source licenses.

```text
SE1_common_pile_shard_review_2026_06_20:
  common-pile/arxiv_abstracts:
    revision: 828e35d1000f94579da8850f5f640c138279bdb5
    row_count_from_hf_viewer: about_2_54M
    compressed_data_files: 4
    compressed_bytes_from_hf_tree_api: 1127871878
    card_license_metadata: unspecified
    row_license_evidence:
      - metadata.license_contains_Creative_Commons_Zero_Public_Domain_in_viewer_rows
      - full_text_license_may_vary_and_is_not_admitted_for_abstract_text_use
    SE1_decision: candidate_admit_for_D0_after_row_filter_manifest
    required_filter:
      - keep_only_rows_where_metadata.license_is_CC0_or_Public_Domain_equivalent
      - use_abstract_text_only
      - preserve_url_and_license_metadata_in_manifest

  common-pile/caselaw_access_project:
    revision: 3c2cb5080b3a16a04d8d8d07b28eaec7c1ba7a90
    row_count_from_hf_viewer: about_5_52M
    compressed_data_files: 48
    compressed_bytes_from_hf_tree_api: 6602477877
    card_license_metadata: unspecified
    row_license_evidence:
      - metadata.license_Public_Domain_in_viewer_rows
    SE1_decision: candidate_admit_for_D0_after_row_filter_manifest
    required_filter:
      - keep_only_rows_where_metadata.license_is_Public_Domain
      - preserve_source_url_and_case_identifier
      - sample_cap_before_full_download

  common-pile/biodiversity_heritage_library:
    revision: ad1bbb00d5df579c0ef0a9dfe46e50e2bb1715b8
    row_count_from_hf_viewer: about_45_6M
    compressed_data_files: 47
    compressed_bytes_from_hf_tree_api: 15830744336
    card_license_metadata: unspecified
    row_license_evidence:
      - metadata.license_Public_Domain_in_viewer_rows
    SE1_decision: candidate_admit_for_D0_after_row_filter_manifest_and_sample_cap
    required_filter:
      - keep_only_rows_where_metadata.license_is_Public_Domain
      - sample_cap_first_due_to_size_and_domain_narrowness
      - preserve_source_url_and_page_identifier

  common-pile/arxiv_papers:
    revision: 963fe980c55b353980653f1a27c1dc0c8a2d7058
    row_count_from_hf_viewer: about_317k
    compressed_data_files: 22
    compressed_bytes_from_hf_tree_api: 6262555629
    card_license_metadata: unspecified
    row_license_evidence:
      - viewer_rows_show_mixed_license_values_including_CC_BY_and_Public_Domain
      - dataset_card_has_license_laundering_warning
    SE1_decision: held_until_explicit_row_license_filter_and_exclusion_report_exists
    required_filter:
      - keep_only_rows_with_allowed_license_family
      - reject_noncommercial_no_derivatives_or_ambiguous_rows
      - record_license_histogram_before_any_training_use
```

D0 consequence:

```text
SE1_D0_base_text_license_decision_v0:
  status: candidate_limited_common_pile_public_domain_scout
  admitted_without_download:
    - common-pile/arxiv_abstracts_after_row_license_filter
    - common-pile/caselaw_access_project_after_row_license_filter
    - common-pile/biodiversity_heritage_library_after_row_license_filter_and_sample_cap
  held:
    - common-pile/arxiv_papers
    - HuggingFaceFW/fineweb-edu_as_default_base_text
  planned_rows: not_written
  row_filter_manifest: artifacts/dataset_license/common_pile_row_filter_manifest.json
  next_required_artifact: SE1_D0_common_pile_license_histogram_and_sample_manifest
```

## Non-Common-Pile Decisions

```text
SE1_dataset_admission_decisions_2026_06_20:
  default_candidate_after_planned_subset:
    openai/gsm8k:
      revision: 740312add88f781978c0658806c59bc2815b9866
      license_metadata: mit
      decision: candidate_admit_after_planned_rows_written

    openai/openai_humaneval:
      revision: 7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544
      license_metadata: mit
      decision: candidate_admit_as_eval_or_oracle_source

    codeparrot/apps:
      revision: 21e74ddf8de1a21436da12e3e653065c5213e9d1
      license_metadata: mit
      decision: candidate_admit_after_config_and_size_are_resolved

    epfl-dlab/JSONSchemaBench:
      revision: 5bd0f4640badc6f3f02df796421d21cb0ca0b141
      license_metadata: mit
      decision: candidate_admit_after_planned_rows_written

  held_for_sharealike_or_attribution_review:
    allenai/ai2_arc:
      revision: 210d026faf9955653af8916fad021475a3f00453
      license_metadata: cc-by-sa-4.0
      decision: held_for_training_use_until_sharealike_implications_are_reviewed

    HuggingFaceFW/fineweb-edu:
      revision: 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9
      license_metadata: odc-by
      decision: quality_scale_ablation_only_not_default_base_text

    roneneldan/TinyStories:
      revision: f54c09fd23315a6f9c86f9dc80f725de7d8f9c64
      license_metadata: cdla-sharing-1.0
      decision: held_as_small_generation_guardrail_until_sharing_terms_reviewed

    databricks/databricks-dolly-15k:
      revision: bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a
      license_metadata: cc-by-sa-3.0
      decision: held_until_sharealike_implications_are_reviewed

  rejected_or_eval_only_by_default:
    allenai/scifact:
      revision: 1fe54665deee011033b2dd98db5752e0d586fdfb
      license_metadata: cc-by-nc-2.0
      decision: evidence_or_eval_only_no_default_training_use

    allenai/openbookqa:
      revision: 388097ea7776314e93a529163e0fea805b8a6454
      license_metadata: unknown
      decision: held_until_distribution_license_resolved

    allenai/scitail:
      revision: 0cc4353235b289165dfde1c7c5d1be983f99ce44
      license_metadata: unspecified
      decision: held_until_distribution_license_resolved
```

## Next Required Artifacts

```text
SE1_dataset_license_next_artifacts_v0:
  D0_common_pile_row_filter_manifest:
    path: artifacts/dataset_license/common_pile_row_filter_manifest.json
    status: draft_created_no_rows_counted_no_download_allowed
    required_fields:
      - dataset_id
      - revision
      - allowed_license_values
      - rejected_license_values
      - sample_cap
      - row_count_before_filter
      - row_count_after_filter
      - source_url_preserved

  D0_common_pile_license_histogram:
    path: artifacts/dataset_license/common_pile_license_histogram.json
    status: not_created
    required_before:
      - any_common_pile_training_download

  fixture_filter_reports:
    status: created_not_common_pile_evidence
    tool: tools/common_pile_license_filter.py
    histogram: artifacts/dataset_license/common_pile_fixture_license_histogram.json
    sample_manifest: artifacts/dataset_license/common_pile_fixture_sample_manifest.json

  sharealike_review_note:
    path: artifacts/dataset_license/sharealike_review.md
    status: created_repo_policy_hold_not_legal_clearance
    applies_to:
      - allenai/ai2_arc
      - roneneldan/TinyStories
      - databricks/databricks-dolly-15k
    consequence:
      - no_held_sharealike_or_sharing_source_admitted_for_training
      - training_download_allowed_false_until_distribution_policy_exists
```
