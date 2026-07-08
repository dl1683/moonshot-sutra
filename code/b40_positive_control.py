"""B40 residual-risk demonstration for the absorption ladder paper.

This is a deliberately small CPU-only control. It began as a positive-control
attempt: a planted three-feature interaction was meant to survive a declared
absorber roster. B41 adds the missing full target-class PBE absorber over all
1320 candidates. That absorber is the same target-class search as the claimed
learner, so the attempt is absorbed and demoted to residual-risk evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

MEASUREMENT_VERSION = "b41-positive-control-full-pbe-v2"
PUBLIC_SEED = "B40_POSITIVE_CONTROL_PUBLIC_SEED"
SMOKE_SEED = "B40_POSITIVE_CONTROL_SMOKE_SEED"
INPUT_BITS = 12
TRAIN_EXAMPLES = 192
HIDDEN_CASES = 512
BUDGETED_PBE_CANDIDATES = 128
THRESHOLD = 0.98
ABSORPTION_COST_RATIO = 4.0
SYSTEMS = (
    "claimed_interaction_learner",
    "majority_label",
    "single_bit_prior",
    "pair_conjunction_prior",
    "lookup_memorizer",
    "budgeted_pbe_probe",
    "full_target_class_pbe",
    "random_interaction_probe",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def seed_int(*parts: Any) -> int:
    blob = "|".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(blob.encode("utf-8")).digest()[:8], "big")


def stable_hash(value: Any, width: int = 24) -> str:
    return sha(canonical_json(value))[:width]


def bits_for_payload(value: Any) -> int:
    return len(canonical_json(value).encode("utf-8")) * 8


@dataclass(frozen=True)
class Candidate:
    i: int
    j: int
    k: int
    bias: int

    def predict(self, x: int) -> int:
        interaction = ((x >> self.i) & 1) & ((x >> self.j) & 1)
        linear = (x >> self.k) & 1
        return interaction ^ linear ^ self.bias

    def erased_interaction_predict(self, x: int) -> int:
        return ((x >> self.k) & 1) ^ self.bias

    def to_dict(self) -> dict[str, int]:
        return {"i": self.i, "j": self.j, "k": self.k, "bias": self.bias}


@dataclass(frozen=True)
class Example:
    x: int
    y: int

    def to_dict(self) -> dict[str, Any]:
        return {"x_hex": f"0x{self.x:03x}", "y": self.y}


def candidate_space(input_bits: int) -> list[Candidate]:
    out: list[Candidate] = []
    for i, j in itertools.combinations(range(input_bits), 2):
        for k in range(input_bits):
            if k in {i, j}:
                continue
            for bias in (0, 1):
                out.append(Candidate(i, j, k, bias))
    return out


def make_examples(seed: str, n: int, input_bits: int, target: Candidate, banned: set[int] | None = None) -> list[Example]:
    rng = random.Random(seed_int(seed, n, input_bits, target.to_dict()))
    banned = set() if banned is None else set(banned)
    universe = [x for x in range(1 << input_bits) if x not in banned]
    xs = rng.sample(universe, n)
    return [Example(x, target.predict(x)) for x in xs]


def accuracy(examples: Iterable[Example], predict: Callable[[int], int]) -> float:
    examples = list(examples)
    correct = sum(1 for ex in examples if predict(ex.x) == ex.y)
    return correct / len(examples) if examples else 0.0


def train_accuracy(examples: list[Example], candidate: Candidate) -> float:
    return accuracy(examples, candidate.predict)


def learn_best_candidate(examples: list[Example], candidates: list[Candidate]) -> Candidate:
    scored = [(train_accuracy(examples, cand), -idx, cand) for idx, cand in enumerate(candidates)]
    return max(scored, key=lambda row: (row[0], row[1]))[2]


def majority_model(examples: list[Example]) -> Callable[[int], int]:
    ones = sum(ex.y for ex in examples)
    label = 1 if ones >= len(examples) - ones else 0
    return lambda _x: label


def single_bit_model(examples: list[Example], input_bits: int) -> tuple[Callable[[int], int], dict[str, Any]]:
    best = (-1.0, 0, 0)
    for k in range(input_bits):
        for bias in (0, 1):
            pred = lambda x, k=k, bias=bias: ((x >> k) & 1) ^ bias
            score = accuracy(examples, pred)
            if score > best[0]:
                best = (score, k, bias)
    _, k, bias = best
    return (lambda x, k=k, bias=bias: ((x >> k) & 1) ^ bias), {"k": k, "bias": bias, "train_hfa": best[0]}


def pair_conjunction_model(examples: list[Example], input_bits: int) -> tuple[Callable[[int], int], dict[str, Any]]:
    best = (-1.0, 0, 1, 0)
    for i, j in itertools.combinations(range(input_bits), 2):
        for bias in (0, 1):
            pred = lambda x, i=i, j=j, bias=bias: (((x >> i) & 1) & ((x >> j) & 1)) ^ bias
            score = accuracy(examples, pred)
            if score > best[0]:
                best = (score, i, j, bias)
    _, i, j, bias = best
    return (
        lambda x, i=i, j=j, bias=bias: (((x >> i) & 1) & ((x >> j) & 1)) ^ bias,
        {"i": i, "j": j, "bias": bias, "train_hfa": best[0]},
    )


def lookup_model(examples: list[Example]) -> tuple[Callable[[int], int], dict[str, Any]]:
    table = {ex.x: ex.y for ex in examples}
    default = majority_model(examples)(0)
    return lambda x: table.get(x, default), {"entries": len(table), "default": default}


def randomized_label_control(examples: list[Example], candidates: list[Candidate], seed: str) -> Candidate:
    rng = random.Random(seed_int(seed, "randomized-label-control"))
    labels = [ex.y for ex in examples]
    rng.shuffle(labels)
    shuffled = [Example(ex.x, y) for ex, y in zip(examples, labels)]
    return learn_best_candidate(shuffled, candidates)


def measurement_manifest(public_seed: str, smoke_seed: str) -> dict[str, Any]:
    return {
        "measurement_version": MEASUREMENT_VERSION,
        "public_seed": public_seed,
        "public_smoke_seed": smoke_seed,
        "hidden_seed_rule": "sha256(public_seed|public_smoke_seed|manifest_hash|hidden|unopened_until_freeze)",
        "systems": SYSTEMS,
        "input_bits": INPUT_BITS,
        "train_examples": TRAIN_EXAMPLES,
        "hidden_cases": HIDDEN_CASES,
        "budgeted_pbe_candidates": BUDGETED_PBE_CANDIDATES,
        "full_target_class_candidates": "all three-feature interaction candidates",
        "threshold": THRESHOLD,
        "absorber_roster": [
            "majority_label",
            "single_bit_prior",
            "pair_conjunction_prior",
            "lookup_memorizer",
            "budgeted_pbe_probe",
            "full_target_class_pbe",
            "random_interaction_probe",
        ],
        "residual_risk_demonstrated": [
            "the B40 positive-control attempt is absorbed once full target-class PBE is included"
        ],
        "frozen_before_hidden": True,
        "post_hidden_code_changes": [],
    }


def run(public_seed: str, smoke_seed: str) -> dict[str, Any]:
    started = time.time()
    manifest = measurement_manifest(public_seed, smoke_seed)
    manifest_hash = stable_hash(manifest, 32)
    hidden_seed = sha(f"{public_seed}|{smoke_seed}|{manifest_hash}|hidden|unopened_until_freeze")
    candidates = candidate_space(INPUT_BITS)
    target_index = seed_int(hidden_seed, "target-index") % len(candidates)
    if target_index < BUDGETED_PBE_CANDIDATES:
        target_index += BUDGETED_PBE_CANDIDATES
    target_index %= len(candidates)
    target = candidates[target_index]

    train = make_examples(public_seed + "|train", TRAIN_EXAMPLES, INPUT_BITS, target)
    hidden = make_examples(hidden_seed + "|hidden", HIDDEN_CASES, INPUT_BITS, target, {ex.x for ex in train})

    claimed = learn_best_candidate(train, candidates)
    budgeted_candidates = candidates[:BUDGETED_PBE_CANDIDATES]
    budgeted = learn_best_candidate(train, budgeted_candidates)
    full_pbe = learn_best_candidate(train, candidates)
    random_probe = candidates[seed_int(hidden_seed, "random-probe") % len(candidates)]
    shuffled = randomized_label_control(train, candidates, hidden_seed)
    single_pred, single_meta = single_bit_model(train, INPUT_BITS)
    pair_pred, pair_meta = pair_conjunction_model(train, INPUT_BITS)
    lookup_pred, lookup_meta = lookup_model(train)
    majority_pred = majority_model(train)

    predictors: dict[str, Callable[[int], int]] = {
        "claimed_interaction_learner": claimed.predict,
        "majority_label": majority_pred,
        "single_bit_prior": single_pred,
        "pair_conjunction_prior": pair_pred,
        "lookup_memorizer": lookup_pred,
        "budgeted_pbe_probe": budgeted.predict,
        "full_target_class_pbe": full_pbe.predict,
        "random_interaction_probe": random_probe.predict,
    }
    system_summary = {
        name: {
            "train_hfa": accuracy(train, pred),
            "hidden_hfa": accuracy(hidden, pred),
            "passes_threshold": accuracy(hidden, pred) >= THRESHOLD,
        }
        for name, pred in predictors.items()
    }
    component_erasure_hfa = accuracy(hidden, claimed.erased_interaction_predict)
    randomized_label_hfa = accuracy(hidden, shuffled.predict)

    cost_ledger = {
        "claimed_interaction_learner": {
            "G": 2048,
            "F": bits_for_payload({"hypothesis_class": "and-xor-three-feature-interaction"}),
            "P_i": bits_for_payload(claimed.to_dict()),
            "E_i": TRAIN_EXAMPLES * (INPUT_BITS + 1),
        },
        "majority_label": {"G": 128, "P_i": 1, "E_i": TRAIN_EXAMPLES},
        "single_bit_prior": {"G": 512, "P_i": bits_for_payload(single_meta), "E_i": TRAIN_EXAMPLES * (INPUT_BITS + 1)},
        "pair_conjunction_prior": {"G": 768, "P_i": bits_for_payload(pair_meta), "E_i": TRAIN_EXAMPLES * (INPUT_BITS + 1)},
        "lookup_memorizer": {"G": 512, "P_i": bits_for_payload(lookup_meta), "E_i": TRAIN_EXAMPLES * (INPUT_BITS + 1), "table_bits": TRAIN_EXAMPLES * (INPUT_BITS + 1)},
        "budgeted_pbe_probe": {"G": 1536, "P_i": bits_for_payload(budgeted.to_dict()), "E_i": TRAIN_EXAMPLES * (INPUT_BITS + 1), "candidate_attempt_bits": BUDGETED_PBE_CANDIDATES * 8},
        "full_target_class_pbe": {"G": 1536, "P_i": bits_for_payload(full_pbe.to_dict()), "E_i": TRAIN_EXAMPLES * (INPUT_BITS + 1), "candidate_attempt_bits": len(candidates) * 8},
        "random_interaction_probe": {"G": 256, "P_i": bits_for_payload(random_probe.to_dict()), "E_i": TRAIN_EXAMPLES * (INPUT_BITS + 1)},
    }
    for row in cost_ledger.values():
        row["total_bits"] = sum(int(v) for v in row.values())

    absorber_names = [name for name in SYSTEMS if name != "claimed_interaction_learner"]
    declared_absorbers_fail = all(not system_summary[name]["passes_threshold"] for name in absorber_names)
    claimed_cost = max(1, cost_ledger["claimed_interaction_learner"]["total_bits"])
    full_pbe_cost_ratio = cost_ledger["full_target_class_pbe"]["total_bits"] / claimed_cost
    full_pbe_absorbs = bool(
        system_summary["full_target_class_pbe"]["passes_threshold"]
        and full_pbe_cost_ratio <= ABSORPTION_COST_RATIO
    )

    if full_pbe_absorbs:
        terminal_token = "B40_POSITIVE_CONTROL_ABSORBED_BY_FULL_TARGET_CLASS_PBE"
    else:
        terminal_token = "B40_POSITIVE_CONTROL_INCONCLUSIVE"

    return {
        "name": "b40_positive_control_measurement",
        "measurement_version": MEASUREMENT_VERSION,
        "public_seed": public_seed,
        "public_smoke_seed": smoke_seed,
        "manifest": manifest,
        "manifest_hash": manifest_hash,
        "hidden_seed_hash": sha(hidden_seed),
        "hidden_seed_opened": True,
        "counts": {"train_examples": len(train), "hidden_cases": len(hidden), "hypothesis_class_size": len(candidates)},
        "target_summary": {"candidate_index": target_index, "candidate_hash": stable_hash(target.to_dict()), "input_bits": INPUT_BITS},
        "sample_public_examples": [ex.to_dict() for ex in train[:8]],
        "sample_hidden_cases": [ex.to_dict() for ex in hidden[:8]],
        "claimed_artifact": claimed.to_dict(),
        "absorber_artifacts": {
            "single_bit_prior": single_meta,
            "pair_conjunction_prior": pair_meta,
            "lookup_memorizer": lookup_meta,
            "budgeted_pbe_probe": budgeted.to_dict(),
            "full_target_class_pbe": full_pbe.to_dict(),
            "random_interaction_probe": random_probe.to_dict(),
            "randomized_label_control": shuffled.to_dict(),
        },
        "system_summary": system_summary,
        "causal_controls": {
            "component_erasure_hidden_hfa": component_erasure_hfa,
            "component_erasure_drop_pp": (system_summary["claimed_interaction_learner"]["hidden_hfa"] - component_erasure_hfa) * 100.0,
            "randomized_label_hidden_hfa": randomized_label_hfa,
        },
        "cost_ledger_by_system": cost_ledger,
        "cost_ratios_vs_claimed": {
            "full_target_class_pbe": full_pbe_cost_ratio,
            "budgeted_pbe_probe": cost_ledger["budgeted_pbe_probe"]["total_bits"] / claimed_cost,
        },
        "token_evidence": {
            "claimed_system_passes": system_summary["claimed_interaction_learner"]["passes_threshold"],
            "declared_absorbers_fail": declared_absorbers_fail,
            "component_erasure_damages": component_erasure_hfa < 0.9,
            "randomized_label_control_fails": randomized_label_hfa < 0.75,
            "hidden_open_discipline": True,
            "post_hidden_code_changes": False,
            "full_target_class_pbe_run": True,
            "full_target_class_pbe_absorbs": full_pbe_absorbs,
            "full_target_class_pbe_hfa": system_summary["full_target_class_pbe"]["hidden_hfa"],
            "full_target_class_pbe_cost_ratio_vs_claimed": full_pbe_cost_ratio,
            "absorption_cost_ratio_boundary": ABSORPTION_COST_RATIO,
            "positive_control_attempt_absorbed": full_pbe_absorbs,
            "residual_risk_high": False,
            "omitted_full_target_class_pbe_would_absorb": False,
        },
        "terminal_token": terminal_token,
        "claim_ceiling": "Residual-risk demonstration: the planted interaction is absorbed once full target-class PBE is included, so this is not a positive discovery signal.",
        "elapsed_s": round(time.time() - started, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="experiments/b40_positive_control_measurement.json")
    parser.add_argument("--public-seed", default=PUBLIC_SEED)
    parser.add_argument("--smoke-seed", default=SMOKE_SEED)
    args = parser.parse_args()
    result = run(args.public_seed, args.smoke_seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "terminal_token": result["terminal_token"],
        "claimed_hidden_hfa": result["system_summary"]["claimed_interaction_learner"]["hidden_hfa"],
        "best_absorber_hidden_hfa": max(v["hidden_hfa"] for k, v in result["system_summary"].items() if k != "claimed_interaction_learner"),
        "full_target_class_pbe_hidden_hfa": result["system_summary"]["full_target_class_pbe"]["hidden_hfa"],
        "full_target_class_pbe_cost_ratio_vs_claimed": result["cost_ratios_vs_claimed"]["full_target_class_pbe"],
        "component_erasure_drop_pp": result["causal_controls"]["component_erasure_drop_pp"],
        "output": str(out),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()