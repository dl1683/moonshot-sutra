"""Run lm-eval tasks one at a time to avoid the aggregation hang.

Usage:
    python code/run_lm_eval_sequential.py --version v061 --checkpoint results/checkpoints_v061/step_1000.pt --output results/sutra_lm_eval_results_v061_step1000.json
    python code/run_lm_eval_sequential.py --version v054 --checkpoint results/checkpoints_v054/step_20000.pt --output results/sutra_lm_eval_results_v054_step20k.json
"""

import sys, json, argparse, gc, torch
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "code"))

import lm_eval
from lm_eval_wrapper import SutraLMEval

TASKS = ["arc_easy", "arc_challenge", "hellaswag", "winogrande", "piqa", "sciq", "lambada_openai"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--version", required=True, choices=["v060a", "v060b", "v060c", "v054", "v061"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    print(f"Loading {args.version} from {args.checkpoint} on {args.device}...")
    model = SutraLMEval(checkpoint=args.checkpoint, batch_size=args.batch_size,
                         device=args.device, version=args.version)

    all_results = {}
    for task_name in TASKS:
        print(f"Evaluating {task_name}...")
        try:
            result = lm_eval.simple_evaluate(
                model=model,
                tasks=[task_name],
                batch_size=args.batch_size,
                bootstrap_iters=0,
                log_samples=False,
            )
            task_result = result["results"].get(task_name, {})
            all_results[task_name] = task_result
            # Print key metric
            acc = task_result.get("acc,none", task_result.get("acc", "N/A"))
            print(f"  {task_name}: {acc}")
        except Exception as e:
            print(f"  ERROR on {task_name}: {e}")
            all_results[task_name] = {"error": str(e)}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save
    out_path = Path(args.output)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for task_name, result in all_results.items():
        acc = result.get("acc,none", result.get("acc", "N/A"))
        print(f"  {task_name}: {acc}")


if __name__ == "__main__":
    main()
