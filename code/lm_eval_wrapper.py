"""lm-eval wrapper for Sutra models.

Allows running standard benchmarks (ARC, HellaSwag, PIQA, etc.) on Sutra
using EleutherAI's lm-evaluation-harness.

Supports both v0.5.4 and v0.6.0a models via --version flag.

Usage:
    python code/lm_eval_wrapper.py --checkpoint results/checkpoints_v060a/rolling_latest.pt --version v060a
    python code/lm_eval_wrapper.py --checkpoint results/checkpoints_v054/step_15000.pt --version v054
    python code/lm_eval_wrapper.py --checkpoint results/checkpoints_v060a/step_20000.pt --tasks arc_easy,arc_challenge
"""

import sys, argparse, torch
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "code"))

import lm_eval
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from transformers import AutoTokenizer


@register_model("sutra")
class SutraLMEval(LM):
    """lm-eval compatible wrapper for Sutra."""

    def __init__(self, checkpoint=None, dim=768, ff_dim=1536,
                 batch_size=8, device="cuda", version="v060a", **kwargs):
        super().__init__()
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model based on version (each version has its own default max_steps)
        if version in ("v060a", "v060b"):
            from launch_v060a import create_v060a
            self.model = create_v060a(dim=dim, ff_dim=ff_dim, max_steps=12,
                                       window=4, k_retrieval=8)
        elif version == "v054":
            from launch_v054 import create_v054
            self.model = create_v054(dim=dim, ff_dim=ff_dim, max_steps=8,
                                      window=4, k_retrieval=8)
        else:
            raise ValueError(f"Unknown version: {version}. Use 'v060a', 'v060b', 'v060c', or 'v054'.")

        if checkpoint:
            ckpt = torch.load(checkpoint, weights_only=False, map_location="cpu")
            state = ckpt["model"] if "model" in ckpt else ckpt
            self.model.load_state_dict(state, strict=False)
        self.model.to(self._device).eval()

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return 512

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string, left_truncate_len=None, add_special_tokens=None):
        encoding = self.tokenizer.encode(string)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        results = []

        # Batch requests by similar length for efficient GPU utilization
        indexed = [(i, req) for i, req in enumerate(requests)]

        for batch_start in range(0, len(indexed), self._batch_size):
            batch = indexed[batch_start:batch_start + self._batch_size]

            # Encode all in batch
            batch_encoded = []
            for _, (context, continuation) in batch:
                ctx_ids = self.tokenizer.encode(context)
                cont_ids = self.tokenizer.encode(continuation)
                all_ids = (ctx_ids + cont_ids)[-512:]
                ctx_len = len(all_ids) - len(cont_ids)
                batch_encoded.append((all_ids, ctx_len))

            # Pad to max length in batch
            max_len = max(len(ids) for ids, _ in batch_encoded)
            padded = torch.full((len(batch_encoded), max_len), self.tokenizer.eos_token_id,
                                dtype=torch.long, device=self._device)
            for j, (ids, _) in enumerate(batch_encoded):
                padded[j, :len(ids)] = torch.tensor(ids, device=self._device)

            # Forward pass — one call for entire batch
            with torch.no_grad():
                logits, _ = self.model(padded)
            log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)

            # Score each item in batch
            for j, (all_ids, ctx_len) in enumerate(batch_encoded):
                lp = log_probs[j]
                total_ll = 0.0
                is_greedy = True
                for i in range(ctx_len, len(all_ids)):
                    if i > 0 and i - 1 < lp.size(0):
                        total_ll += lp[i - 1, all_ids[i]].item()
                        if lp[i - 1].argmax().item() != all_ids[i]:
                            is_greedy = False
                results.append((total_ll, is_greedy))

        return results

    def loglikelihood(self, requests, disable_tqdm=False):
        new_reqs = []
        for req in requests:
            if hasattr(req, 'args'):
                new_reqs.append(req.args)
            else:
                new_reqs.append(req)
        return self._loglikelihood_tokens(new_reqs, disable_tqdm)

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        results = []
        for req in requests:
            string = req.args[0] if hasattr(req, 'args') else req[0]
            ids = self.tokenizer.encode(string)[-512:]
            input_ids = torch.tensor([ids], device=self._device)
            with torch.no_grad():
                logits, _ = self.model(input_ids)
            log_probs = torch.nn.functional.log_softmax(logits[0].float(), dim=-1)
            total_ll = sum(log_probs[i, ids[i + 1]].item() for i in range(len(ids) - 1))
            results.append((total_ll,))
        return results

    def generate_until(self, requests, disable_tqdm=False):
        results = []
        for req in requests:
            context = req.args[0] if hasattr(req, 'args') else req[0]
            until = req.args[1] if hasattr(req, 'args') else req[1]
            if isinstance(until, dict):
                until = until.get("until", ["\n"])

            ids = self.tokenizer.encode(context)[-400:]
            input_ids = torch.tensor([ids], device=self._device)

            for _ in range(self.max_gen_toks):
                with torch.no_grad():
                    logits, _ = self.model(input_ids[:, -512:])
                next_id = logits[0, -1].argmax().item()
                input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=self._device)], dim=1)
                decoded = self.tokenizer.decode(input_ids[0, len(ids):].tolist())
                if any(s in decoded for s in (until if isinstance(until, list) else [until])):
                    break

            gen_text = self.tokenizer.decode(input_ids[0, len(ids):].tolist())
            results.append(gen_text)
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(REPO / "results/checkpoints_v060a/rolling_latest.pt"))
    parser.add_argument("--tasks", default="arc_easy,arc_challenge,hellaswag,winogrande,piqa,sciq,lambada_openai")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--version", default="v060a", choices=["v060a", "v060b", "v060c", "v054", "v061"])
    parser.add_argument("--output", default=None, help="Output JSON path (default: results/sutra_lm_eval_results.json)")
    args = parser.parse_args()

    print(f"Running lm-eval on Sutra {args.version}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Tasks: {args.tasks}")
    print(f"Device: {args.device}")

    model = SutraLMEval(checkpoint=args.checkpoint, batch_size=args.batch_size,
                         device=args.device, version=args.version)

    task_list = args.tasks.split(",")
    results = lm_eval.simple_evaluate(
        model=model,
        tasks=task_list,
        batch_size=args.batch_size,
        bootstrap_iters=0,  # Skip 100K bootstrap resampling — was causing OOM/hang
        log_samples=False,   # Don't hold all samples in memory
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for task_name, task_result in results["results"].items():
        print(f"\n{task_name}:")
        for metric, value in task_result.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")

    import json
    out_path = Path(args.output) if args.output else REPO / "results" / "sutra_lm_eval_results.json"
    json.dump(results["results"], open(out_path, "w"), indent=2, default=str)
    print(f"\nSaved: {out_path}")
