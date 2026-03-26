"""lm-eval wrapper for Dense/EDSR models.

Allows running standard benchmarks (ARC, HellaSwag, PIQA, etc.)
using EleutherAI's lm-evaluation-harness.

Usage:
    python code/lm_eval_wrapper.py --checkpoint results/checkpoints_edsr_98m/rolling_latest.pt
    python code/lm_eval_wrapper.py --checkpoint results/checkpoints_dense_f4/rolling_latest.pt --tasks arc_easy,arc_challenge
"""

import sys, argparse, gc, torch
from pathlib import Path
from tqdm import tqdm

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "code"))

import lm_eval
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


@register_model("sutra")
class SutraLMEval(LM):
    """lm-eval compatible wrapper for Dense/EDSR models."""

    def __init__(self, checkpoint=None, batch_size=4, device="cuda",
                 fp16=False, **kwargs):
        super().__init__()
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._batch_size = batch_size
        self._fp16 = fp16

        from transformers import PreTrainedTokenizerFast
        tok_path = REPO / "data" / "tokenizer_16k" / "tokenizer.json"
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tok_path))
        self.tokenizer.pad_token = self.tokenizer.convert_ids_to_tokens(0)
        self.tokenizer.pad_token_id = 0

        from dense_baseline import DenseTransformer
        ckpt = torch.load(checkpoint, weights_only=False, map_location="cpu")
        cfg = ckpt.get("config", {})
        self.model = DenseTransformer(
            vocab_size=cfg.get("vocab_size", 16000),
            dim=cfg.get("dim", 512),
            n_layers=cfg.get("n_layers", 11),
            n_heads=cfg.get("n_heads", 8),
            ff_dim=cfg.get("ff_dim", 1536),
            exit_layers=cfg.get("exit_layers", None),
        )
        self.model.load_state_dict(ckpt["model"])
        if fp16:
            self.model.half()
        self.model.to(self._device).eval()

    @property
    def eot_token_id(self):
        eos = self.tokenizer.eos_token_id
        return eos if eos is not None else self.tokenizer.pad_token_id or 0

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
        # Pre-encode and sort by length to minimize padding waste
        encoded = []
        for idx, (context, continuation) in enumerate(requests):
            ctx_ids = self.tokenizer.encode(context)
            cont_ids = self.tokenizer.encode(continuation)
            all_ids = (ctx_ids + cont_ids)[-512:]
            ctx_len = len(all_ids) - len(cont_ids)
            encoded.append((idx, all_ids, ctx_len))
        encoded.sort(key=lambda x: len(x[1]))

        results_by_idx = {}
        n_total = len(encoded)
        batch_starts = range(0, n_total, self._batch_size)
        if not disable_tqdm:
            total_batches = (n_total + self._batch_size - 1) // self._batch_size
            batch_starts = tqdm(
                batch_starts,
                total=total_batches,
                desc=f"loglikelihood ({self._device.type})",
            )

        for batch_start in batch_starts:
            batch = encoded[batch_start:batch_start + self._batch_size]
            batch_encoded = [(ids, cl) for _, ids, cl in batch]
            batch_indices = [idx for idx, _, _ in batch]

            # Pad to max length in batch
            max_len = max(len(ids) for ids, _ in batch_encoded)
            pad_id = self.tokenizer.pad_token_id or 0
            padded = torch.full((len(batch_encoded), max_len), pad_id,
                                dtype=torch.long, device=self._device)
            for j, (ids, _) in enumerate(batch_encoded):
                padded[j, :len(ids)] = torch.tensor(ids, device=self._device)

            with torch.inference_mode():
                logits = self.model(padded)
            # Avoid allocating full B*T*V log_probs tensor (saves ~800MB at B=4)
            logits_f = logits.float()
            lse = torch.logsumexp(logits_f, dim=-1)  # (B, T)

            for j, (all_ids, ctx_len) in enumerate(batch_encoded):
                total_ll = 0.0
                is_greedy = True
                for i in range(ctx_len, len(all_ids)):
                    if i > 0 and i - 1 < logits_f.size(1):
                        target_lp = logits_f[j, i - 1, all_ids[i]].item() - lse[j, i - 1].item()
                        total_ll += target_lp
                        if logits_f[j, i - 1].argmax().item() != all_ids[i]:
                            is_greedy = False
                results_by_idx[batch_indices[j]] = (total_ll, is_greedy)
            del logits, logits_f, lse, padded

        return [results_by_idx[i] for i in range(n_total)]

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
            with torch.inference_mode():
                logits = self.model(input_ids)
            logits_f = logits[0].float()
            lse = torch.logsumexp(logits_f, dim=-1)
            total_ll = sum(
                (logits_f[i, ids[i + 1]].item() - lse[i].item())
                for i in range(len(ids) - 1)
            )
            del logits, logits_f, lse
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
                with torch.inference_mode():
                    logits = self.model(input_ids[:, -512:])
                next_id = logits[0, -1].argmax().item()
                input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=self._device)], dim=1)
                decoded = self.tokenizer.decode(input_ids[0, len(ids):].tolist())
                if any(s in decoded for s in (until if isinstance(until, list) else [until])):
                    break

            gen_text = self.tokenizer.decode(input_ids[0, len(ids):].tolist())
            results.append(gen_text)
        return results


if __name__ == "__main__":
    import json, time

    parser = argparse.ArgumentParser(description="lm-eval for Dense/EDSR models")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--tasks", default="arc_easy,arc_challenge,hellaswag,winogrande,piqa,sciq,lambada_openai")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fp16", action="store_true", help="Run inference in FP16 (halves VRAM)")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    out_path = Path(args.output) if args.output else REPO / "results" / "sutra_lm_eval_results.json"

    print(f"Running lm-eval")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Tasks: {args.tasks}")
    print(f"Device: {args.device}")

    model = SutraLMEval(checkpoint=args.checkpoint, batch_size=args.batch_size,
                         device=args.device, fp16=args.fp16)

    task_list = args.tasks.split(",")

    all_results = {}
    if out_path.exists():
        try:
            all_results = json.load(open(out_path))
            print(f"Resuming: {list(all_results.keys())} already done")
        except Exception:
            pass

    t0 = time.time()
    for task_name in task_list:
        if task_name in all_results:
            print(f"\n  {task_name}: SKIPPED (already done)")
            continue

        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            free_mb = (torch.cuda.get_device_properties(0).total_memory
                       - torch.cuda.memory_allocated()) / 1e6
            print(f"\n  VRAM free: {free_mb:.0f}MB", flush=True)

        print(f"Running {task_name}...", flush=True)
        try:
            result = lm_eval.simple_evaluate(
                model=model,
                tasks=[task_name],
                batch_size=args.batch_size,
                limit=args.limit,
                bootstrap_iters=0,
                log_samples=False,
            )
            for k, v in result["results"].items():
                all_results[k] = v
                print(f"  {k}: {v}", flush=True)
            json.dump(all_results, open(out_path, "w"), indent=2, default=str)
            print(f"  Saved after {task_name}", flush=True)
        except Exception as e:
            print(f"  ERROR on {task_name}: {e}", flush=True)

    print(f"\n{'='*60}")
    print(f"RESULTS ({time.time()-t0:.0f}s)")
    print(f"{'='*60}")
    for task_name, task_result in all_results.items():
        if isinstance(task_result, dict):
            acc = task_result.get("acc,none", task_result.get("acc_norm,none", "?"))
            print(f"  {task_name:20s} {acc}")
    print(f"\nSaved: {out_path}")
