"""kNN-LM Ceiling Probe -- Codex R4 Request #1.

Tests how much upside exact memory has before any retraining.
Builds a FAISS datastore from training data hidden states, then
interpolates kNN distribution with model logits at test time.

Measures per-token-type improvement to validate the retrieval spectrum thesis:
if exact retrieval lifts numbers/entities/acronyms significantly, the spectrum
is load-bearing. If it barely helps, the thesis weakens.

Runs on CPU only. Does NOT touch GPU.
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from launch_v060a import SutraV060a
from data_loader import ShardedDataset

REPO = Path(__file__).parent.parent

# Token classification (same as probe_token_type_recall.py)
FUNC_WORDS = {
    ' the', ' a', ' an', ' is', ' are', ' was', ' were', ' in', ' on', ' at',
    ' to', ' for', ' of', ' and', ' or', ' but', ' not', ' if', ' then',
    ' this', ' that', ' it', ' he', ' she', ' we', ' they', ' my', ' your',
    ' has', ' had', ' have', ' will', ' can', ' do', ' does', ' did',
    ' with', ' from', ' by', ' as', ' be', ' been', ' being',
}


def classify_token(text):
    t = text.strip()
    if not t:
        return 'whitespace'
    if t.isdigit() or (len(t) > 1 and t[0] == '-' and t[1:].isdigit()):
        return 'number'
    if t.isupper() and len(t) > 1:
        return 'acronym'
    if len(t) > 1 and t[0].isupper() and not text.startswith(' '):
        return 'proper_noun'
    if any(c in t for c in '{}()[]<>;:=+*/\\|@#$%^&~`'):
        return 'code_symbol'
    if t.startswith('_') or (len(t) > 2 and '_' in t):
        return 'code_identifier'
    if text.lower() in FUNC_WORDS:
        return 'function_word'
    return 'content_word'


def build_datastore(model, ds, tokenizer, n_tokens=262144, seq_len=512, batch_size=4):
    """Extract hidden states from training data to build kNN datastore.

    Returns:
        keys: (N, D) numpy array of L2-normalized hidden states
        vals: (N,) numpy array of next-token IDs
    """
    print(f'Building datastore from ~{n_tokens} training tokens...')
    all_keys = []
    all_vals = []
    collected = 0
    batch_idx = 0

    with torch.no_grad():
        while collected < n_tokens:
            tokens, targets = ds.sample_batch(
                batch_size=batch_size, seq_len=seq_len,
                device='cpu', split='train'
            )
            B, T = tokens.shape

            # Get final-pass hidden states
            logits, aux = model(tokens, collect_history=True)
            mu_hist = aux.get('mu_hist', None)
            if mu_hist is None:
                print('ERROR: mu_hist is None. Model not returning history.')
                return None, None

            # Final pass hidden states: (B, T, D)
            h_final = mu_hist[:, :, -1, :]

            # Use positions 0..T-2 as keys, tokens 1..T-1 as values
            keys = h_final[:, :-1, :].reshape(-1, h_final.shape[-1])  # (B*(T-1), D)
            vals = tokens[:, 1:].reshape(-1)  # (B*(T-1),)

            # L2 normalize for cosine similarity via inner product
            keys_np = keys.float().numpy()
            norms = np.linalg.norm(keys_np, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            keys_np = keys_np / norms

            all_keys.append(keys_np)
            all_vals.append(vals.numpy())
            collected += keys_np.shape[0]
            batch_idx += 1

            if batch_idx % 10 == 0:
                print(f'  {collected}/{n_tokens} tokens collected ({batch_idx} batches)')

    keys = np.concatenate(all_keys, axis=0)[:n_tokens]
    vals = np.concatenate(all_vals, axis=0)[:n_tokens]
    print(f'Datastore built: {keys.shape[0]} entries, dim={keys.shape[1]}')
    return keys, vals


def knn_search_numpy(query, keys, k=64):
    """Brute-force kNN search using numpy (no FAISS dependency).

    Args:
        query: (Q, D) normalized query vectors
        keys: (N, D) normalized key vectors
        k: number of neighbors

    Returns:
        distances: (Q, k) cosine similarities
        indices: (Q, k) neighbor indices
    """
    # Cosine similarity = dot product of normalized vectors
    # Process in chunks to limit memory
    chunk_size = 32
    all_dists = []
    all_idxs = []

    for i in range(0, query.shape[0], chunk_size):
        q_chunk = query[i:i + chunk_size]  # (chunk, D)
        sims = q_chunk @ keys.T  # (chunk, N)
        topk_idx = np.argpartition(-sims, k, axis=1)[:, :k]  # (chunk, k)
        topk_sims = np.take_along_axis(sims, topk_idx, axis=1)

        # Sort within top-k
        sort_idx = np.argsort(-topk_sims, axis=1)
        topk_idx = np.take_along_axis(topk_idx, sort_idx, axis=1)
        topk_sims = np.take_along_axis(topk_sims, sort_idx, axis=1)

        all_dists.append(topk_sims)
        all_idxs.append(topk_idx)

    return np.concatenate(all_dists, axis=0), np.concatenate(all_idxs, axis=0)


def knn_distribution(distances, indices, vals, vocab_size, temperature=10.0):
    """Convert kNN results to a probability distribution.

    Args:
        distances: (Q, k) cosine similarities
        indices: (Q, k) neighbor indices
        vals: (N,) next-token IDs for all datastore entries
        vocab_size: vocabulary size
        temperature: softmax temperature for kNN distribution

    Returns:
        knn_probs: (Q, V) probability distribution over vocab
    """
    Q, k = distances.shape
    neighbor_tokens = vals[indices]  # (Q, k)

    # Softmax over scaled distances
    scaled = distances * temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)  # numerical stability
    weights = np.exp(scaled)
    weights = weights / weights.sum(axis=1, keepdims=True)  # (Q, k)

    # Scatter into vocab distribution
    knn_probs = np.zeros((Q, vocab_size), dtype=np.float32)
    np.add.at(knn_probs, (np.arange(Q)[:, None], neighbor_tokens), weights)

    return knn_probs


def evaluate_with_knn(model, ds, tokenizer, ds_keys, ds_vals,
                      n_eval_batches=8, seq_len=512, batch_size=2,
                      k=64, temperature=10.0, lambdas=(0.0, 0.1, 0.2, 0.3, 0.5)):
    """Evaluate model with kNN interpolation on test data.

    For each lambda value, computes:
        P(w) = (1-lambda) * P_model(w) + lambda * P_knn(w)

    Returns per-token-type CE at each lambda.
    """
    vocab_size = 50257  # GPT-2

    # Results: {lambda: {category: {count, total_ce, correct_top1}}}
    results = {lam: defaultdict(lambda: {'count': 0, 'total_ce': 0.0, 'correct_top1': 0})
               for lam in lambdas}

    print(f'Evaluating with kNN (k={k}, temp={temperature}, lambdas={lambdas})...')
    t0 = time.time()

    with torch.no_grad():
        for batch_idx in range(n_eval_batches):
            tokens, targets = ds.sample_batch(
                batch_size=batch_size, seq_len=seq_len,
                device='cpu', split='test'
            )
            B, T = tokens.shape

            # Model forward
            logits, aux = model(tokens, collect_history=True)
            mu_hist = aux['mu_hist']

            # Model probabilities: shift by 1
            shift_logits = logits[:, :-1, :]  # (B, T-1, V)
            shift_labels = tokens[:, 1:]  # (B, T-1)
            model_log_probs = F.log_softmax(shift_logits, dim=-1)
            model_probs = model_log_probs.exp().numpy()

            # Hidden states for kNN query
            h_final = mu_hist[:, :-1, -1, :]  # (B, T-1, D)

            for b in range(B):
                # Normalize query vectors
                q = h_final[b].float().numpy()  # (T-1, D)
                norms = np.linalg.norm(q, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-8)
                q = q / norms

                # kNN search
                dists, idxs = knn_search_numpy(q, ds_keys, k=k)

                # kNN distribution
                knn_probs = knn_distribution(dists, idxs, ds_vals, vocab_size, temperature)

                labels_b = shift_labels[b].numpy()  # (T-1,)
                model_probs_b = model_probs[b]  # (T-1, V)

                for t_pos in range(T - 1):
                    tid = labels_b[t_pos]
                    text = tokenizer.decode([tid])
                    cat = classify_token(text)

                    for lam in lambdas:
                        if lam == 0.0:
                            p = model_probs_b[t_pos]
                        else:
                            p = (1.0 - lam) * model_probs_b[t_pos] + lam * knn_probs[t_pos]

                        p_target = max(p[tid], 1e-10)
                        ce = -np.log(p_target)
                        pred = np.argmax(p)

                        results[lam][cat]['count'] += 1
                        results[lam][cat]['total_ce'] += float(ce)
                        results[lam][cat]['correct_top1'] += int(pred == tid)

            elapsed = time.time() - t0
            print(f'  Batch {batch_idx + 1}/{n_eval_batches} done ({elapsed:.0f}s)')

    return results


def main():
    print('=== kNN-LM CEILING PROBE (Codex R4 Request #1) ===')
    print('Purpose: Test how much upside exact memory has before retraining.')
    print()

    # Load model
    print('Loading model on CPU...')
    model = SutraV060a()
    ckpt_path = REPO / 'results' / 'checkpoints_v060a' / 'rolling_latest.pt'
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    step = ckpt.get('step', 0)
    print(f'Model loaded at step {step}')

    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained('gpt2')

    ds = ShardedDataset()

    # Phase 1: Build datastore from training data
    # Use 128K tokens (conservative for CPU RAM)
    ds_keys, ds_vals = build_datastore(
        model, ds, tok,
        n_tokens=131072,  # 128K tokens
        seq_len=512,
        batch_size=4
    )
    if ds_keys is None:
        print('FATAL: Could not build datastore')
        return

    # Phase 2: Evaluate with kNN interpolation on test data
    results = evaluate_with_knn(
        model, ds, tok, ds_keys, ds_vals,
        n_eval_batches=8,
        seq_len=512,
        batch_size=2,
        k=64,
        temperature=10.0,
        lambdas=[0.0, 0.1, 0.2, 0.3, 0.5]
    )

    # Phase 3: Report
    print()
    print('=' * 80)
    print(f'kNN-LM CEILING PROBE RESULTS (step {step}, datastore={ds_keys.shape[0]} tokens)')
    print('=' * 80)

    summary = {}
    for lam in sorted(results.keys()):
        print(f'\n--- lambda={lam:.1f} {"(baseline)" if lam == 0.0 else ""} ---')
        cats = results[lam]
        hdr = f'{"Category":<18} {"Count":>7} {"Mean CE":>8} {"Top-1 Acc":>10}'
        print(hdr)
        print('-' * len(hdr))

        lam_summary = {}
        for cat in sorted(cats.keys(),
                          key=lambda x: cats[x]['total_ce'] / max(cats[x]['count'], 1)):
            r = cats[cat]
            n = max(r['count'], 1)
            mean_ce = r['total_ce'] / n
            acc = r['correct_top1'] / n
            print(f'{cat:<18} {r["count"]:>7} {mean_ce:>8.3f} {acc:>10.3f}')
            lam_summary[cat] = {
                'count': r['count'],
                'mean_ce': round(mean_ce, 4),
                'top1_accuracy': round(acc, 4),
            }
        summary[str(lam)] = lam_summary

    # Compute deltas vs baseline
    print('\n' + '=' * 80)
    print('DELTA vs BASELINE (lambda=0.0)')
    print('=' * 80)
    baseline = results[0.0]
    best_lambda_per_cat = {}

    for cat in sorted(baseline.keys()):
        n = max(baseline[cat]['count'], 1)
        base_ce = baseline[cat]['total_ce'] / n
        print(f'\n{cat} (n={baseline[cat]["count"]}, base_ce={base_ce:.3f}):')

        best_lam = 0.0
        best_delta = 0.0
        for lam in sorted(results.keys()):
            if lam == 0.0:
                continue
            r = results[lam][cat]
            n2 = max(r['count'], 1)
            lam_ce = r['total_ce'] / n2
            delta_pct = (lam_ce - base_ce) / base_ce * 100
            marker = ' <-- BEST' if delta_pct < best_delta else ''
            if delta_pct < best_delta:
                best_delta = delta_pct
                best_lam = lam
            print(f'  lambda={lam:.1f}: CE={lam_ce:.3f} ({delta_pct:+.1f}%){marker}')

        best_lambda_per_cat[cat] = {'best_lambda': best_lam, 'best_delta_pct': round(best_delta, 2)}

    # Summary verdict
    print('\n' + '=' * 80)
    print('VERDICT')
    print('=' * 80)
    retrieval_targets = ['number', 'proper_noun', 'acronym', 'code_identifier']
    general_cats = ['function_word', 'content_word', 'whitespace']

    target_improvements = []
    general_impacts = []
    for cat, info in best_lambda_per_cat.items():
        if cat in retrieval_targets:
            target_improvements.append((cat, info['best_delta_pct']))
        elif cat in general_cats:
            general_impacts.append((cat, info['best_delta_pct']))

    print('Retrieval-target categories (numbers, names, acronyms, code):')
    for cat, delta in target_improvements:
        print(f'  {cat}: {delta:+.1f}% CE at best lambda')

    print('General categories (function words, content, whitespace):')
    for cat, delta in general_impacts:
        print(f'  {cat}: {delta:+.1f}% CE at best lambda')

    avg_target = np.mean([d for _, d in target_improvements]) if target_improvements else 0
    avg_general = np.mean([d for _, d in general_impacts]) if general_impacts else 0
    print(f'\nAvg retrieval-target improvement: {avg_target:+.1f}%')
    print(f'Avg general impact: {avg_general:+.1f}%')

    if avg_target < -5.0:
        verdict = 'STRONG POSITIVE: Exact memory has significant upside for hard tokens'
    elif avg_target < -2.0:
        verdict = 'MODERATE POSITIVE: Exact memory helps, but not transformative alone'
    elif avg_target < 0.0:
        verdict = 'WEAK POSITIVE: Marginal improvement, spectrum thesis needs more evidence'
    else:
        verdict = 'NEGATIVE: Exact memory does not help. Retrieval spectrum thesis weakened.'

    print(f'\nVERDICT: {verdict}')

    # Save results
    output = {
        'probe': 'knn_lm_ceiling',
        'step': step,
        'datastore_size': int(ds_keys.shape[0]),
        'k': 64,
        'temperature': 10.0,
        'per_lambda': summary,
        'best_lambda_per_category': best_lambda_per_cat,
        'avg_retrieval_target_delta': round(avg_target, 2),
        'avg_general_delta': round(avg_general, 2),
        'verdict': verdict,
    }
    out_path = REPO / 'results' / 'probe_knn_ceiling.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nSaved to {out_path}')


if __name__ == '__main__':
    main()
