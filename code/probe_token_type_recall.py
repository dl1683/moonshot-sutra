"""Token-Type Recall Audit — CPU probe for R3.

Classifies tokens into categories (numbers, proper nouns, function words, code, etc.)
and measures per-category: CE loss, top-1 accuracy, pass disagreement.
Tells us which token types the model struggles with most, informing
the precise-vs-general retrieval spectrum design.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from launch_v060a import SutraV060a
from data_loader import ShardedDataset

FUNC_WORDS = {
    ' the', ' a', ' an', ' is', ' are', ' was', ' were', ' in', ' on', ' at',
    ' to', ' for', ' of', ' and', ' or', ' but', ' not', ' if', ' then',
    ' this', ' that', ' it', ' he', ' she', ' we', ' they', ' my', ' your',
    ' has', ' had', ' have', ' will', ' can', ' do', ' does', ' did',
    ' with', ' from', ' by', ' as', ' be', ' been', ' being',
}


def classify_token(text):
    """Classify a decoded token into categories for recall analysis."""
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


def main():
    print('Loading model on CPU...')
    model = SutraV060a()
    ckpt_path = Path('results/checkpoints_v060a/rolling_latest.pt')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    step = ckpt.get('step', 0)
    print(f'Model loaded at step {step}')

    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained('gpt2')

    ds = ShardedDataset()

    results = defaultdict(lambda: {
        'count': 0, 'total_ce': 0.0,
        'total_pass_disagree': 0.0, 'correct_top1': 0
    })
    n_batches = 8

    print(f'Running token-type recall audit ({n_batches} batches)...')
    with torch.no_grad():
        for batch_idx in range(n_batches):
            tokens, targets = ds.sample_batch(batch_size=4, seq_len=512,
                                              device='cpu', split='test')
            B, T = tokens.shape

            logits, aux = model(tokens, collect_history=True)
            mu_hist = aux.get('mu_hist', None)

            shift_logits = logits[:, :-1].contiguous()
            shift_labels = tokens[:, 1:].contiguous()
            ce_per_token = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1), reduction='none'
            ).view(B, T - 1)

            preds = shift_logits.argmax(dim=-1)
            correct = (preds == shift_labels).float()

            if mu_hist is not None and mu_hist.shape[2] >= 2:
                h_last = mu_hist[:, :, -1, :]
                h_prev = mu_hist[:, :, -2, :]
                cos = F.cosine_similarity(h_last, h_prev, dim=-1)
                disagree = 1.0 - cos[:, :-1]
            else:
                disagree = torch.zeros(B, T - 1)

            for b in range(B):
                for t in range(T - 1):
                    tid = shift_labels[b, t].item()
                    text = tok.decode([tid])
                    cat = classify_token(text)
                    results[cat]['count'] += 1
                    results[cat]['total_ce'] += ce_per_token[b, t].item()
                    results[cat]['total_pass_disagree'] += disagree[b, t].item()
                    results[cat]['correct_top1'] += correct[b, t].item()

            print(f'  Batch {batch_idx + 1}/{n_batches} done')

    print()
    print('=== TOKEN-TYPE RECALL AUDIT ===')
    print(f'Step: {step}, Batches: {n_batches}')
    print()
    hdr = f'{"Category":<18} {"Count":>7} {"Mean CE":>8} {"Top-1 Acc":>10} {"Pass Disagree":>14}'
    print(hdr)
    print('-' * len(hdr))

    summary = {}
    for cat in sorted(results.keys(),
                      key=lambda x: results[x]['total_ce'] / max(results[x]['count'], 1)):
        r = results[cat]
        n = max(r['count'], 1)
        mean_ce = r['total_ce'] / n
        acc = r['correct_top1'] / n
        dis = r['total_pass_disagree'] / n
        print(f'{cat:<18} {r["count"]:>7} {mean_ce:>8.3f} {acc:>10.3f} {dis:>14.4f}')
        summary[cat] = {
            'count': r['count'],
            'mean_ce': round(mean_ce, 4),
            'top1_accuracy': round(acc, 4),
            'mean_pass_disagreement': round(dis, 4),
        }

    output = {
        'step': step,
        'probe': 'token_type_recall_audit',
        'n_batches': n_batches,
        'categories': summary,
    }
    out_path = Path('results/token_type_recall_audit.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nSaved to {out_path}')


if __name__ == '__main__':
    main()
