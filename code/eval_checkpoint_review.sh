#!/bin/bash
# Generation-based eval: THE metric for deciding arch improvements.
# 10 random questions from the 500-question eval set.
# If generations are better -> ship. If not -> don't. All else irrelevant.
#
# Usage: bash code/eval_checkpoint_review.sh [checkpoint] [version]

CKPT=${1:-"results/checkpoints_v053/step_5000.pt"}
VERSION=${2:-"v053"}
REPO="$(cd "$(dirname "$0")/.." && pwd)"

cd "$REPO"
python -u -c "
import sys, json, random, torch, torch.nn.functional as F
sys.path.insert(0, 'code')

random.seed(42)
with open('eval/sutra_eval_500.jsonl', encoding='utf-8') as f:
    questions = [json.loads(line) for line in f]
sample = random.sample(questions, 10)

version = '$VERSION'
if version == 'v054':
    from launch_v054 import create_v054
    model = create_v054(dim=768, ff_dim=1536, max_steps=8, window=4, k_retrieval=8)
else:
    from launch_v053 import create_v053
    model = create_v053(dim=768, ff_dim=1536, max_steps=8, window=4, k_retrieval=8)

ckpt = torch.load('$CKPT', weights_only=False, map_location='cpu')
state = ckpt['model'] if 'model' in ckpt else ckpt
model.load_state_dict(state, strict=False)
model.eval()

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')

print('=== GENERATION EVAL: 10 Random Questions ===')
print(f'Model: {version}, Checkpoint: $CKPT')
print()

for i, q in enumerate(sample):
    prompt = f'Question: {q[\"question\"]}\nAnswer:'
    ids = tokenizer.encode(prompt)[-400:]
    tokens = torch.tensor([ids])
    with torch.no_grad():
        for _ in range(150):
            ctx = tokens[:, -512:] if tokens.size(1) > 512 else tokens
            logits, _ = model(ctx)
            next_logits = logits[:, -1, :].float() / 0.8
            topk_vals, topk_idx = next_logits.topk(30)
            filtered = torch.full_like(next_logits, float('-inf'))
            filtered.scatter_(-1, topk_idx, topk_vals)
            next_token = torch.multinomial(F.softmax(filtered, dim=-1), 1)
            tokens = torch.cat([tokens, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    gen = tokenizer.decode(tokens[0, len(ids):].tolist())
    gen = gen.encode('ascii', errors='replace').decode('ascii')[:300]
    print(f'--- Q{i+1} [{q[\"id\"]}] ({q[\"category\"]}/{q[\"difficulty\"]}) ---')
    print(f'Q: {q[\"question\"][:150]}')
    if q.get('expected_answer'):
        print(f'Expected: {str(q[\"expected_answer\"])[:100]}')
    print(f'Model: {gen}')
    print()
"
