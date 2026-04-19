"""Probe: side-by-side generation across session's 3 checkpoints.

3 checkpoints have similar mean BPB. Per §12.28, they trade off across
byte classes. Per §12.29, late-window BPB is much lower than start. Generation
quality at meaningful prompts may show differences not visible in the metrics.
"""
import sys
import math
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from sutra_dyad import SutraDyadS1, DEVICE

CHECKPOINTS = [
    ("diagnostic", "results/checkpoints_diagnostic_classical_kd/best.pt"),
    ("r2", "results/checkpoints_classical_continue_r2/best.pt"),
    ("rbor_v1b", "results/checkpoints_rbor_v1b/best.pt"),
]

PROMPTS = [
    b"The meaning of intelligence is",
    b"In the year 2050, the world",
    b"def fibonacci(n):\n    if n <= 1:\n        return n\n    return ",
    b"The Pythagorean theorem states that",
    b"Once upon a time, in a small village,",
    b"Q: What is the capital of France?\nA:",
    b"E = mc^2 means that",
    b"const factorial = (n) => n <= 1 ? 1 :",
]
MAX_NEW_BYTES = 200
TEMPERATURE = 0.7

results = {}

for name, path in CHECKPOINTS:
    print(f"\n{'='*70}")
    print(f"Checkpoint: {name}  ({path})")
    print(f"{'='*70}")
    if not Path(path).exists():
        print("  SKIPPED")
        continue

    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg = ckpt.get("config", {})
    seq_bytes = cfg.get("seq_bytes", 1536)

    model = SutraDyadS1(max_seq_bytes=seq_bytes)
    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(DEVICE)
    model.eval()

    gens = []
    for prompt in PROMPTS:
        t0 = time.time()
        gen_ids = model.generate(list(prompt), max_new_bytes=MAX_NEW_BYTES, temperature=TEMPERATURE)
        elapsed = time.time() - t0
        # Skip the prompt bytes in output — generate may or may not include them
        full_text = bytes(gen_ids).decode('utf-8', errors='replace')
        if full_text.startswith(prompt.decode('utf-8', errors='replace')):
            generated = full_text[len(prompt):]
        else:
            generated = full_text
        # ASCII-safe display
        display = generated.encode('ascii', errors='replace').decode('ascii')
        prompt_disp = prompt.decode('utf-8', errors='replace').encode('ascii', errors='replace').decode('ascii')
        print(f"\nPROMPT: {prompt_disp!r}")
        print(f"  GEN ({elapsed:.1f}s): {display!r}")
        gens.append({
            "prompt": prompt.decode('utf-8', errors='replace'),
            "generated": generated,
            "elapsed_s": elapsed,
        })

    results[name] = {"gens": gens}
    del model
    torch.cuda.empty_cache()

import json
Path("results/probe_generation_compare.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
print(f"\nSaved to results/probe_generation_compare.json")
